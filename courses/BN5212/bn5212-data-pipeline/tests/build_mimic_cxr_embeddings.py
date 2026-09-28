"""Build reusable VGG16 embeddings from the local MIMIC-CXR dataset.

Output: mimic_cxr_vgg16_embeddings.pt
The file contains embeddings, weak labels, and the source image/report names.
"""

import io
import os
import re
import zipfile

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ROOT = r"E:\School\# Doctor\BN5212\BN5212"
ZIP_PATH = os.path.join(ROOT, "MIMIC-CXR", "dataset.zip")
OUTPUT_PATH = os.path.join(ROOT, "mimic_cxr_vgg16_embeddings.pt")


def report_label(report: str):
    text = re.sub(r"\s+", " ", report.lower())
    negative = [
        "no pneumonia",
        "without pneumonia",
        "negative for pneumonia",
        "no focal consolidation",
        "no acute cardiopulmonary abnormality",
        "no acute cardiopulmonary disease",
        "lungs are clear",
    ]
    positive = [
        "pneumonia",
        "focal consolidation",
        "airspace opacity",
        "airspace opacities",
        "pulmonary infiltrate",
        "infiltrates",
    ]
    if any(term in text for term in negative):
        return 0
    if any(term in text for term in positive):
        return 1
    return None


def build_index(zip_path):
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for report_name in names:
            if not (report_name.startswith("dataset/") and report_name.endswith(".txt")):
                continue
            report = archive.read(report_name).decode("utf-8", errors="replace")
            label = report_label(report)
            if label is None:
                continue
            study_dir = report_name[:-4] + "/"
            for image_name in names:
                if image_name.startswith(study_dir) and image_name.endswith(".dcm"):
                    rows.append({"image_name": image_name, "report_name": report_name, "label": label})
    return rows


class MimicCXRImages(Dataset):
    def __init__(self, zip_path, rows, transform):
        self.zip_path = zip_path
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with zipfile.ZipFile(self.zip_path) as archive:
            raw = archive.read(row["image_name"])
        dicom = pydicom.dcmread(io.BytesIO(raw))
        pixels = dicom.pixel_array.astype(np.float32)
        if getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
            pixels = pixels.max() - pixels
        pixels -= pixels.min()
        if pixels.max() > 0:
            pixels /= pixels.max()
        image = Image.fromarray((pixels * 255).astype(np.uint8)).convert("RGB")
        return self.transform(image), index


def main():
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError(ZIP_PATH)

    rows = build_index(ZIP_PATH)
    if not rows:
        raise RuntimeError("No labelled MIMIC-CXR images were found.")
    print(f"Using {len(rows)} labelled images")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loader = DataLoader(MimicCXRImages(ZIP_PATH, rows, transform), batch_size=16, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = models.VGG16_Weights.DEFAULT
    backbone = models.vgg16(weights=weights).to(device).eval()
    feature_extractor = torch.nn.Sequential(*list(backbone.classifier.children())[:-1])

    vectors = []
    metadata = []
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device)
            features = feature_extractor(backbone.avgpool(backbone.features(images)).flatten(1))
            vectors.append(features.cpu())
            metadata.extend(rows[i] for i in indices.tolist())

    package = {
        "embeddings": torch.cat(vectors).float(),
        "labels": torch.tensor([row["label"] for row in metadata], dtype=torch.long),
        "image_names": [row["image_name"] for row in metadata],
        "report_names": [row["report_name"] for row in metadata],
        "model": "VGG16 ImageNet pretrained classifier features, 4096 dimensions",
        "label_definition": "weak labels generated from report keywords; 0=negative, 1=positive",
        "source_zip": ZIP_PATH,
    }
    torch.save(package, OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Shape: {tuple(package['embeddings'].shape)}")


if __name__ == "__main__":
    main()
