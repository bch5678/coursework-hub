#!/usr/bin/env python3
"""Create entirely artificial records and gradient images for offline verification."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def write_dicom(path, subject, date, time, view="AP", invert=False):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.PatientID = "p" + str(subject); ds.PatientName = "SYNTHETIC^ONLY"
    ds.StudyDate = date; ds.StudyTime = time; ds.ViewPosition = view
    ds.Rows = 48; ds.Columns = 64; ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME1" if invert else "MONOCHROME2"
    ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
    ds.RescaleIntercept = -100; ds.RescaleSlope = 1
    pixels = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    ds.PixelData = pixels.tobytes()
    ds.save_as(path, enforce_file_format=True)


def make_synthetic(root, image_format="png", n_subjects=36):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Synthetic fixture destination must be empty")
    hosp = root / "raw" / "mimiciv" / "hosp"
    images = root / "raw" / "cxr"
    hosp.mkdir(parents=True); images.mkdir(parents=True)
    patients, admissions, metadata = [], [], []

    def image_row(subject, study, dicom, stamp, view="AP", exists=True, corrupt=False):
        ext = ".dcm" if image_format == "dicom" else ".png"
        relative = f"files/p{str(subject)[:2]}/p{subject}/s{study}/{dicom}{ext}"
        path = images / relative
        date, time = stamp.strftime("%Y%m%d"), stamp.strftime("%H%M%S") + ".125000"
        if exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            if corrupt:
                path.write_bytes(b"synthetic corrupt image")
            elif image_format == "dicom":
                write_dicom(path, subject, date, time, view, int(subject) % 2 == 0)
            else:
                pixels = (np.arange(48 * 64).reshape(48, 64) % 256).astype(np.uint8)
                Image.fromarray(pixels).save(path)
        metadata.append({"subject_id": str(subject), "study_id": str(study), "dicom_id": dicom,
                         "StudyDate": date, "StudyTime": time, "ViewPosition": view, "image_path": relative})

    for i in range(n_subjects):
        subject = 90000000 + i
        patients.append({"subject_id": str(subject), "anchor_age": "17" if i == n_subjects - 1 else str(30 + i), "anchor_year": "2145", "gender": "F" if i % 2 else "M"})
        for visit in range(2):
            hadm = 80000000 + i * 10 + visit
            admission = datetime(2145, 1, 1, 0, 0, 0) + timedelta(days=visit * 30)
            death = admission + timedelta(hours=60) if i % 4 == 0 and visit == 1 else None
            flag = "1" if death else "0"
            if i == 1 and visit == 1:
                flag = ""  # No negative-label imputation.
            if i == 2 and visit == 1:
                death = admission + timedelta(hours=60)  # Contradiction with flag=0.
            admissions.append({"subject_id": str(subject), "hadm_id": str(hadm), "admittime": admission.strftime("%Y-%m-%d %H:%M:%S"),
                               "dischtime": (admission + timedelta(hours=96)).strftime("%Y-%m-%d %H:%M:%S"),
                               "deathtime": death.strftime("%Y-%m-%d %H:%M:%S") if death else "", "hospital_expire_flag": flag,
                               "custom_outcome": "yes" if i % 3 == 0 else "no"})
            for k, (hour, view) in enumerate([(6, "AP"), (12, "PA"), (6, "LATERAL"), (72, "AP")]):
                image_row(subject, 70000000 + i * 100 + visit * 10 + k, f"synthetic-{i:03d}-{visit}-{k}", admission + timedelta(hours=hour), view)
    # Extra records cover missing timestamps, absent subset images and corrupt files.
    image_row(90000000, 79999001, "synthetic-before", datetime(2144, 12, 31, 20))
    image_row(90000000, 79999002, "synthetic-missing", datetime(2145, 1, 1, 3), exists=False)
    image_row(90000000, 79999003, "synthetic-corrupt", datetime(2145, 1, 1, 2), corrupt=True)
    image_row(90009999, 79999004, "synthetic-unmatched", datetime(2145, 1, 1, 6))
    if image_format != "dicom":
        image_row(90000000, 79999005, "synthetic-no-time", datetime(2145, 1, 1, 4))
        metadata[-1]["StudyTime"] = ""
        metadata.append(dict(metadata[0]))
    admissions.append(dict(admissions[0]))
    pd.DataFrame(patients).to_csv(hosp / "patients.csv.gz", index=False)
    pd.DataFrame(admissions).to_csv(hosp / "admissions.csv.gz", index=False)
    metadata_path = root / "metadata.csv.gz"
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)
    default_path = Path(__file__).resolve().parents[1] / "config" / "default.json"
    cfg = json.loads(default_path.read_text(encoding="utf-8"))
    cfg["paths"] = {"mimic_iv_root": "raw/mimiciv", "image_root": "raw/cxr", "cxr_metadata": None if image_format == "dicom" else "metadata.csv.gz", "output_dir": "processed"}
    cfg["loader"]["image_size"] = 32
    cfg["loader"]["batch_size"] = 4
    cfg["audit"]["hash_images"] = True
    config_path = root / "synthetic_config.json"
    config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (root / "SYNTHETIC_ONLY.txt").write_text("Generated artificial data. Not real MIMIC records or clinical images.\n", encoding="utf-8")
    return config_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["png", "dicom"], default="png")
    args = parser.parse_args()
    print(make_synthetic(args.output, args.format))
