# MIMIC-CXR VGG16 Embeddings

`build_mimic_cxr_embeddings.py` 从本地 MIMIC-CXR 数据集中提取胸片的 VGG16 特征，并根据放射报告的关键词为每张图打上肺炎阴性/阳性弱标签，打包保存成一个 `.pt` 文件，方便后续直接做分类实验，不用每次重新读 DICOM。

## 环境

- Python：`D:\Users\Bai Chenhao\AppData\Local\Programs\Python\Python312\python.exe`
- 依赖：`torch`、`torchvision`、`numpy`、`Pillow`、`pydicom`（已安装）
- 有 GPU 时自动使用 CUDA，没有则用 CPU

## 输入

脚本直接读取 ZIP，不需要解压：

```
E:\School\# Doctor\BN5212\BN5212\MIMIC-CXR\dataset.zip
```

ZIP 内部结构为 `dataset/p<患者ID>/s<检查ID>.txt`（报告）和 `dataset/p<患者ID>/s<检查ID>/*.dcm`（该次检查的图像）。路径写死在脚本开头的 `ROOT` / `ZIP_PATH` / `OUTPUT_PATH`，换位置时改这里即可。

## 运行

```powershell
& "D:\Users\Bai Chenhao\AppData\Local\Programs\Python\Python312\python.exe" "E:\School\# Doctor\BN5212\build_mimic_cxr_embeddings.py"
```

路径里有空格和 `#`，必须加引号和 `&`。运行期间没有进度条，终端里首先打印 `Using N labelled images`，结束时打印保存路径和特征形状。耗时主要在逐张解码 DICOM 上。

正常输出：

```
Using 3380 labelled images
Saved: E:\School\# Doctor\BN5212\BN5212\mimic_cxr_vgg16_embeddings.pt
Shape: (3380, 4096)
```

## 输出文件

`E:\School\# Doctor\BN5212\BN5212\mimic_cxr_vgg16_embeddings.pt` 是一个字典：

| 键 | 类型 | 内容 |
| --- | --- | --- |
| `embeddings` | `FloatTensor [N, 4096]` | 每张图的 VGG16 特征 |
| `labels` | `LongTensor [N]` | 0 = 阴性，1 = 阳性（弱标签） |
| `image_names` | `list[str]` | 对应 DICOM 在 ZIP 中的路径 |
| `report_names` | `list[str]` | 对应报告在 ZIP 中的路径 |
| `model` | `str` | 特征提取模型说明 |
| `label_definition` | `str` | 标签定义说明 |
| `source_zip` | `str` | 源 ZIP 路径 |

当前这一版：3380 张图、1879 份报告，阴性 1812 / 阳性 1568。

## 读取与使用

```python
import torch

package = torch.load(
    r"E:\School\# Doctor\BN5212\BN5212\mimic_cxr_vgg16_embeddings.pt",
    map_location="cpu",
)
X = package["embeddings"].numpy()   # (3380, 4096)
y = package["labels"].numpy()       # (3380,)
```

训练一个简单的分类器（按患者划分训练/测试集）：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 患者ID取自路径 dataset/p<患者ID>/...
groups = [name.split("/")[1] for name in package["image_names"]]

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
clf.fit(X[train_idx], y[train_idx])

prob = clf.predict_proba(X[test_idx])[:, 1]
print("Accuracy:", accuracy_score(y[test_idx], prob > 0.5))
print("AUC:", roc_auc_score(y[test_idx], prob))
```

在当前数据上（851 名患者，训练 2726 / 测试 654 张），这段代码的结果大约是 Accuracy 0.60、AUC 0.62，可以作为基线参考。分数不高主要是因为特征来自未微调的 ImageNet 模型，加上标签本身有噪声。

**一定要按患者划分**，不要用普通的 `train_test_split`。同一次检查的多张图（正位、侧位）共用同一个标签，同一个患者还可能有多次检查。随机按图像划分会让几乎相同的样本同时出现在训练集和测试集，结果会虚高。

## 特征是怎么提取的

1. 读取 DICOM 像素；`MONOCHROME1` 的图像先做反色，使骨骼统一为亮色
2. 归一化到 0–255，转成 3 通道 RGB
3. 缩放到 224×224，按 ImageNet 均值/方差标准化
4. 送入 ImageNet 预训练的 VGG16：`features` → `avgpool` → `classifier` 去掉最后一层分类头，取倒数第二个全连接层（fc7，ReLU + Dropout 之后，推理模式下 Dropout 不生效）的 4096 维输出

VGG16 没有在胸片上微调，特征是通用的 ImageNet 特征。

## 标签规则

对每份报告转小写后按顺序匹配：

1. 出现任一阴性短语 → **0**：`no pneumonia`、`without pneumonia`、`negative for pneumonia`、`no focal consolidation`、`no acute cardiopulmonary abnormality`、`no acute cardiopulmonary disease`、`lungs are clear`
2. 否则出现任一阳性短语 → **1**：`pneumonia`、`focal consolidation`、`airspace opacity`、`airspace opacities`、`pulmonary infiltrate`、`infiltrates`
3. 两者都没有 → 丢弃，不进入数据集

该检查目录下的所有 DICOM 都继承这份报告的标签。

## 注意事项

- **标签是弱标签**，不是专家标注。阴性规则优先，所以一份写了 "lungs are clear … possible pneumonia at the left base" 的报告会被标成 0；`pneumonia` 这种宽泛匹配也会把 "history of pneumonia" 之类标成 1。写报告时应把这一点列为局限性。
- 正位（PA/AP）和侧位图混在一起，没有区分。
- 如果修改了标签规则或换了数据，重新运行脚本即可，会覆盖旧的 `.pt` 文件。
