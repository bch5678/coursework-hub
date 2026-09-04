# Unified Dataset & DataLoader Specification v1.0

## 研究对象与预测时点

默认一行对应某位患者某次住院的首张合格早期 AP/PA 胸片，预测该次住院最终是否死亡。`study_time` 是唯一预测时点。默认窗口为入院后 48 小时内；患者可能有多次入选住院。更改 sampling 规则后，一次住院可以有多行，但所有行仍属于同一个患者 split。

## 输入表契约

| 来源 | 必需列 |
|---|---|
| admissions.csv[.gz] | subject_id, hadm_id, admittime, dischtime, deathtime, hospital_expire_flag；自定义标签时额外需要 label.column |
| patients.csv[.gz] | subject_id, anchor_age, anchor_year, gender |
| CXR 元数据 CSV[.gz] | subject_id, study_id, dicom_id, StudyDate, StudyTime, ViewPosition；可选 image_path |
| DICOM 自动扫描 | 目录 p<subject_id>/s<study_id>/<dicom_id>.dcm；头字段 StudyDate, StudyTime, ViewPosition |

医院时间格式为 `YYYY-MM-DD HH:MM:SS`，使用数据提供的匿名化本地时间，不转换 UTC。StudyDate 为 YYYYMMDD，StudyTime 为 HHMMSS[.ffffff]。CSV 将时间数字化丢失前导零时补成 6 位；因此自制 CSV 不要用仅小时或小时分钟的缩略形式。DICOM 头时间不足秒精度时直接排除。日志时间单独采用 UTC。

## index.csv 列定义（顺序固定）

| 列 | 类型 / 约束 | 用途 |
|---|---|---|
| sample_id | string，唯一，cxr_ + dicom_id | 样本追踪 |
| subject_id | string，正整数文字 | 患者分组，禁止作为模型特征 |
| hadm_id | string，正整数文字 | 唯一匹配住院 |
| study_id | string，正整数文字 | 影像研究 |
| dicom_id | string，唯一 | 原始图像标识 |
| image_path | string，根目录内相对 POSIX 路径 | 定位原图，禁止路径穿越 |
| study_time | timestamp，无时区 | 预测时点，保留微秒 |
| admittime | timestamp，无时区 | 入院时间，仅追踪 |
| dischtime | timestamp，无时区 | 出院结局时间，仅审计，不能作输入 |
| deathtime | nullable timestamp | 已知住院死亡时间，仅审计，不能作输入 |
| hours_since_admission | float ≥0 | study_time - admittime，小时 |
| view | string | AP/PA 或配置允许值 |
| gender | string | 来自 patients，未作为默认模型输入 |
| age_at_admission | float | anchor_age + 入院年 - anchor_year，近似值 |
| age_is_topcoded | boolean | anchor_age 是否为高龄匿名化值 91 |
| label | int 0/1，无缺失 | 当前配置二分类标签 |
| label_name | string | in_hospital_mortality 或自定义列名 |
| sample_weight | float >0 | 1 / 本次住院入选图像数 |
| split | train / val / test | 患者级集合 |

关键不变量：同一 subject_id、hadm_id、study_id、dicom_id、image_path 均不能横跨多个 split；一图一行；同次住院标签一致；`admittime <= study_time < min(已知 deathtime, dischtime)`。死于院内但缺失精确死亡时间时，只能检查早于出院，QA 会说明。

`split_assignments.csv` 包含 subject_id、patient_stratum、split。patient_stratum 是该患者入选记录的最大标签，仅用于分层，不是另一个预测目标。

## PyTorch 接口

```python
from src.data.dataset import MimicCXRDataset, make_dataloader

dataset = MimicCXRDataset(run_dir, split="train", transform=None)
sample = dataset[0]
loader = make_dataloader(
    run_dir, "train", batch_size=16, num_workers=0,
    shuffle=None, seed=None, transform=None,
)
```

run_dir 指向已完成运行目录。Dataset 要求 `SUCCESS.json` 存在并核对 index.csv 与 dataset_spec.json 中的哈希；完整产物哈希由 `test_dataloader.py` 校验。

| 键 | 单样本 | 默认 collate 后 |
|---|---|---|
| image | torch.float32 [C,H,W] | torch.float32 [B,C,H,W] |
| label | torch.int64 标量 | torch.int64 [B] |
| sample_weight | torch.float32 标量 | torch.float32 [B] |
| sample_id, subject_id, hadm_id, study_id, dicom_id | string | list[str]，各长 B |

不返回报告文本、出院状态、死亡时间或临床结局字段作为特征。`image_root` 在 dataset_spec.json 中为本次运行的绝对路径，image_path 为相对路径，原图必须存在。

## 变换约定

1. JPG/PNG 读取后转单通道 8 bit 灰度；只接受非恒定、最小边至少 2 像素的图像。
2. DICOM 只支持单帧 MONOCHROME1/2。解码、modality LUT/rescale、VOI LUT/windowing，排除 padding 对强度范围的影响；MONOCHROME1 反相。每张图按有效像素范围映射至 8 bit。这不等同于官方 JPG 转换的逐像素复现。
3. 保持比例缩放并在黑色背景上居中补边，默认 H=W=224。得到 [0,1] 浮点值。
4. 按固定配置做 `(x-mean)/std`。默认 C=1、mean=[0.5]、std=[0.5]，范围 [-1,1]。C=3 时重复灰度通道，mean/std 长度必须为 3。
5. 用户 transform 接收**已归一化 tensor**，应返回相同形状/类型的 tensor。可用于训练增强；验证和测试通常不传入随机增强。本项目默认没有随机图像增强。

没有跨患者统计量拟合。未来增加从数据估计的强度/临床标准化参数时，必须仅拟合 train 并冻结到 val/test。

## 随机性、采样和模型侧说明

- 默认仅 train 打乱，val/test 顺序固定；drop_last=False，不做重复采样，不自动处理类别不平衡。
- shuffle seed 默认取本次 split seed；DataLoader 配置 generator 和 worker_init_fn 同步 Python/NumPy worker 种子。相同环境、相同调用顺序可重复；不保证跨 PyTorch 版本的完全一致。
- num_workers 默认 0；多进程时启用 persistent_workers。Windows 调用入口需要 main guard，transform 也必须可序列化。
- 二分类两输出 logits `[B,2]` 可直接用 CrossEntropyLoss(logits, label)。单输出 `[B]` 可用 BCEWithLogitsLoss(logits, label.float())；注意形状对齐。
- 多图像采样可计算每图损失后乘 sample_weight。评估是否按图、住院或患者聚合必须事先确定；不要将同次住院多张图片当作相互独立患者。
- Dataset 未加入临床时序特征。若小组增加 labevents/chartevents，应新增清洗与训练集拟合步骤，至少限制事件时间和实际可用/存储时间不晚于 study_time。

## 失败行为与交接验收

未知配置字段、冲突 ID、空 cohort、少于三位患者、图片错误比例超阈值、产物哈希变更会报错；非法或不可关联的普通记录按规则排除并计数。DataLoader 不吞掉读图错误，也不静默跳过样本或返回空 tensor。

交接时同时提供：已完成运行目录、对应只读原图挂载、运行配置/版本和 `test_dataloader.py` 通过记录。必须共用同一份患者划分。真实数据衍生文件仍按课程数据访问规则保管，项目压缩包只包含代码与人工测试生成器。
