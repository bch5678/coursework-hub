# BN5212：MIMIC-IV + MIMIC-CXR 数据流水线

本项目交付数据侧工作：cohort 构建、住院时间对齐、可配置二分类标签、patient-level split、图像与表格清洗、统一索引、PyTorch Dataset/DataLoader，以及服务器运行和验证脚本。默认任务是**从胸片预测该次住院死亡**。模型训练与最终效果评估由下游成员接入。

已核对上传的 **BN5212 Group Project Instructions (AY2026/27 Semester 1)** 第 1 页：课程使用 MIMIC-IV **v3.1**、MIMIC-CXR **v2.1.0**，提供胸片子集，可用 `subject_id` 关联，住院死亡预测是课程示例任务。课程还要求小组完成训练、评估并在中期展示初步数据分析；本项目自动生成的 cohort 流程与分布表可用于数据部分。

项目包只含代码、配置、文档和人工合成数据生成器，**没有课程数据、报告正文、下载密码或账号凭证**。真实数据路径、课程下载地址与凭证由运行者在本地/服务器提供。

## 1. 最快开始

需要 Python 3.10+。本项目验证环境是 Python 3.12，CPU 即可完成数据处理和接口测试。

```bash
# 在项目目录运行，首次安装
bash setup_env.sh

# 先用纯合成数据完成端到端验证，不需课程数据或账号
.venv/bin/python scripts/make_synthetic_data.py --output demo/png
PYTHON=.venv/bin/python bash run_pipeline.sh --config demo/png/synthetic_config.json

# 测试原始 DICOM 路径
.venv/bin/python scripts/make_synthetic_data.py --output demo/dicom --format dicom
PYTHON=.venv/bin/python bash run_pipeline.sh --config demo/dicom/synthetic_config.json

# 所有接口与边界测试
.venv/bin/python -m pytest
```

如果服务器已有 PyTorch 环境，可直接 `python -m pip install -r requirements.txt`，然后运行 Python 入口。大部分预处理只读表格和单张图像，不把整套图像载入内存。首次安装 PyTorch 的包较大；使用 GPU 的组员可先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装服务器所需版本，再安装其余依赖。本项目无需 torchvision。

Windows 等效命令：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/make_synthetic_data.py --output demo/png
.\.venv\Scripts\python.exe run_pipeline.py --config demo/png/synthetic_config.json --test-loader
```

`run_pipeline.py` 是跨平台的一键入口；`run_pipeline.sh` 和 `run_pipeline.ps1` 是便利封装。运行脚本不自动安装依赖，安装只需执行一次。

## 2. 接入真实课程数据

先在服务器配置 JSON 中修改四个路径。所有相对路径均**相对于该 JSON 文件所在目录**，不是终端当前目录。支持绝对路径和 `${ENV_VAR}`。输出目录必须是尚不存在的新目录。

```json
"paths": {
  "mimic_iv_root": "/srv/course/mimiciv/3.1",
  "image_root": "/srv/course/mimic-cxr/2.1.0",
  "cxr_metadata": null,
  "output_dir": "/srv/derived/bn5212/mortality_v1"
}
```

编辑完整的 `config/default.json`，或复制为 `config/server.local.json` 后编辑。上面只是 `paths` 段。

### A. 课程子集为原始 DICOM

`cxr_metadata` 留空。脚本直接从本地 DICOM 头提取 `StudyDate`、`StudyTime`、`ViewPosition`，从目录名取得 `subject_id`、`study_id`，并校验头中的 PatientID（存在时）。需要类似结构：

```text
/srv/course/mimiciv/3.1/hosp/
  admissions.csv.gz
  patients.csv.gz
/srv/course/mimic-cxr/2.1.0/
  files/pXX/p<SUBJECT_ID>/s<STUDY_ID>/<DICOM_ID>.dcm
```

医院表支持 `.csv` 或 `.csv.gz`，位于 `hosp/` 或 `mimic_iv_root` 直接目录；同名多个候选文件会报错，避免误用不同版本。扫描 DICOM 不依赖全量 CXR 下载，也不需要报告文件。

如果原始子集目录被重命名，使用下面的显式元数据方式指定文件路径；不要从文件顺序猜测患者或研究编号。`cxr-record-list.csv.gz` 仅提供标识符映射，不能单独替代本项目要求的时间与视角元数据。

### B. 子集为 JPG/PNG，或有现成元数据

设置 `paths.cxr_metadata` 为课程提供的对应元数据文件，或官方 JPG 配套元数据路径。必需列：

```text
subject_id,study_id,dicom_id,StudyDate,StudyTime,ViewPosition
```

`image_path` 是可选列，存在时应为相对 `image_root` 的路径。自定义目录或 PNG 子集建议明确提供这一列。没有 `image_path` 时，脚本查找 `files/pXX/p<SUBJECT_ID>/s<STUDY_ID>/<DICOM_ID>.dcm/.jpg/.jpeg/.png`，多种格式同时存在则报错。

官方 [MIMIC-CXR-JPG v2.1.0](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) 页面列出的元数据文件仍名为 **`mimic-cxr-2.0.0-metadata.csv.gz`**。这是实际文件名，不代表将课程版本改为 2.0.0。JPG 是原始 CXR 的独立配套发布，需要自行确保已有相应文件和访问权限。默认 DICOM 路径不要求额外下载 JPG 数据集。

有完整元数据但只有课程图像子集时，流水线只保留**本地存在**的文件，并报告未在子集中的图像数量。缺失文件不自动当作损坏文件，也不强制补齐整个 MIMIC-CXR。

```bash
PYTHON=.venv/bin/python bash run_pipeline.sh --config config/server.local.json
```

## 3. 可选：将下载和预处理合并为一次运行

已下载/解压的数据可以直接运行上面的命令。需要下载时，将 `config/download_manifest.example.json` 复制为私有文件（建议名称 `private_manifest.local.json`），填入授权的**直接 HTTPS 文件地址**、目标位置和可信 SHA-256。

```bash
PYTHON=.venv/bin/python bash run_pipeline.sh \
  --config config/server.local.json \
  --download-manifest config/private_manifest.local.json
```

流程为：下载或恢复 `.part` → SHA-256 验证 → 可选解压 → 数据处理 → 全索引校验 → DataLoader 接口测试。已有正确校验和的文件会复用，已完成且标记匹配的解压目录会复用。改变实验配置时使用新的 `output_dir`。

下载清单可列出整个课程压缩包，或分别列出医院表和所需胸片文件。`destination`、`extract_to` 相对清单中的 `destination_root`；后者相对于清单目录。可选解压支持 ZIP/TAR/TAR.GZ；`max_extract_bytes` 控制解压上限，单位字节。请将配置中的数据根目录指向**解压后的实际层级**。

下载支持 HTTP Basic Auth：运行前设置 `DATA_HTTP_USERNAME`、`DATA_HTTP_PASSWORD`，并将 `DATA_HTTP_AUTH_HOST` 设为唯一允许接收该凭证的主机名（例如 physionet.org）；如果 ZIP 本身加密，可设置 `DATA_ARCHIVE_PASSWORD`。请通过服务器的安全环境配置或交互方式提供，勿写入共享脚本、仓库或命令历史。没有这些变量时按无需 Basic Auth 的文件地址下载。课程网页/SSO 登录页面不等同于直接文件地址：此时先用课程允许的方式下载解压，再运行本地路径模式。脚本拒绝重定向，防止认证信息随跳转发送；使用最终授权文件 URL。

可信 SHA-256 应来自课程提供方、已验证的源文件，或你本人在可信下载后计算的值。项目无法预先填写未拿到的数据的真实哈希，也不会绕过课程/PhysioNet 的访问要求。清单示例中的地址和哈希是占位符。

## 4. 默认研究定义与可调参数

| 项目 | 默认行为 | 配置位置 |
|---|---|---|
| 研究单位 | 每次符合条件的住院一张图像 | `cohort.selection` |
| 人群 | 估计入院年龄 ≥18 岁 | `cohort.min_age` |
| 视角 | AP、PA；同时间默认 AP 优先 | `cohort.views` |
| 住院关联 | 相同 subject_id，且 `admittime <= study_time < dischtime`，唯一匹配 | 固定校验规则 |
| 预测时间 | 所选胸片的 StudyDate + StudyTime | 固定规则 |
| 选片窗口 | 入院后 ≤48 小时 | `alignment.max_hours_after_admission` |
| 结局前截止 | 严格早于 deathtime（存在时），否则早于 dischtime | 固定规则 |
| 额外预留时间 | 0 小时，可设为 ≥6 小时等 | `alignment.minimum_hours_before_end` |
| 标签 | 该次住院的 hospital_expire_flag，1=死亡，0=存活 | `label` |
| 分组 | 同一患者所有记录只属于一个集合 | 固定 `subject_id` |
| 比例 | 70% / 15% / 15%，seed=5212 | `split` |
| 解码验证 | 验证所有候选图像，损坏比例 >5% 时失败 | `cleaning` |
| 图像批次 | 灰度，224×224，float32，固定均值/标准差 | `loader` |

**48 小时、成年人、AP/PA、首张可用图像、70/15/15 都是项目默认设计，不是课程强制要求。** 预测任务是“入院后早期取得这张胸片时，预测当前住院最终是否死亡”，不是“入院时预测”，也不是“48 小时内死亡”。将窗口设为 `null` 会允许整个住院期间的图像，研究解释也随之改变。

`first_per_admission` 选择最早的**本地可用且通过清洗**的图像。另有 `first_per_study`（每研究一张）和 `all_images`。同时间按视角列表、study_id、dicom_id 确定次序；随机 ID 只用于同时间破除平局。选择多张图像时，`sample_weight=1/该次住院入选图像数`，下游可按住院均衡损失，但 DataLoader 不会自动加权损失。

标签扩展示例：在 admissions 输入表中提供 `custom_outcome` 后，将标签段改为：

```json
"label": {
  "kind": "admission_binary",
  "column": "custom_outcome",
  "positive_values": ["yes"],
  "negative_values": ["no"]
}
```

未在映射中的值会排除，缺失标签不会补成 0。本项目提供住院级二分类适配；时变标签、多分类、30 天生存/删失建模应另定义观察窗与结局规则，不能直接换个列名当作已实现。住院时间和死亡时间一致性检查对自定义二分类同样生效。

## 5. 清洗与避免信息泄漏

- 标识符以字符串读入，防止浮点 ID；完全重复的记录去重，同一关键 ID 的冲突记录报错。
- 缺失/非法胸片时间排除；数字化丢失的 StudyTime 前导零会补齐，保留微秒。直接 DICOM 头要求完整 HHMMSS 精度。
- 时间匹配只在同一患者内进行。重复或重叠住院导致多重匹配时，排除该胸片，不用最近入院时间强行选择。
- 严格排除结局后影像。死亡标志、死亡时间冲突的记录排除；死亡标志为 1 但缺失 deathtime 时保留该标签、用出院时间截止，并在 QA 中提醒这一局限。
- 年龄用 `anchor_age + 入院年 - anchor_year` 估计；`anchor_age=91` 是高龄匿名化标记，索引保留 `age_is_topcoded`，不能当作精确年龄。
- 不用患者 `dod` 生成住院死亡标签，不把住院死亡和出院后死亡混合。
- 划分时以患者的“入选住院是否至少一次标签为 1”作为分层依据。两类均至少 3 位患者时分层，每类的各集合至少 1 人；小样本比例可能偏离指定值。条件不满足则使用可复现的非分层患者划分并告警，少于 3 位患者直接失败。
- 分配先按 seed+subject_id 的稳定哈希排序；输入行重排不会改变 split。**改变 cohort、标签、患者集合或比例可能改变分配**，不同模型比较应共同使用已输出的同一份 index/split_assignments，不要分别重新生成。
- 不使用报告正文、出院信息、死亡时间、最终诊断等作为模型特征。索引内结局字段仅供审计，默认 Dataset 只返回图像、标签、权重和标识符。
- 图像使用逐图确定的解码和固定 `mean/std`，无全数据拟合。未来加入实验室指标、标准化或缺失值填补，应仅在 train 患者上拟合，并以胸片预测时点截断所有事件及其可用时间。

MIMIC 各模块对同一患者使用一致的日期偏移，可在患者内对齐；不同患者的匿名化年份不能直接作为真实日历年份做时间拆分。[MIMIC-IV 官方关联说明](https://physionet.org/content/mimiciv/3.1/)

## 6. 运行输出与交接

```text
runs/mortality_v1/
  index.csv                      # 统一 dataset，含所有 split
  index_train.csv / index_val.csv / index_test.csv
  split_assignments.csv          # 一位患者一行，必须冻结复用
  dataset_spec.json              # 机器可读接口、图像根路径、变换参数、index 哈希
  cohort_flow.csv                # 分步纳入/排除数量与统计单位
  qa_report.json                 # 图像/研究/住院/患者数量、分布、告警
  cxr_metadata_snapshot.csv      # 有效且本地存在的影像候选元数据
  input_image_inventory.csv      # 文件相对路径、大小、修改时间
  config.snapshot.json           # 本次原始配置
  run_manifest.json              # 输入哈希、依赖/系统版本、预处理代码哈希
  pipeline.log                   # UTC 时间日志
  SUCCESS.json                   # 完成标志与输出文件哈希
```

统一 dataset 是“**原图文件 + index.csv + dataset_spec.json**”，不复制或嵌入图像。需要一同保留原图目录。换服务器/挂载点时，在新的配置里指向新根目录并生成新 run；不要单独移动 CSV 后期待旧绝对根路径自动变化。

默认 `audit.hash_images=false`：医院表/元数据文件做 SHA-256，图像目录生成文件清单；设为 `true` 可对最终入选原图逐张哈希，额外增加一次磁盘读取。直接扫描 DICOM 时，本次读取到的元数据另行保存，完整原始图像内容指纹仍需开启这一选项。

输出先写到同目录下 `.名称.staging-*`，所有检查完成后才发布最终目录。失败只保留 staging 内的 `FAILED.json` 和日志，不发布 `SUCCESS.json`。已有结果目录拒绝覆盖：保持不同实验可追溯，用新的 `output_dir` 重跑。运行中断时清理仅对应的 staging 目录后再重试，不要删除原始数据。

真实运行产生的索引、split、快照和日志仍属于数据衍生物，应保存在课程允许的位置；不要加入公开项目包。`.gitignore` 只是辅助，使用自定义输出位置时仍需自行管理。

## 7. 给建模同学的接口

详细字段和约定见 [DataLoader specification](docs/DATALOADER_SPEC.md)。最小读取示例：

```python
from src.data.dataset import make_dataloader

loader = make_dataloader("/srv/derived/bn5212/mortality_v1", "train", batch_size=16)
batch = next(iter(loader))
x = batch["image"]       # float32 [B, 1, 224, 224]，默认范围约 [-1, 1]
y = batch["label"]       # int64 [B]，适合 2 类 CrossEntropyLoss
w = batch["sample_weight"]
```

3 通道模型：将配置中的 `channels` 改为 3，并将 `mean/std` 改成各含 3 个值的列表后生成新 run。通道是同一灰度图重复，不是额外影像模态。

服务器交接验证：

```bash
python test_dataloader.py --run-dir /srv/derived/bn5212/mortality_v1
```

该脚本验证所有产物哈希、完整索引的患者/住院/研究隔离与时间关系，再实际读取每个 split 的一个 batch；并非读取全部图像的性能测试。默认预处理已对所有候选图像解码校验。`num_workers>0` 的脚本调用在 Windows 下应放在 `if __name__ == "__main__":` 内。

## 8. 文件导航

| 文件 | 作用 |
|---|---|
| `src/data/tables.py` | 医院表、年龄和基础清洗 |
| `src/data/metadata.py` | CXR 元数据适配、DICOM 头、时间解析 |
| `src/data/cohort.py` | 唯一住院关联、标签、影像选择 |
| `src/data/splits.py` | 患者分组划分与整体校验 |
| `src/data/images.py` | 灰度解码、DICOM 极性/LUT、缩放填充 |
| `src/data/dataset.py` | PyTorch Dataset/DataLoader |
| `src/data/pipeline.py` | 输出、统计、日志和完成标志 |
| `src/data/download.py` | 校验和下载、续传、安全解压 |
| `scripts/make_synthetic_data.py` | 人工测试数据，PNG/DICOM 两种 |
| `tests/` | 边界情况、真实批次、分组、下载/解压测试 |

## 9. 已验证内容与实际数据局限

本项目已在 Windows/Python 3.12 上通过合成数据端到端和多进程 DataLoader 测试，详见 [验证记录](docs/VALIDATION.md)。尚未访问或运行课程受限数据，因此不会提供真实 cohort 人数、死亡率或模型表现；68 个样本等测试数字仅来自人工数据。

当前处理的是能与合法住院区间关联、具有可用胸片的患者子集，不能代表所有住院患者。严格时间和视角条件也会改变入选人群。对于住院外/急诊但尚未正式入院的图像，默认排除；若要包含 ED 图像，应明确新增 ED 关联和标签定义。未使用官方 JPG benchmark split，本项目独立生成面向课程死亡任务的患者划分；不能把本项目结果称作官方 benchmark 结果。

压缩 DICOM 如果系统缺少对应解码器，预处理会报错或触发损坏比例阈值；按实际 Transfer Syntax 安装所需 pydicom 像素解码依赖。测试覆盖的是未压缩单帧灰度 DICOM。图像转换采用 modality/VOI LUT、MONOCHROME1 反相、逐图缩放到 8 bit，并保持长宽比补边到固定尺寸；不保证与官方 JPG 转换逐像素相同。数据集版本由配置声明，不能仅凭 CSV 的内容自动证明版本，应核对课程来源。

## 10. 参考资料

- 上传的 `BN5212_Group_Project_Instructions_AY2026-27.pdf`，第 1 页数据要求与任务示例，第 2–3 页项目交付要求。
- [MIMIC-IV v3.1：字段含义、年龄与跨模块时间关联](https://physionet.org/content/mimiciv/3.1/)
- [MIMIC-CXR v2.1.0：DICOM 目录和标识符](https://physionet.org/content/mimic-cxr/2.1.0/)
- [MIMIC-CXR-JPG v2.1.0：StudyDate、StudyTime、元数据实际文件名](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)
- [PyTorch Dataset/DataLoader](https://docs.pytorch.org/docs/stable/data)
- [PyTorch 可复现性与 worker 随机种子](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [pydicom 像素处理](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_voi_lut.html)

课程参考论文仅作为任务动机，本项目没有声称复现其 cohort 或预处理：Khader et al. (2023), [Medical transformer for multimodal survival prediction in intensive care](https://doi.org/10.1038/s41598-023-37835-1)。
