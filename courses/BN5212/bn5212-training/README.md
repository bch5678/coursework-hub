# BN5212：通用训练框架与单模态基线

## 项目目标

本目录负责 BN5212 项目的**模型训练侧**：一套所有实验共用的训练框架，以及 clinical-only 和
CXR-only 两组单模态基线。目标是让项目计划里的四个实验只差一个 fusion 名字，从而把
"MeTra 的 joint self-attention 和显式 cross-attention 哪个更好" 变成一个公平的对照实验，
而不是两套各自调过参的脚本之间的比较。

本目录**不负责** cohort 构建（`bn5212-data-pipeline`）和最终测试指标
（`benchmark-evaluation`）。训练只输出 checkpoint 和预测概率，报告里的数字统一由评测项目产生。

### 四个实验 = 四个 config

| 实验 | 配置 | modalities | fusion |
|---|---|---|---|
| 1. Clinical-only | `configs/clinical_only.json` | clinical | `clinical_only` |
| 2. CXR-only | `configs/cxr_only.json` | cxr | `image_only` |
| 简单融合参照 | `configs/concat_fusion.json` | clinical + cxr | `concat_mlp` |
| 3. MeTra baseline | `configs/metra_joint.json` | clinical + cxr | `joint_self_attention` |
| 4. 提出的方法 | `configs/cross_attention.json` | clinical + cxr | `cross_attention` |

Encoder、prediction head、optimizer、schedule、loss、seed、checkpoint 选择规则、预测导出
在五个实验之间完全一致。

## 成员与接口

### 上游：`bn5212-data-pipeline`（数据负责人）

框架通过 `MimicCXRDataset` 读取已冻结的运行目录，因此每次训练都会重新校验
`SUCCESS.json`、`dataset_spec.json` 和 `index.csv` 的哈希。不需要 `pip install` 上游项目：
`upstream.py` 会把它作为私有包 `bn5212_pipeline` 载入，避免和本项目的 `src/` 目录撞名。
默认在同级目录 `../bn5212-data-pipeline` 查找，也可用环境变量
`BN5212_DATA_PIPELINE` 或 `--data-pipeline-path` 指定。

### 下游：`benchmark-evaluation`（评测负责人）

每次训练输出 `predictions_val.csv` 和 `predictions_test.csv`，只含 `sample_id,y_score`
两列，`y_score` 是院内死亡正类概率。写文件前会校验：无重复 ID、无 NaN/Inf、值域在
`[0,1]`、ID 与对应 split 完全一致。`run_manifest.json` 记录评测项目要求的
git commit、dataset index hash、checkpoint SHA-256、seed 和软件环境。

模型本身还实现了评测项目的 `ModelAdapter` 协议（`eval()` 和 `predict_logits(batch)`），
所以评测方也可以直接对 checkpoint 跑 inference，不需要额外适配层。

### 多模态负责人

`cross_attention` 归多模态负责人所有。框架提供了一份可运行的参考实现，使四个实验今天就能
端到端跑通；替换它不需要改动框架其他任何文件。接口见 [docs/FUSION_API.md](docs/FUSION_API.md)。

## 环境准备

需要 Python 3.10+（本地验证环境为 Python 3.13，CPU）。本项目维护独立虚拟环境。

Windows PowerShell：

```powershell
cd courses\BN5212\bn5212-training
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Linux / macOS：

```bash
cd courses/BN5212/bn5212-training
python -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
```

MeTra 使用 ImageNet 预训练 ViT-B/16，需要额外安装 timm：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[pretrained]"
```

GPU 环境请先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本，
再安装其余依赖。

## 在本机跑训练（不需要课程数据）

真实 MIMIC 数据只能在获授权的位置读取，但整条训练链路可以完全用合成数据在本机验证。

### 第 1 步：生成合成运行目录

这一步用数据流水线自己的合成生成器，产出一个结构与真实运行完全一致的 `run_dir`。
不需要课程账号、下载密码或任何真实数据。

```powershell
cd ..\bn5212-data-pipeline
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\make_synthetic_data.py --output demo\png
.\.venv\Scripts\python.exe run_pipeline.py --config demo\png\synthetic_config.json --test-loader
cd ..\bn5212-training
```

产出目录为 `..\bn5212-data-pipeline\demo\png\processed`，包含 46 / 12 / 10 条
train / val / test 样本，图像为 32×32 单通道。

### 第 2 步：跑通整条链路

```powershell
.\.venv\Scripts\bn5212-train.exe --config configs\smoke.json --run-dir ..\bn5212-data-pipeline\demo\png\processed
```

约几十秒完成，输出到 `outputs\smoke\<时间戳>\`。

> `configs/smoke.json` 故意给合成临床数据注入了与标签相关的信号（`clinical_signal: 1.5`），
> 用来证明训练循环确实能学到东西，因此它的 AUROC 会接近 1.0。
> **这是自检，不是任何实验结果。** 真实结论只能来自真实数据。

### 第 3 步：在合成数据上跑各个实验

合成图像只有 32×32，用不了 ViT-B/16，所以 `configs/synthetic/` 下另备了五份小模型配置。
一条命令跑完全部五个实验并输出对比表：

```powershell
.\scripts\run_all_experiments.ps1
```

Linux / macOS 用 `./scripts/run_all_experiments.sh`。单独跑某一个：

```powershell
.\.venv\Scripts\bn5212-train.exe --config configs\synthetic\cxr_only.json
.\.venv\Scripts\bn5212-train.exe --config configs\synthetic\metra_joint.json
.\.venv\Scripts\bn5212-train.exe --config configs\synthetic\cross_attention.json
```

> 合成图像是与标签无关的渐变图案，合成临床值是噪声，
> **所以 AUROC 应该在 0.5 附近——那才是正确结果**，不是模型没训好。
> 这一步验证的是代码路径，不是模型效果。

如果要直接用 `configs/` 下的真实配置在合成数据上试，需要手动覆盖模型尺寸：

```powershell
# CXR-only
.\.venv\Scripts\bn5212-train.exe --config configs\cxr_only.json `
  --run-dir ..\bn5212-data-pipeline\demo\png\processed `
  --set image_encoder.name=vit --set image_encoder.embed_dim=96 `
  --set image_encoder.patch_size=8 --set image_encoder.depth=2 `
  --set image_encoder.num_heads=3 --set fusion.embed_dim=96 `
  --set optim.epochs=5 --set optim.lr=0.001 --set optim.amp=false

# Clinical-only（合成 provider，无需真实临床表）
.\.venv\Scripts\bn5212-train.exe --config configs\clinical_only.json `
  --run-dir ..\bn5212-data-pipeline\demo\png\processed `
  --set data.clinical_provider=synthetic --set data.clinical_timesteps=12 `
  --set optim.epochs=5

# MeTra joint self-attention
.\.venv\Scripts\bn5212-train.exe --config configs\metra_joint.json `
  --run-dir ..\bn5212-data-pipeline\demo\png\processed `
  --set data.clinical_provider=synthetic --set data.clinical_timesteps=12 `
  --set image_encoder.name=vit --set image_encoder.embed_dim=96 `
  --set image_encoder.patch_size=8 --set image_encoder.depth=2 `
  --set image_encoder.num_heads=3 --set clinical_encoder.embed_dim=96 `
  --set fusion.embed_dim=96 --set fusion.num_heads=3 `
  --set optim.epochs=5 --set optim.lr=0.001 --set optim.amp=false

# 提出的 cross-attention（只换 fusion）
.\.venv\Scripts\bn5212-train.exe --config configs\cross_attention.json `
  --run-dir ..\bn5212-data-pipeline\demo\png\processed `
  --set data.clinical_provider=synthetic --set data.clinical_timesteps=12 `
  --set image_encoder.name=vit --set image_encoder.embed_dim=96 `
  --set image_encoder.patch_size=8 --set image_encoder.depth=2 `
  --set image_encoder.num_heads=3 --set clinical_encoder.embed_dim=96 `
  --set fusion.embed_dim=96 --set fusion.num_heads=3 `
  --set optim.epochs=5 --set optim.lr=0.001 --set optim.amp=false
```

`--set` 可以覆盖任意配置字段，格式为 `section.key=value`，值按 JSON 解析。

### 第 4 步：交给评测项目

```powershell
cd ..\benchmark-evaluation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\bn5212-evaluate.exe `
  --run-dir ..\bn5212-data-pipeline\demo\png\processed `
  --val-predictions ..\bn5212-training\outputs\cxr_only\<run-id>\predictions_val.csv `
  --test-predictions ..\bn5212-training\outputs\cxr_only\<run-id>\predictions_test.csv `
  --model-name cxr-only --model-version v1 `
  --output-dir outputs\cxr-only-v1
```

### 第 5 步：比较多次训练（验证集，仅供开发时参考）

```powershell
cd ..\bn5212-training
.\.venv\Scripts\bn5212-summarize.exe outputs\cxr_only\<run-id> outputs\clinical_only\<run-id> `
  --output outputs\training_comparison
```

最终对外的结果表由 `bn5212-compare` 生成，不用这张表。

## 在本机 GPU 上跑真实数据

### 1. 安装能在你的显卡上运行的 PyTorch

**RTX 50 系列（Blackwell，compute capability 12.0 / sm_120）必须用 CUDA 12.8 以上的
PyTorch 构建。** 默认的 `pip install torch` 装的是 CPU 版；cu121 / cu124 的轮子虽然能
`import`，`torch.cuda.is_available()` 也返回 True，但一跑 kernel 就报
`no kernel image is available for execution on the device`。

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch `
  --index-url https://download.pytorch.org/whl/cu128
```

装完务必**真的跑一次 kernel** 验证，光看 `is_available()` 不够：

```powershell
.\.venv\Scripts\python.exe -c @'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())
a = torch.randn(2048, 2048, device="cuda")
print("matmul OK", float((a @ a).sum()))
'@
```

`get_arch_list()` 里必须出现 `sm_120`。

### 2. 需要下载什么，占多少空间

完整 MIMIC-CXR-JPG 约 **570 GB**，一般装不下，也没必要。课程提供的是胸片**子集**，
按课程渠道拿那一份。各部分大致体量：

| 数据 | 用途 | 量级 |
|---|---|---|
| `admissions.csv.gz` + `patients.csv.gz` | cohort、标签、年龄性别 | 约 10 MB |
| 课程胸片子集 | 图像分支 | 取决于子集大小 |
| `chartevents.csv.gz` | 生命体征（临床分支必需） | 约 30 GB 压缩 |
| `labevents.csv.gz` | 实验室指标 | 约 10 GB 压缩 |

访问 MIMIC 需要 PhysioNet credentialed access（CITI 培训 + 签署 DUA），审批要时间；
如果课程已经提供数据，走课程渠道即可。**不要把任何真实数据放进这个 Git 仓库。**

### 3. 分两步走

**第一步只做 CXR-only**，因为它只需要 10 MB 的两张表加图像子集，不需要 30 GB 的
`chartevents`，可以最快把真实数据链路跑通。

复制 `config/default.json` 为 `config/server.local.json`（`.gitignore` 已排除 `*.local.json`），
改四个路径后跑流水线：

```powershell
cd ..\bn5212-data-pipeline
.\.venv\Scripts\python.exe run_pipeline.py --config config\server.local.json --test-loader
```

然后训练：

```powershell
cd ..\bn5212-training
.\.venv\Scripts\bn5212-train.exe --config configs\cxr_only.json `
  --run-dir <流水线输出目录> --device cuda
```

**第二步**再下载 `chartevents.csv.gz`（约 30 GB）抽取临床特征。
抽取前**务必先核对 itemid**，不同 MIMIC 版本的 itemid 会变：

```powershell
.\.venv\Scripts\bn5212-extract-clinical.exe --inspect-items `
  --run-dir <流水线输出目录> --mimic-root <MIMIC-IV 根目录>
```

逐行确认每个 itemid 在 `d_items` 里的真实 label；写错会显示成不相干的 label 或
`<NOT FOUND>`，需要调整就改 `configs/clinical_items.json`。确认后抽取：

```powershell
.\.venv\Scripts\bn5212-extract-clinical.exe `
  --run-dir <流水线输出目录> `
  --mimic-root <MIMIC-IV 根目录> `
  --output <输出路径>\clinical_features.csv --max-hours 48
```

30 GB 是流式扫描的，内存占用平稳；因为一进来就按 cohort 过滤，输出通常只有几十 MB。
同时生成 `clinical_features_qa.json`（每个变量的覆盖率和分类丢弃计数），
可以直接写进报告的数据章节。细节见
[docs/CLINICAL_FEATURE_SPEC.md](docs/CLINICAL_FEATURE_SPEC.md)。

然后跑全部五个实验：

```powershell
.\scripts\run_all_experiments.ps1 -Preset real `
  -RunDir <流水线输出目录> `
  -ClinicalSource <输出路径>\clinical_features.csv `
  -Device cuda
```

### 4. 显存调参

配置已按 **8 GB 显存**设定：ViT-B/16 在 224×224 下，微批 8 × 梯度累积 2 = 有效批 16，
并开启 AMP。ViT-B/16 光是参数、梯度和 AdamW 的两个状态就约 1.4 GB，剩下才是激活值。

显存不够时按这个顺序调：

```powershell
--set data.batch_size=4 --set optim.grad_accumulation_steps=4   # 有效批不变
--set image_encoder.freeze=true                                  # 冻结 backbone，显存大降
```

显卡更大时反向调：提高 `data.batch_size`，把 `optim.grad_accumulation_steps` 降回 1。

### 5. MeTra 的 384×384 设置

MeTra 用 384×384、3 通道、ImageNet 归一化。图像尺寸和归一化参数由**流水线**决定，
不由本项目决定，所以需要用这样的 `loader` 配置另做一次流水线运行：

```json
"loader": {
  "image_size": 384, "channels": 3, "batch_size": 8, "num_workers": 4,
  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]
}
```

然后把 `image_encoder.timm_model` 改为 `vit_base_patch16_384`。
384 的 token 数是 224 的约 3 倍，8 GB 显存下需要把微批降到 2–4 并相应提高梯度累积。
时间紧张时先用 224 出完整结果，384 作为对照补充。

## 输入数据

- 图像、标签、patient split：来自数据流水线的冻结运行目录，只读。
- 临床时序：由数据负责人提供的长格式表；当前流水线尚未实现这部分抽取，
  因此本项目内置 `synthetic` provider 供离线开发，见下方"后续工作"。
- 本目录不含任何真实数据、凭证或下载地址。`outputs/`、`.venv/`、`*.pt` 已在 `.gitignore` 中排除。

## 运行方式（真实数据，服务器）

```bash
.venv/bin/bn5212-train \
  --config configs/metra_joint.json \
  --run-dir /srv/derived/bn5212/mortality_v1 \
  --set data.clinical_source=/srv/derived/bn5212/clinical_features.csv.gz \
  --device cuda
```

每次运行产出一个不可覆盖的目录：

```text
outputs/<experiment>/<run-id>/
  config.json                      完整配置快照
  run_manifest.json                git commit、dataset hash、checkpoint hash、seed、环境
  checkpoint_best.pt               按验证集选出的 checkpoint
  checkpoint_last.pt
  clinical_normalizer.json         仅在 train 上拟合的标准化统计量
  metrics_val.json                 验证集指标（仅用于选模）
  predictions_val.csv              sample_id,y_score —— 交给评测项目
  predictions_test.csv             sample_id,y_score —— 交给评测项目
  predictions_*_detailed.csv       含 label / subject_id，供自查
  summary.csv / summary.md
  figures/
    training_curves.png  (+ .csv)
    roc_curve_val.png    (+ .csv)
    pr_curve_val.png     (+ .csv)
    score_distribution_val.png (+ .csv)
    attention_matrix.png / attention_patch_maps.png   仅 cross_attention
```

**不生成 HTML 报告**：结果展示统一用图片、表格和数据文件，最终报告由评测项目负责。
每张图旁边都有同名 `.csv`，方便直接取数或重画。

## 设计要点

### fusion 是唯一的变量

所有 fusion 模块实现同一个签名：

```python
forward(image_tokens:    [B, N, D] | None,
        clinical_tokens: [B, M, D] | None,
        clinical_mask:   [B, M]    | None) -> [B, output_dim]
```

单模态实验对不用的模态传 `None`。这样"换 fusion"就是配置里改一个名字，
不会顺手带进 encoder 或训练策略的差异。

### 临床 token 的含义决定 RQ3 能不能回答

`tokenization: per_variable` 让一个 token 对应一个临床变量（M = K）。
只有这样，cross-attention 权重才是"某个临床变量关注了哪些影像 patch"，RQ3 才有意义。
`per_timestep` 可用于消融，但那时权重变成"某个小时关注了哪里"。

### 两条防泄漏规则由代码强制，而不是靠约定

1. 临床观测必须严格早于预测时点 `study_time`。即使输入文件包含之后的数据，
   provider 也会丢弃（`tests/test_clinical.py` 有对应测试）。
2. 标准化统计量只在 train split 上拟合，传 `split="val"` 会直接报错。

### 验证集与测试集严格分离

框架只在验证集上算指标、选 checkpoint、画图。测试集只导出预测概率，
本项目任何代码都不读取测试集标签算分数。

### MeTra 的复现方式

`joint_self_attention` 是按论文（Nature Sci Rep `s41598-023-37835-1`）描述**独立实现**的，
不是照搬作者代码——[官方仓库](https://github.com/FirasGit/MeTra) 未标注 license，
不适合并入共享仓库。已对齐的部分：

| 项目 | MeTra 原文 | 本框架 |
|---|---|---|
| 图像 backbone | ImageNet 预训练 ViT，patch 16 | `timm_vit`，`vit_base_patch16_224/384` |
| 输入分辨率 | 384×384 | 由流水线 `loader.image_size` 决定 |
| 临床输入 | K×48（15 个参数，ICU 前 48 小时） | `[K, T]`，K/T 由 provider 决定 |
| 临床投影 | 单个 linear layer | `linear_projection` |
| 融合 | CLS + 可学习位置嵌入 + joint MHSA | `joint_self_attention` |
| optimizer | AdamW | AdamW |
| 学习率 | 5e-6，cosine 退火到 1e-7 | `lr: 5e-6`, `min_lr: 1e-7` |
| epochs | 200 | 200 |
| loss | binary cross-entropy | `num_outputs: 1` + BCEWithLogits |
| 类别不平衡 | 指出但未处理 | `positive_class_weight: null`（默认不处理） |
| modality dropout | vision dropout 30% | `optim.image_dropout_prob: 0.3` |

已知差异（需要在报告中说明）：MeTra 用 MIMIC-IV v1.0 且以 **ICU 入住**后 48 小时为窗口；
本项目的流水线基于 v3.1 且以**住院**入院后 48 小时内的首张胸片为预测时点，
也没有 ICU 层级（无 `stay_id`）。这一差异见下方"后续工作"第 2 条。

## 验证与结果

在本机（Windows，Python 3.13，CPU，torch 2.14.0）执行：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

**66 项测试全部通过**，全部基于合成数据，不读取任何真实或受限数据。覆盖：

- 五个 fusion 模块的形状、掩码语义、缺失模态的拒绝行为、cross-attention 三个方向；
- 临床 provider 的时间截断、缺失掩码不变量、标准化只在 train 拟合、长格式表的错误处理；
- 配置校验：未知字段、embedding 宽度不一致、shipped config 全部可加载；
- 选模指标与手算值对照，单一标签 split 返回 `None` 而非伪造数值；
- 五个实验端到端运行并产出全部交付物；
- 预测文件与冻结 split 完全对齐、checkpoint 往返一致、同 seed 可复现；
- **预测文件能通过 `benchmark-evaluation` 的 `load_predictions` 校验**（需装上评测项目，
  未装时该测试自动跳过）。

已完成的端到端联调：合成 `run_dir` → 本框架训练 → `bn5212-evaluate`，
评测项目正常产出指标与图表。框架算出的验证集 Brier score 与评测项目的结果
逐位一致，说明两侧的概率定义没有分歧。

**目前没有任何真实数据结果。** 以上全部为合成数据自检，不能作为 benchmark 表现。

## 后续工作

按优先级，前两项需要组内确认：

1. **临床特征抽取已实现，但尚未跑过真实数据。** 数据流水线本身只读 `admissions` 和
   `patients`，所以临床时序由本项目的 `bn5212-extract-clinical` 补上
   （规格与用法：[docs/CLINICAL_FEATURE_SPEC.md](docs/CLINICAL_FEATURE_SPEC.md)）。
   它只产出 side-car 文件，不修改流水线代码。
   **抽取前必须用 `--inspect-items` 核对 itemid**：配置里的 itemid 来自公开文档，
   必须对着实际数据的 `d_items` 确认一遍。目前全部验证都基于合成的 MIMIC 形状表格，
   没有跑过真实 `chartevents`。
   如果数据负责人接手，把 `extract.py` 和 `configs/clinical_items.json` 搬进流水线即可。

2. **临床时间窗口与 MeTra 对不上，需要全组拍板。**
   MeTra 用 ICU 入住后 48 小时；现在的 cohort 以住院入院后 48 小时内的首张 AP/PA 胸片
   作为唯一预测时点，且评测协议要求所有临床特征以 `study_time` 为截止点。
   若胸片在入院第 6 小时，可用临床历史只有 0–6 小时，拿不到完整 48 格。
   - **A 案（建议）**：保留 `study_time`，窗口 = `[admittime, study_time]`，
     变长后用 mask 补齐。无泄漏，现有 split 不用重新冻结，报告中说明偏离 MeTra。
     框架已按此实现（mask 贯穿 encoder 和所有 fusion）。
   - **B 案**：改为 ICU 级，预测时点 = ICU 入住 + 48 小时。更贴近 MeTra，
     但要数据负责人加入 `icustays` 重建 cohort，并处理 48 小时内死亡/出院的
     immortal time bias。

3. **MeTra 的 384×384 / 3 通道 / ImageNet 归一化设置**需要数据负责人另做一次运行，
   见上方"接入真实数据"。

4. `cross_attention` 的参考实现交由多模态负责人接手并扩展
   （Optional 1 的方向对比、Optional 2 的双向融合已留好 `fusion.direction` 开关）。

5. 真实数据到位后：跑通四个实验、多 seed 重复、把 checkpoint 和预测交给评测项目汇总。
