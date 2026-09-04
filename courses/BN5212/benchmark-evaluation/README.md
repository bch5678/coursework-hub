# BN5212：Benchmark Evaluation

## 工作概述

本目录负责 BN5212 项目的 benchmark evaluation。核心任务不是设计新的训练框架或多模态模型，而是建立统一、公平、可复现的评测流程，在同一份冻结数据上测试若干 baseline，并持续测试团队每一版训练模型的效果。

这项工作最终需要回答：

1. 各单模态本身能达到什么效果。
2. 简单多模态融合能否超过单模态。
3. 团队设计的多模态 module 能否超过简单融合 baseline。
4. 不同模型版本之间的提升是否稳定、可信且可复现。

## 当前实现

第一版通用框架已经提供：

- 冻结 dataset/index hash 和 patient split 校验。
- 统一的 `sample_id,y_score` 预测接口。
- validation threshold selection 和独立 test evaluation。
- 默认住院级加权聚合，以及显式 sample-level 选项。
- AUROC、AUPRC、分类指标、Brier score 和患者级 bootstrap 95% CI。
- ROC、PR、calibration 和 confusion matrix 图。
- 单次模型的自包含离线 `report.html`。
- 多模型 `leaderboard.csv` 和 `leaderboard.html`。
- train-prevalence baseline 和 PyTorch model adapter 接口。
- 数据/预测边界情况和端到端自动测试。

## 环境准备

需要 Python 3.10+。本项目维护独立环境，不修改数据流水线的依赖。

```bash
cd courses/BN5212/benchmark-evaluation
python -m venv .venv
.venv/bin/python -m pip install -e ".[test]"
.venv/bin/python -m pytest
```

Windows 激活环境后使用 `.venv\\Scripts\\python.exe` 和 `.venv\\Scripts\\bn5212-evaluate.exe` 等对应入口。

## 最小运行方式

先用 train set prevalence 作为所有样本的预测概率，验证真实数据目录到 HTML 报告的完整链路：

```bash
.venv/bin/bn5212-prevalence-baseline \
  --run-dir /srv/derived/bn5212/mortality_v1 \
  --output-dir outputs/prevalence-v1
```

测试任意模型时，模型分别对 validation 和 test 输出仅包含以下两列的 CSV：

```csv
sample_id,y_score
cxr_example_1,0.137
cxr_example_2,0.824
```

`y_score` 必须是院内死亡正类的概率，范围为 `[0,1]`。评分器从冻结 index 读取真实标签，并要求预测 ID 与对应 split 完全一致。

```bash
.venv/bin/bn5212-evaluate \
  --run-dir /srv/derived/bn5212/mortality_v1 \
  --val-predictions /srv/predictions/model-v1-val.csv \
  --test-predictions /srv/predictions/model-v1-test.csv \
  --model-name proposed-multimodal \
  --model-version v1 \
  --checkpoint /srv/checkpoints/model-v1.pt \
  --output-dir outputs/proposed-multimodal-v1
```

默认按 `hadm_id` 评测。一次住院有多张图像时，以 `sample_weight` 对预测概率加权平均；当前默认 cohort 一次住院只有一张图，因此聚合不会改变结果。只有在评测协议明确要求图像级结果时才使用 `--evaluation-unit sample`。

模型比较命令：

```bash
.venv/bin/bn5212-compare \
  outputs/prevalence-v1 outputs/proposed-multimodal-v1 \
  --output-csv outputs/leaderboard.csv \
  --output-html outputs/leaderboard.html
```

`report.html` 和 `leaderboard.html` 可直接双击或拖入浏览器打开。单模型报告将所有图像以内嵌 base64 保存，因此断网时仍可完整查看；报告内的 CSV/JSON 下载链接使用同目录相对路径。

## 职责范围

### 1. 固定评测协议

- 所有模型共同使用数据流水线生成的同一份 `index.csv`、`split_assignments.csv` 和 train/val/test 划分。
- validation set 用于选择 checkpoint、超参数和分类阈值；test set 只用于最终评测，不能根据 test 结果继续调参。
- 事先确定评测单位。当前默认配置每次入选住院只包含一张胸片，样本级结果基本等同于住院级结果；如果以后一次住院包含多张图像，需要先按 `hadm_id` 聚合或使用 `sample_weight`，不能把这些图像当成相互独立的患者。
- 按 `subject_id` 保持患者级隔离。评测代码不得重新生成 split，也不得改变 test cohort。
- 固定模型输入预处理、缺失模态处理、随机种子和概率输出定义。
- 将本项目结果描述为课程项目内部 benchmark，不称为官方 MIMIC benchmark。

### 2. 搭建通用 evaluation 框架

评测框架应同时支持单模态模型和多模态模型，并提供统一的模型适配接口。目标是一条命令加载数据、模型配置和 checkpoint，完成 inference、指标计算和结果保存。

框架至少应实现：

- 校验数据运行目录的 `SUCCESS.json`、dataset schema 和 index hash。
- 加载模型 checkpoint，并把不同模型的输出统一转换为正类概率。
- 使用 `model.eval()` 和无梯度模式执行确定性 validation/test inference。
- 检查样本是否遗漏、重复，输出是否包含 NaN/Inf，预测 ID 是否与冻结 index 对齐。
- 保存逐样本预测、汇总指标、图表和完整运行配置。
- 记录 dataset hash、checkpoint hash、Git commit、模型版本、seed 和软件环境。
- 对空 split、单一标签 split、checkpoint 不兼容等情况给出明确错误或警告。单一标签 split 的 AUROC 应标记为未定义，不能伪造数值。

建议模型通过统一接口接收 batch 并返回 logits；不符合接口的模型通过 adapter 接入，而不是为每个模型复制一套评测代码。

### 3. 建立 baseline

最终 baseline 需要根据团队实际采用的模态确定。建议至少包含以下层次：

| Baseline | 目的 |
|---|---|
| 常数或 prevalence baseline | 检查评测流程和类别不平衡下的最低参照 |
| 每种模态的单模态 baseline | 判断各模态独立提供的信息量 |
| 标准图像模型 | 建立 CXR 单模态参照，例如统一训练设置下的 CNN/ResNet/DenseNet |
| 简单多模态融合 | 建立不使用团队新 module 的参照，例如特征拼接加 MLP 或 late fusion |
| 团队提出的多模态模型 | 与简单融合和各单模态结果进行公平比较 |

baseline 和主模型必须使用相同的数据版本、split、预处理和评测代码。baseline 的训练由 benchmark 负责人还是通用训练框架负责人执行，需要组内明确；无论由谁训练，最终结果统一由本目录的 evaluation 框架生成。

### 4. 评测每一版团队模型

每个模型版本应登记并保存：

- 唯一实验名称和模型版本。
- 使用的模态和模型配置。
- checkpoint 路径及 SHA-256。
- 对应 Git commit、训练 seed 和数据 index hash。
- validation set 的 checkpoint/threshold 选择依据。
- validation 和 test 指标。
- 失败、缺失或不兼容原因。

最终维护统一结果表，保证不同版本可以直接横向比较，而不是在不同脚本、不同 split 或不同阈值下各自汇报结果。

## 指标与统计

院内死亡通常是类别不平衡任务，因此不能只汇报 accuracy。建议核心结果包括：

- AUROC。
- AUPRC。
- Sensitivity 和 specificity。
- Precision、recall 和 F1。
- Confusion matrix。
- Brier score 或 calibration curve。
- 95% confidence interval，优先按患者进行 bootstrap。

所有依赖分类阈值的指标必须先在 validation set 按预先约定的规则确定阈值，然后将该阈值原样应用到 test set。若计算多个随机种子的实验，应汇报均值、标准差和每次独立运行结果。

在样本量允许时，可以补充性别、年龄段、AP/PA view 和预测时间窗口等 subgroup 分析。子组结果必须同时报告样本量和不确定性，避免对小样本差异作过度解释。

## 输入接口

### 数据流水线提供

- 完成真实数据运行后的 `run_dir` 和只读原图挂载。
- `index.csv`、`index_train.csv`、`index_val.csv`、`index_test.csv`。
- `split_assignments.csv` 和 `dataset_spec.json`。
- `qa_report.json`、`run_manifest.json` 和 `SUCCESS.json`。
- `test_dataloader.py` 的通过记录。

当前 DataLoader batch 已提供 `image`、`label`、`sample_weight`、`sample_id`、`subject_id`、`hadm_id`、`study_id` 和 `dicom_id`。`dischtime`、`deathtime` 等结局后信息只能用于数据审计，不能作为模型特征。

### 训练负责人提供

- 可加载的 checkpoint。
- 模型构造配置和依赖版本。
- 模型输入、输出 shape 和正类定义。
- 训练 seed、Git commit 和使用的数据版本。
- 最佳 checkpoint 的 validation 选择规则。

### 多模态负责人提供

- 多模态 batch schema 和各字段含义。
- 各模态的 preprocessing 参数。
- 模态缺失时的处理方式。
- 与 evaluation adapter 对接的 inference 接口。

如果多模态输入包含实验室指标、生命体征或其他临床时序数据，所有特征必须以胸片 `study_time` 为截止点，并且标准化、填补等统计量只能在 train set 上拟合。

## 输出与交付物

建议每次评测生成独立、不可覆盖的结果目录：

```text
outputs/<dataset-version>/<model-version>/<run-id>/
  predictions.csv
  metrics.json
  metrics.csv
  evaluation_config.json
  run_manifest.json
  roc_curve.png
  precision_recall_curve.png
  calibration_curve.png
  confusion_matrix.png
  report.html
  SUCCESS.json
```

其中 `predictions.csv` 至少包含 `sample_id`、`subject_id`、`hadm_id`、真实标签、预测概率、split、模型版本和 checkpoint hash。结果文件不应包含受限的原始数据或账号凭证。

最终交付包括：

- 可复用的 evaluation 代码和配置。
- baseline 注册与运行方式。
- evaluation 单元测试和合成数据端到端测试。
- 各 baseline 和团队模型版本的逐样本预测。
- 指标、置信区间和图表。
- 统一 leaderboard。
- 可直接用于报告和汇报的结果总结。

## 完成标准

- 同一条评测命令可以测试单模态和多模态 checkpoint。
- 每次运行都能证明使用了同一份冻结数据和患者划分。
- validation 阈值选择与最终 test 评测严格分离。
- 所有模型输出统一格式的 predictions、metrics 和 manifest。
- benchmark 至少覆盖单模态、简单融合和团队提出的多模态模型。
- 结果可以从保存的配置、checkpoint 和代码版本复现。
- 报告清楚区分合成测试结果与真实 MIMIC 数据结果。

## 当前状态与后续工作

通用评分、HTML 报告、prevalence baseline 和模型 adapter 第一版已经实现，并已通过合成数据自动测试。当前尚无真实课程数据结果，也尚未接入另外两位成员的训练 checkpoint；现有测试数值不能作为真实 benchmark 表现。

下一步需要：

1. 等数据负责人完成真实服务器运行并冻结首版 index/split。
2. 与两位建模负责人确定单模态和多模态 checkpoint 接口。
3. 确定主指标、阈值选择规则、评测聚合单位和 baseline 清单。
4. 在服务器真实数据目录运行 prevalence baseline，确认端到端报告产物。
5. 依次接入图像单模态、另一模态、简单融合和团队多模态模型，并维护统一结果表。
