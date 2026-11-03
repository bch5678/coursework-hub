# 多模态 v3：训练与评测

当前保留版本为 `transfer_multimodal_v3_20261030_161749`，任务是预测拍片后的同次住院 ICU 转入。
入口统一为 `scripts/run_icu_transfer.sh`，调用 `run_three_multimodal.sh` 顺序运行
Concat、Joint attention、Cross-attention Adapter。两个脚本默认均使用 v3 配置。

## 运行

在训练项目目录中激活已安装训练、benchmark 和 timm 的 Python 环境。
源码已移至 `src/`；使用旧安装的环境先执行 `python -m pip install --no-deps -e .` 更新包映射：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_icu_transfer.sh \
  --prepare --device cuda --run-id transfer_multimodal_v3_rerun
```

使用新 run-id 保存复现实验；现有 v3 原始结果继续保留。`PYTHON_BIN=/path/to/python`
可选择解释器，GPU 由调用者的 `CUDA_VISIBLE_DEVICES` 决定。
如果原始数据已处理好，去掉 `--prepare` 即可复用固定队列、临床表与图像缓存。

默认基础配置为 `configs/icu_transfer/cross_attention_adapter.json`，由 v3 实际保存配置整理；
另有 `concat.json` 和 `joint_attention.json` 供单模型训练。三者共享数据、编码器、优化器及 seed，
只改变实验名和融合方式。学习率 `0.0003`，`min_epochs_before_selection=0`，
最多 100 epochs，warmup 5，patience 15，按最低 validation loss 选择 checkpoint。

仅准备数据：

```bash
bash scripts/run_icu_transfer.sh --prepare-only --run-id transfer_prepare
```

默认预处理配置为 `configs/preparation/icu_transfer_pipeline.json`。
输入与标签说明见 [ICU_TRANSFER.md](ICU_TRANSFER.md)，数据下载见
[DOWNLOAD_COURSE_DATA.md](DOWNLOAD_COURSE_DATA.md)。
输入文件必须是真实完整的原图和 MIMIC 表；派生 embedding 包不能代替它们。
转入任务不使用旧死亡任务的临床预训练权重，也无需 `--pretrain-clinical`。

## 路径与参数

服务器路径不同时，可追加以下参数：

```bash
  --run-dir /path/to/published/run \
  --clinical-source /path/to/clinical_features.csv \
  --image-cache /path/to/images.json
```

`--prepare` 时还可传 `--pipeline-config /path/to/pipeline.json`、`--mimic-root /path/to/mimiciv`。
训练脚本的相对路径以训练项目为基准；pipeline JSON 的路径以该 JSON 所在目录为基准。
`--expected-index-sha256` 可核对预处理队列；原始 v3 index 为
`e80bc703a0bc108ae77566cb6412d7e004dc8ed1fa5dbe747d11aad8d0e9b0be`。
入口检查 `SUCCESS.json`、index hash 和任务类型，缺失数据时应修复输入或重新准备数据。

用 `--set SECTION.KEY=VALUE` 调整配置时，请使用新的 run-id 并记录差异。
`--include-unimodal` 可追加同队列单模态对照；`--freeze-from-unimodal` 会先训练对照再冻结编码器。
这些属于新增实验，不是现有 v3 的设置。v3 只冻结 ImageNet ViT backbone，其投影与临床分支参与训练。

## 评测与保存位置

固定 train 拟合 → val 选择最佳权重 → 导出 val/test 预测 → benchmark 在 val 按 Youden J
选择阈值 → 评分 test。默认按住院聚合，患者 bootstrap 2,000 次，seed 5212。
测试分数不参与 checkpoint 选择。这是固定划分评估，不能当作 nested CV 或 OOF 结果。

```text
outputs/three_multimodal/<run-id>/
  plan.json
  configs/<model>.json
  training/<model>/fit/
    checkpoint_best.pt
    checkpoint_last.pt
    config.json
    predictions_val.csv
    predictions_test.csv
    run_manifest.json
    figures/
  benchmark/<model>/
    metrics.json
    predictions.csv
    report.html
    run_manifest.json
    SUCCESS.json
  logs/
  validation_comparison.csv
  benchmark_leaderboard.csv
  benchmark_leaderboard.html
```

使用完全相同命令加 `--resume` 可跳过已完成阶段并保留权重；未完成训练会从头开始。
已有成功 benchmark 不覆盖，配置、数据或评测参数变化时须使用新的 run-id。
历史 v3 使用的完整命令保存在其 `outputs/training_launches/` 记录中；
对原批次续跑须沿用原始命令和配置，新的相对路径配置用于新批次复现。

可分享的汇总表与结果索引见 [results/multimodal_v3](../results/multimodal_v3/README.md)。
完整权重和患者级输出保留在本地 `outputs/`，该目录继续由 Git 忽略。
