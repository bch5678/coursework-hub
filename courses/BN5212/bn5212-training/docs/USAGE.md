# 环境与通用训练接口

当前多模态版本为 ICU 转入 v3，使用 [三模型运行说明](RUN_THREE_MULTIMODAL.md)。
原有院内死亡单模态基线保留其独立配置与结果，不能直接作为 v3 的同队列对照。

## 安装

在训练项目目录，使用 Python 3.10 或以上：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,pretrained]'
python -m pip install -e ../benchmark-evaluation
python -m pip install -r ../bn5212-data-pipeline/requirements.txt
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活相同环境。
GPU 环境按服务器已有 PyTorch/CUDA 安装配置使用。

## 源码目录与模块调用

源码直接位于 `src/`，融合实现位于 `src/fusion/`。
安装配置将 `src/` 映射到包名 `bn5212_training`；包名是导入接口，无需与磁盘目录同名。
因此命令继续使用 `python -m bn5212_training.prepare` 等形式，`bn5212-train` 等入口也保持一致。
源码中的相对导入通过包名解析，请使用安装后的命令或 `python -m` 调用模块。

已安装旧目录版本的环境需要在训练项目目录重新执行一次：

```bash
python -m pip install --no-deps -e .
```

启动脚本通过安装后的包映射加载训练模块，不再将训练目录的 `src/` 加入 `PYTHONPATH`。
`PYTHON_BIN` 指定的解释器也需要完成这一步安装。

## 配置入口

| 实验 | 配置 |
| --- | --- |
| 多模态 v3 Concat | `configs/icu_transfer/concat.json` |
| 多模态 v3 Joint attention | `configs/icu_transfer/joint_attention.json` |
| 多模态 v3 Cross-attention Adapter | `configs/icu_transfer/cross_attention_adapter.json` |
| 院内死亡 Clinical-only | `configs/icu/clinical_only.json` |
| 院内死亡 Clinical-only，外部预训练 | `configs/icu_pretrained/clinical_only.json` |
| 院内死亡 CXR-only | `configs/icu/cxr_only.json` |
| 合成数据功能验证 | `configs/synthetic/` |

临床变量、窗口与时点由各任务配置指定；详见 [ICU_TRANSFER](ICU_TRANSFER.md)、
[CLINICAL_FEATURE_SPEC](CLINICAL_FEATURE_SPEC.md) 与单模态的 [DATA_STRATEGY](DATA_STRATEGY.md)。
`configs/` 下的顶层单模态文件保留通用服务器模板；`configs/icu/` 保留既有基线及编码器消融。

## 本地合成数据验证

```bash
python ../bn5212-data-pipeline/scripts/make_synthetic_data.py --help
python -m pytest
```

测试自动构造合成数据。手动演示时先按 data-pipeline README 构建合成数据 run，再执行：

```bash
bn5212-train --config configs/synthetic/cross_attention_adapter.json \
  --run-dir /path/to/synthetic/processed --run-id smoke
```

`run-dir` 必须是包含 `SUCCESS.json`、`dataset_spec.json`、`index.csv` 和固定患者划分的已发布数据目录。
合成数据的分数只验证接口，不是临床结果。

## 单模型训练与预测

```bash
bn5212-train --config configs/icu_transfer/concat.json --device cuda --run-id concat_v3_rerun
bn5212-predict --help
bn5212-summarize --help
```

可通过 `--set SECTION.KEY=VALUE` 覆盖配置，未知字段会报错。
训练输出包含 best/last checkpoint、实际配置、验证指标、val/test 预测、manifest、训练曲线及诊断图。
独立预测使用 checkpoint 中保存的模型配置与训练标准化参数。
正式 test 指标由 benchmark-evaluation 计算；统一三模型入口会自动完成评测和报告导出。

## 单模态预训练与交叉验证

院内死亡单模态实验的临床特征抽取、外部预训练、患者分组 nested CV 和结果导出命令
保留在 [README](../README.md)。

```bash
bn5212-crossval --config configs/icu_pretrained/clinical_only.json --output-dir outputs/nested_cv --run-id v3
bn5212-crossval --config configs/icu/cxr_only.json --output-dir outputs/nested_cv --run-id v2
python scripts/compare_crossval.py --help
python scripts/export_results.py --help
```

`scripts/run_all_experiments.sh` 和 `.ps1` 默认只运行两个单模态基线。
`scripts/export_results.py` 仅导出原有单模态和编码器消融结果；v3 多模态汇总由统一入口生成。

## 数据准备与下载

转入任务的一键准备、训练、评测使用 `scripts/run_icu_transfer.sh --prepare`。
下载工具见 [DOWNLOAD_COURSE_DATA](DOWNLOAD_COURSE_DATA.md)，包内容见
[PROJECT_DATA_INVENTORY](PROJECT_DATA_INVENTORY.md)。
通用接口包括 `bn5212-extract-clinical`、`bn5212-build-image-cache`、
`python -m bn5212_training.prepare` 和 `python -m bn5212_training.pretrain`，参数见各命令 `--help`。

权重、临床表、患者级预测及原始数据继续保存在被 Git 忽略的本地目录。
代码扩展见 [FUSION_API](FUSION_API.md)；可分享的 v3 汇总见
[results/multimodal_v3](../results/multimodal_v3/README.md)。
