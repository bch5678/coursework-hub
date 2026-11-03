# 项目自带数据：实际文件与接口核对

项目中确实有已抽取的数据，不只有 `tests/` 下的几个 embedding 文件。
此前启动脚本只按原图/临床时序的 frozen-run 接口查找 `SUCCESS.json`，
漏查了 data-pipeline 目录中的两个 `.7z` 包，不能据此判断项目没有可用数据。

## 已检查并解包的文件

| 源文件 | label | 住院数 | 患者数 | 阳性数 |
| --- | --- | ---: | ---: | ---: |
| `bn5212-data-pipeline/mimic_mm_data_icu.7z` | `icu` | 1054 | 600 | 189 |
| `bn5212-data-pipeline/mimic_mm_data.7z` | `mortality` | 1156 | 651 | 47 |

已分别解压到训练目录的 `data/project_bundles/icu/` 和
`data/project_bundles/admission/`，原压缩包保持不变，解包目录被 Git 忽略。
以上计数来自数据包内 `results_admission_history/metrics.json`，
并核对了临床表、embedding 和 OOF 表的 hadm_id 及顺序一致。
这些统计不能替代时间窗口和标签生成规则审计；尤其不能把 1054 次住院包直接当作
项目总结中的 244 次住院队列或训练 README 的 198 次 ICU 住院队列。

两个压缩包均实际包含：

```text
clinical_admission.csv
clinical_history.csv
clinical_discharge.npy
clinical_schema.json
embeddings_admission_history.pt
results_admission_history/
  metrics.json
  oof_predictions.csv
  cross_attention_maps_r0.npy
```

`clinical_admission.csv` 是每次住院一行的原始特征表，含数值缺失和类别列；
`clinical_history.csv` 为既往病史特征。它们与现有 table provider 要求的
`hadm_id/hour/variable/value` 临床时序长表是不同接口，不能仅替换路径直接加载。

## 影像和临床 embedding 的实际形状

| `embeddings` 的 key | ICU 包 | mortality 包 |
| --- | --- | --- |
| `input_clinical` | [1054,146] | [1156,147] |
| `input_image` | [1054,64] | [1156,64] |
| `clinical_only` | [1054,64] | [1156,64] |
| `image_only` | [1054,64] | [1156,64] |
| `cross_attention` | [1054,128] | [1156,128] |

影像是已经池化的向量，没有空间 token 轴。
`cross_attention_map` 是 `[N,7,7]` 的注意力权重，不能恢复原图 patch 特征。
数据包 `note` 明确写明学习得到的 embedding 使用全部住院标签训练，不能
重新切分后当成独立测试的输入证据。
即便 `input_image` 被原文档标为未训练，已有降维/标准化统计量是否来自训练折仍须核对。

data-pipeline 的 `tests/README.md` 在“输出文件”中说明了
`image_features_densenet121.pt`，其中应含 `tokens[N,49,1024]` 和 hadm_id。
**该文件不在这两个压缩包的实际文件列表内，也没有出现在当前项目的其他路径中。**
它是复用原 DenseNet 空间特征、实现真实 patch attention 所需的输入。
ViT 196 个 patch 的原方案同样需要对应的原始图像或 ViT token 缓存。
已有原图时无需单独准备 token 文件：现有 ViT encoder 会在前向中直接生成 patch。
`run_three_multimodal.sh --prepare --pipeline-config ...` 已支持将原数据管线、
临床抽取、缓存及可选的外部临床预训练接到三模型启动前，详见
[同一命令运行预处理](RUN_THREE_MULTIMODAL.md)。

## 原实验划分和 benchmark

OOF 表保留了 `fold_r0`、`fold_r1`、`fold_r2`，每列为 5 个 fold。
已核对同一患者在每个 repeat 内只属于一个 fold。要复现实验应保留这些分折，
不能自行重新随机生成后声称与原结果可比。

embedding 的 `split` 仅含 `fit` 和 `early_stopping`，没有独立 `test`。
不能直接改名为 train/val/test 并生成 `SUCCESS.json`；
现有 frozen-run benchmark 与这些重复 OOF 结果之间需要显式的数据/评估适配。
`clinical_discharge.npy` 含出院汇总，不用于 t0 时刻的前瞻性预测。

## 接入边界

- 临床表、既往史、样本标识、标签和原患者 folds 可以复用，需要适配器。
- 原图 frozen-run 的 `SUCCESS.json` 检查继续保留，保护已经使用该接口的旧训练流程。
- 保持 patch 模块设计时，应使用文档中的未池化 image token 文件及其标识映射。
- 如改用现有 pooled 影像向量，应显式作为另一种特征实验，并单独说明预处理、
  注意力方向/退化问题和评估限制；不能伪装成 49/196 个真实 patch。
