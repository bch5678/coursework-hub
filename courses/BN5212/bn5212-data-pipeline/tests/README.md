# MIMIC-IV × MIMIC-CXR 多模态院内死亡预测（mimic_mm）

以住院为单位，把住院期间的第一张正位胸片和"截至拍片时刻"的临床数据结合，预测院内死亡。流程分为 4 步，对应以下 7 个环节：MIMIC-IV / MIMIC-CXR 匹配、队列构建、临床特征提取、图像选择、缺失值预处理、标签构建、按患者划分。

| 步骤 | 脚本 | 内容 | 耗时 |
| --- | --- | --- | --- |
| 0 | `..\build_mimic_iv_embeddings.py` | MIMIC-IV 哈希特征；现在另外输出"入院时可知"和"出院汇总"两个矩阵 | 约 4 分钟 |
| 1 | `step1_cohort.py` | 匹配、队列构建、图像选择、标签、按患者的重复折 | 约 20 秒 |
| 2 | `step2_clinical.py` | 临床特征（两组） | 首次约 2 分钟（扫描 labevents，结果会缓存） |
| 3 | `step3_images.py` | 冻结的 DenseNet121，提取 7×7 空间 token | 约 3 分钟 |
| 4 | `step4_train.py` | 缺失值预处理、9 个模型、3×5 折交叉验证 | 5–10 分钟（RTX 4070 SUPER） |
| 5 | `step5_export_embeddings.py` | 在全部住院上训练最终模型，导出分开的和融合的 embedding | 约 30 秒 |

```powershell
$py = "D:\Users\Bai Chenhao\AppData\Local\Programs\Python\Python312\python.exe"
cd "E:\School\# Doctor\BN5212\mimic_mm"
& $py step1_cohort.py
& $py step2_clinical.py
& $py step3_images.py
& $py step4_train.py                                  # 只用入院时可知的临床特征（主分析）
& $py step4_train.py --features admission+discharge   # 另加出院汇总（回顾性对照）
& $py step5_export_embeddings.py                      # 导出 embedding（同样可加 --features admission+discharge）
```

## Embedding（第 5 步）

每次住院一行（1156 行，顺序与 `cohort.csv` 相同）。入院时可知特征版本在 `mimic_mm_data\embeddings_admission.pt`，加出院汇总的版本在 `embeddings_admission_discharge.pt`。

| 键 `embeddings[...]` | 维度 | 类型 | 内容 |
| --- | --- | --- | --- |
| `input_clinical` | 109（加出院汇总后 621） | 分开 · 未训练 | 预处理后的临床向量：填补、z-score、缺失指示变量、one-hot |
| `input_image` | 64 | 分开 · 未训练 | DenseNet121 token 经 PCA 降到 64 维后的均值 |
| `input_early_fusion` | 173（685） | 融合 · 未训练 | `[input_clinical \| input_image]` |
| `clinical_only` | 64 | 分开 · 学习得到 | 临床单模态模型的池化表示 |
| `image_only` | 64 | 分开 · 学习得到 | 影像单模态模型的池化表示 |
| `late` | 128 | 融合 · 学习得到 | `[clinical_only \| image_only]`（concat 模块的输出与此相同，因此不单独导出） |
| `gated` | 64 | 融合 · 学习得到 | 门控多模态单元的输出 |
| `cross_attention` | 128 | 融合 · 学习得到 | 双向交叉注意力后，[临床 token 均值 \| 影像 token 均值] |

同一文件中还有：`risk_in_sample`（各模型的死亡风险）、`cross_attention_map` [1156, 7, 7]、`hadm_id` / `subject_id` / `study_id` / `mortality`，以及 `split`（`fit`：训练；`early_stopping`：早停用的验证集）。更原始的 1024 维 DenseNet 向量在 `image_features_densenet121.pt` 的 `pooled` 键里。

```python
import torch
pkg = torch.load(r"E:\School\# Doctor\BN5212\BN5212\mimic_mm_data\embeddings_admission.pt")
Z = pkg["embeddings"]["cross_attention"]      # 融合 embedding
Zc = pkg["embeddings"]["clinical_only"]       # 分开：临床
Zi = pkg["embeddings"]["image_only"]          # 分开：影像
y = pkg["mortality"]
```

**注意**：导出用的模型在全部住院上训练过，见过所有标签，所以这些 embedding 适合做可视化、聚类，或作为其他任务的输入，**但不能再用它们在同一批住院上评估死亡预测性能**，评估请以第 4 步的交叉验证结果为准。学习得到的 embedding 取自 5 个种子中的第 1 个，因为不同种子的向量不在同一空间；`risk_in_sample` 是 5 个种子的平均。

所有设计参数都在 `common.py` 中（时间窗、backbone、折数、种子），训练超参数在 `step4_train.py` 开头。所有步骤都已运行过，输出位于 `BN5212\mimic_mm_data\`。

## 1. 匹配（MIMIC-IV / MIMIC-CXR）

- 两个库共用 `subject_id`，每位患者的日期偏移相同。
- 拍片时间取自 `dataset.zip` 中 DICOM 头的 `StudyDate` + `StudyTime`（只读文件头）；同一 study 取最早的时间。
- 若 study 时间落在同一患者某次住院的 `[min(edregtime, admittime), dischtime]` 窗口内，就匹配到该次住院。窗口必须包含急诊段：65% 的入选胸片拍于正式入院之前的急诊阶段。
- 同一 study 落入两次重叠住院时，归到开始时间较晚的那次。

## 2. 队列构建与流程（`cohort_flow.csv`）

| 步骤 | 数量 |
| --- | --- |
| dataset.zip 中的 CXR 患者 / study / 图像 | 1000 / 3351 / 5534 |
| 其中有 MIMIC-IV 住院记录的患者 | 786 |
| 这些患者所有住院中的院内死亡（不论是否拍片） | 3214 次住院中 64 例 |
| 匹配到住院窗口的 study | 2356 个 study，涉及 1172 次住院 |
| 住院期间有正位（PA/AP）study 的住院 | 1156 |
| 排除年龄 < 18、排除 t0 或之前已记录死亡 | 未排除任何住院 |
| **最终队列** | **1156 次住院 / 651 名患者 / 47 例死亡** |

**关于"约 1000 名患者、64 个死亡事件"**：1000 是 CXR 数据集的患者总数；64 是这些患者在全部 3214 次住院中的院内死亡数，其中很多次住院根本没有拍胸片。要求"本次住院有胸片"之后，可用于多模态建模的是 651 名患者、47 个死亡事件。写报告时建议采用这组数字，并附上上面的流程表。

- 单位是住院。222 名患者有多次住院（最多 20 次），同一患者的所有住院始终在同一折中。
- 如需"早期预测"版本，可以设置 `common.MAX_IMAGE_HOURS_AFTER_ADMIT = 24` 或 `48`，只保留入院后 24 或 48 小时内拍的片子。事先估算过，24 小时时约 998 次住院、38 例死亡。

## 3. 图像选择与预测时间点 t0

- 每次住院取**第一个含正位图像的 study**，其拍摄时间就是预测时间点 **t0**。
- 该 study 中优先取 PA，没有 PA 再取 AP，因为 PA 的成像质量更好。入选图像中 AP 682 张、PA 474 张。
- 临床特征只能使用 t0 之前的信息（出院汇总组除外，该组是回顾性的，见下文）。

## 4. 临床特征提取（`step2_clinical.py`）

**入院时可知组 `admission`**（47 个数值特征 + 7 个类别特征）：

| 组 | 特征 |
| --- | --- |
| 人口学 / 入院 | 年龄、性别、种族（合并为 6 类）、语言、保险、婚姻状况、入院类型、入院来源、是否经急诊、入院到 t0 的小时数 |
| 既往史（只用本次入院**之前**已出院的住院） | 既往住院次数（总计 / 近 1 年）、距上次出院的天数、既往 ICU 次数、17 类 Charlson 合并症与 CCI 总分（Quan 2005 ICD-9/10 编码） |
| 化验（t0 前 24 小时内最后一次值） | WBC、Hb、PLT、RDW、Na、K、Cl、HCO3、AG、BUN、Cr、Glu、Ca、Mg、乳酸、INR、PTT、总胆红素、ALT、白蛋白、肌钙蛋白 T、pH |

化验按 `subject_id` 匹配，因为急诊化验没有 `hadm_id`。各项覆盖率：常规生化和血常规约 77%，INR/PTT 45%，乳酸 27%，肌钙蛋白 9%，pH 8%；80% 的住院至少有一项化验。

**出院汇总组 `discharge`**（可选，回顾性）：取自 `mimic_iv_discharge_embeddings.npz` 的 512 维哈希向量，包括住院时长、本次 ICD 诊断和手术码、科室、ICU 住院。这些信息要到出院时才确定，**不能用于 t0 时刻的预测**，只作为上限和泄漏对照。

`build_mimic_iv_embeddings.py` 现在输出三个矩阵：原来的合并矩阵（逐位不变）、`mimic_iv_admission_embeddings.npz`（256 列，入院时可知）和 `mimic_iv_discharge_embeddings.npz`（512 列，出院汇总）。

## 5. 缺失值预处理（`step4_train.py`，每折只在训练行上拟合）

1. 偏态化验取 `log1p`：WBC、PLT、BUN、Cr、Glu、乳酸、INR、PTT、胆红素、ALT、肌钙蛋白。
2. 按训练行的 1% / 99% 分位数截尾。
3. 用训练行的**中位数填补**，同时保留**缺失掩码**：逻辑回归作为 0/1 指示变量使用；神经网络中，缺失特征的 token 替换为一个可学习的"缺失"嵌入。
4. 用训练行的均值和标准差做 z-score，并截断到 ±5。
5. 类别变量缺失记为 `UNKNOWN`；训练行中出现少于 10 次的类别并入"其他"。
6. 图像 token 先按通道做 z-score，再做 PCA 降到 64 维，两者都只用训练行拟合。

核对脚本验证过：修改测试行的原始值，训练行的预处理输出完全不变。

## 6. 标签

`mortality` = 该次住院的 `hospital_expire_flag`，与 `deathtime` 是否非空逐行一致（0 处不一致）。t0 当时或之前已记录死亡的住院会被排除，本队列中没有这种情况。

## 7. 按患者划分与交叉验证

- **外层**：5 折 `StratifiedGroupKFold`（按 `subject_id` 分组，按患者是否死亡分层），用 3 个不同种子重复 3 次（`fold_r0..r2`）。三次划分相互独立（ARI ≈ 0），每折 8–10 例死亡。
- **内层**：外层训练部分再按患者划出 20% 作为验证集，用于早停（神经网络）；逻辑回归的 C 由按患者分组的 3 折交叉验证选择。
- **报告**：每个模型报告 3 次重复各自 AUROC 的均值 ± 标准差；每次住院在 3 次重复中各有一个折外预测，取平均后计算 AUROC，并给出按患者 bootstrap（1000 次）的 95% CI。配对差值在同一批 bootstrap 样本上计算。

## 8. 模型

**Backbone**：torchvision DenseNet121（ImageNet 预训练，约 800 万参数），**冻结**，只提取一次特征。每张图得到最后一层 7×7×1024 的特征图，即 49 个空间 token。可在 `common.BACKBONE` 中改为 `resnet18`。

**模块化 fusion**（d = 64）：

```
临床: 每个特征一个 token（FT-Transformer tokenizer，缺失时用"缺失"嵌入）──┐
影像: 49 个 token → PCA 64 → Linear + 位置嵌入 ──────────────────────────┤
                                                                         ▼
阶段 1  clinical_only / image_only：token 取均值 → MLP 头（各自训练、各自早停）
阶段 2  冻结阶段 1 的模型：
        late            = w1·logit_clinical + w2·logit_image + b
        concat / gated / cross_attention = late + 交互残差（输出层零初始化）
          concat           [mean(临床), mean(影像)] → MLP
          gated            门控多模态单元（逐维 sigmoid 门）
          cross_attention  双向多头交叉注意力（4 头）：临床 token 查询影像区域，
                           影像区域查询临床 token；残差 + LayerNorm + FFN
```

- **为什么分两阶段**：从零开始联合训练时，影像分支几个 epoch 就过拟合，早停在临床分支学到东西之前就触发。当临床特征很强时（出院汇总组，临床单模态 0.935），联合训练的融合模型只有约 0.80，几乎完全忽略了临床信息。
- **为什么用 late + 残差**：只冻结编码器、从零训练融合头仍会丢信息（出院汇总组约 0.86–0.89）。改为以 late fusion 为起点、交互模块只学残差后，融合模型不再低于单模态。它与 late 的差值直接反映了跨模态交互的额外价值。
- 每折每个神经网络模型训练 5 个种子，取预测的平均值，以降低早停带来的方差。
- **基线**：同一套预处理特征上的 L2 逻辑回归（`clinical_lr` / `image_lr` / `concat_lr`）。
- 训练超参数（lr 1e-3、AdamW、wd 1e-2、dropout 0.2、patience 20）是事先设定的。开发过程中只比较过内层验证损失，各组之间没有实质差别，从未根据测试折挑选参数。

## 9. 结果

1156 次住院、47 例死亡。表中 "AUROC" 为 3 次重复的均值 ± 标准差；"平均预测 AUROC" 为折外预测在 3 次重复间取平均后的 AUROC，附 95% CI；AUPRC 基线（患病率）为 0.041。

**主分析：只用入院时可知的临床特征**（`results_admission\`）

| 模型 | AUROC | 平均预测 AUROC [95% CI] | AUPRC | 与 clinical_only 之差 [95% CI] |
| --- | --- | --- | --- | --- |
| clinical_lr | 0.719 ± 0.016 | 0.734 [0.660, 0.804] | 0.123 | +0.024 [−0.027, +0.078] |
| image_lr | 0.758 ± 0.025 | 0.750 [0.671, 0.821] | 0.146 | +0.041 [−0.053, +0.125] |
| concat_lr | 0.803 ± 0.008 | 0.813 [0.745, 0.871] | 0.182 | +0.104 [+0.043, +0.166] |
| clinical_only | 0.695 ± 0.032 | 0.710 [0.639, 0.775] | 0.083 | — |
| image_only | 0.776 ± 0.009 | 0.785 [0.710, 0.849] | 0.163 | +0.075 [−0.014, +0.159] |
| late | 0.791 ± 0.026 | 0.809 [0.747, 0.866] | 0.155 | +0.100 [+0.044, +0.153] |
| concat | 0.794 ± 0.025 | 0.810 [0.745, 0.868] | 0.163 | +0.101 [+0.039, +0.161] |
| gated | 0.794 ± 0.026 | 0.811 [0.746, 0.869] | 0.161 | +0.101 [+0.041, +0.162] |
| cross_attention | 0.792 ± 0.029 | 0.809 [0.744, 0.866] | 0.159 | +0.100 [+0.037, +0.160] |

交互模块与 late 之差：concat +0.001 [−0.010, +0.011]，gated +0.001 [−0.008, +0.011]，cross_attention −0.000 [−0.011, +0.011]。

- 胸片确实带来了增量：所有多模态模型都比只用临床特征高约 0.10，且 CI 不含 0。
- 在 47 个事件的条件下，三种交互模块（包括 cross-attention）都**没有超过** late fusion。最好的简单模型（concat_lr，0.813）与最好的神经网络融合模型（0.811）相当。

**回顾性对照：另加出院汇总**（`results_admission_discharge\`）

| 模型 | 平均预测 AUROC [95% CI] | AUPRC |
| --- | --- | --- |
| clinical_lr | 0.960 [0.940, 0.974] | 0.491 |
| concat_lr | 0.961 [0.943, 0.975] | 0.490 |
| clinical_only | 0.935 [0.906, 0.959] | 0.389 |
| late | 0.925 [0.890, 0.952] | 0.383 |
| concat / gated / cross_attention | 0.913 / 0.915 / 0.921 | 0.345 / 0.354 / 0.377 |

AUROC 从 0.81 跳到 0.96，原因是出院诊断码中含有几乎等同于结局的编码（如 Z51.5 姑息治疗、Z66 不复苏）。**这组结果不能当作预测性能报告**，只用来说明泄漏的影响。

`cross_attention_maps_r0.npy` 保存了第 0 次重复中，每次住院在测试折上的 7×7 注意力图（临床 token 对影像区域的注意力，5 个种子取平均），可叠加到原图上做可视化。

## 10. 局限

- **事件数少**：47 例死亡，每折 8–10 例，所有 CI 都较宽。模型间 0.01 量级的差别不应解读为优劣。
- **冻结的 ImageNet backbone** 没有针对胸片预训练。影像信号中有相当一部分可能来自 AP（床旁片）与 PA 的差异，也就是病情严重程度的代理，而不是具体的肺部表现。
- **概率未校准**：训练时按类别加权，预测概率会系统性偏高，因此只报告区分度（AUROC / AUPRC）。
- **入院时特征不含生命体征**：本地没有 MIMIC-IV-ED 模块，ICU 的 chartevents 只覆盖 ICU 患者。
- 同一患者的多次住院并非独立样本；已按患者划分和按患者 bootstrap，但没有做进一步的聚类调整。

## 输出文件（`BN5212\mimic_mm_data\`）

| 文件 | 内容 |
| --- | --- |
| `cxr_images.csv` | 5534 张图像的 subject / study / dicom、拍摄时间和视图 |
| `cohort.csv` | 队列（每行一次住院）：ID、所选图像、t0、住院时间、年龄、`mortality`、`fold_r0..r2` |
| `cohort_flow.csv` | 纳入 / 排除流程 |
| `labs_cxr_subjects.csv` | 从 labevents 缓存的 1000 名 CXR 患者的 22 项化验 |
| `clinical_admission.csv`, `clinical_schema.json` | 入院时可知特征（含 NaN）与特征说明 |
| `clinical_discharge.npy` | 出院汇总哈希向量（与 cohort 行对齐） |
| `image_features_densenet121.pt` | `tokens` [1156, 49, 1024] (fp16)、`pooled`、`hadm_id` |
| `results_*\metrics.json`, `oof_predictions.csv`, `cross_attention_maps_r0.npy` | 指标、每次重复的折外预测、注意力图 |
| `embeddings_admission.pt`, `embeddings_admission_discharge.pt` | 分开的和融合的 embedding（见上文"Embedding"一节） |

上一版的 `build_mimic_fused_embeddings.py` 和 `train_mimic_fused.py`（肺炎 + 死亡多任务，VGG16 向量）已被本流程取代，保留仅供参考。
