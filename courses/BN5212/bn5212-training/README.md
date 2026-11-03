# BN5212：多模态 v3 与单模态基线

本项目结合 MIMIC-IV 临床时间序列与 MIMIC-CXR 胸片，研究早期胸片之后、同次住院内的 ICU 转入预测。
核心问题是：加入胸片能否提高临床模型的预测能力，不同融合方式是否会影响增益？
当前多模态 v3 已完成 Concat、Joint attention 和 Cross-attention Adapter 的训练与评测。
本页集中记录模型设计、结果图表、后续验证建议和历史单模态背景。

## 阅读导航

- [快速运行与项目入口](#quick-start)
- [当前多模态 v3：模型设计、结果与后续验证](#multimodal-v3)
- [历史 ICU 转入单模态实验（244 次住院）](#historical-icu-transfer)
- [院内死亡单模态基线（198 次 ICU 住院）](#mortality-baselines)

当前 v3 使用 280 次住院的固定患者级划分；历史 ICU 转入单模态使用 244 次住院的嵌套交叉验证；
院内死亡基线使用另一任务的 198 次 ICU 住院。三组结果的队列与评估协议不同，不能直接作配对比较。

<a id="quick-start"></a>

## 快速运行与项目入口

源码直接放在 `src/`，如 `src/trainer.py`、`src/cli.py` 和 `src/fusion/`。
`pyproject.toml` 将该目录映射为 Python 包 `bn5212_training`，所以现有 `bn5212-*` 命令、
`import bn5212_training` 和 `python -m bn5212_training.<模块>` 继续可用。

目录调整后，在已配置依赖的环境中进入本目录，重新安装一次以更新包映射：

```bash
python -m pip install --no-deps -e .
```

首次配置环境见 [安装说明](docs/USAGE.md)。之后运行：

```bash
bash scripts/run_icu_transfer.sh --prepare --device cuda --run-id transfer_multimodal_v3_rerun
```

默认配置已与原始 v3 对齐：学习率 0.0003，从训练开始按最低验证 loss 选模。
已准备好数据时可去掉 `--prepare`，使用新 run-id 保存复现实验。

| 入口 | 内容 |
| --- | --- |
| [环境与通用接口](docs/USAGE.md) | 依赖安装、合成验证、训练与预测 |
| [三模型运行说明](docs/RUN_THREE_MULTIMODAL.md) | 数据准备、顺序训练、评测及续跑 |
| [v3 结果索引](results/multimodal_v3/README.md) | 汇总 CSV、本地完整输出及清理核验 |
| [ICU 转入任务](docs/ICU_TRANSFER.md) | 标签、18 个临床变量与时间对齐 |
| [v3 配置](configs/icu_transfer/) | 三个模型的复现参数 |
| [数据清单](docs/PROJECT_DATA_INVENTORY.md) | 课程数据包与训练接口 |

<a id="multimodal-v3"></a>

## 当前多模态 v3：模型设计与结果

- 整理日期：2026-11-01；目录清理更新：2026-11-03
- 实验版本：`transfer_multimodal_v3_20261030_161749`
- 完成时间：2026-10-30 16:21 UTC
- 任务：结合早期胸片与拍片前临床数据，预测同次住院后续 ICU 转入。

### 1. 核心结论

本轮三个模型均已完成训练及评测。Concat 的测试 AUROC（0.795）和 AUPRC（0.488）最高；Joint attention 的验证 AUROC（0.734）最高；Cross-attention Adapter 的测试 AUROC 为 0.671，当前没有表现出优于另外两种融合方法的优势。

测试集仅包含 37 次住院、33 位患者、7 个阳性事件，结果应视为小样本探索性证据。

本轮只包含三个多模态模型，尚不能回答“融合是否优于临床或胸片单模态”。历史实验与当前队列、划分和评估协议不同，不能直接作为配对对照。

### 2. 研究问题与数据设计

#### 2.1 预测时间点与标签

- 数据来源：MIMIC-IV v3.1、MIMIC-CXR v2.1.0 的本地可用子集。
- 研究单位：一次住院；每次住院选取一张符合条件的胸片。
- 预测时间点 `t0`：入院后 48 小时内首张符合条件的 AP/PA 胸片采集时间。
- 排除条件：同次住院在 `t0` 或更早已有 ICU 入科记录，包括曾进入 ICU、出科后再拍片的情况。
- 阳性：首次 ICU 入科严格晚于 `t0`，且严格早于同次住院死亡或出院截止时间。
- 阴性：同次住院中没有符合上述条件的后续 ICU 转入。

这里的结局是 ICU 转入，不是院内死亡；“入院后 48 小时”限定胸片选择时间，并不表示只预测未来 48 小时内的转入。

#### 2.2 队列与划分

使用 `icu_transfer_v1` 队列，按患者划分固定 train/val/test，随机种子为 5212。三个模型共享相同数据和患者划分。

| 划分 | 住院次数 / 胸片数 | 患者数 | ICU 转入阳性数 | 阳性比例 |
| --- | ---: | ---: | ---: | ---: |
| Train | 195 | 152 | 32 | 16.41% |
| Validation | 48 | 33 | 8 | 16.67% |
| Test | 37 | 33 | 7 | 18.92% |
| 合计 | 280 | 218 | 47 | 16.79% |

数据索引 SHA-256：`e80bc703a0bc108ae77566cb6412d7e004dc8ed1fa5dbe747d11aad8d0e9b0be`。

#### 2.3 临床输入与时间对齐

临床窗口为 `[t0−48h, t0)`，按 48 个小时格组织，共 18 个变量：

- 血液检验：白细胞、血红蛋白、红细胞压积、血小板、钠、钾、氯、碳酸氢盐、尿素氮、肌酐、葡萄糖、乳酸、pH、INR、总胆红素、白蛋白。
- 人口学变量：年龄、性别（Female）。

检验的 `charttime` 和 `storetime` 均须早于 `t0`，未知 `storetime` 的记录排除，避免使用拍片时尚不可获得的信息。年龄和性别在最后一个小时格记录一次。标准化统计量仅在训练患者上拟合，输入保留观测 mask；变量在窗口内完全缺失时保留对应位置，并用 mask 在融合时排除其贡献。

### 3. 模型总体设计

设计目标是在共享输入与训练设置下，比较三个层次的融合：分别汇总后拼接、所有 token 联合交互、由临床变量定向查询影像区域。

```mermaid
flowchart TD
    A[早期胸片] --> B[冻结的 ImageNet ViT-B/16]
    B --> C[可训练投影：768 → 64 维]
    D[18 个临床变量 × 48 小时及观测 mask] --> E[可训练临床线性编码器]
    E --> F[18 个 64 维临床 token]
    C --> G{融合方案}
    F --> G
    G --> H[Concat：分别池化并拼接]
    G --> I[Joint：联合 self-attention]
    G --> J[Adapter：临床 Q 查询影像 patch K/V]
    H --> K[对应模型的 MLP 分类头]
    I --> K
    J --> K
    K --> L[后续 ICU 转入概率]
```

图中的三个融合分支是三套独立模型，不是一个集成模型。

#### 3.1 共享编码器与分类头

**影像分支。** 实际使用 `timm` 的 `vit_base_patch16_224`，加载 ImageNet 预训练权重并冻结 backbone；输出 token 从 768 维投影到 64 维，该投影参与训练。影像 token 包含原始 ViT 的 CLS 和 patch token。配置中的其他通用编码器字段不代表本轮启用了其他 backbone。

**临床分支。** 使用 `linear_projection` 和 `per_variable` tokenization：每个变量的 48 小时数值与 48 小时观测指示拼接，形成 96 维特征，经线性层投影为 64 维，再加入变量身份嵌入和归一化。共生成 18 个变量 token。临床编码器在本轮训练集上学习，不加载旧死亡预测任务的预训练权重。

**分类头。** 隐藏维度 64、dropout 0.1，输出单个 logit，转换为二分类概率。Concat 的分类头输入为 128 维，另外两种模型为 64 维。

本轮训练的是影像投影、临床编码器、融合模块和分类头；并非两个模态编码器全部冻结的方案。

#### 3.2 Concat：分别汇总后拼接

1. 排除影像 CLS，对影像 patch token 取均值，得到 64 维影像表示。
2. 对有效临床 token 做 masked mean，得到 64 维临床表示。
3. 拼接为 128 维向量，直接送入分类头。

实际配置为 `concat_image_pooling=patch_mean`、`concat_project=false`，因此拼接后没有额外的 128→64 融合投影层。

设计动机：用结构简单的融合方式建立参照，检验两类全局信息组合是否已足够有效。其局限是池化后丢失变量与具体影像区域之间的细粒度对应关系。

#### 3.3 Joint attention：联合 token 交互

将新增的融合 CLS、全部影像 token（含影像自身 CLS）和临床 token 拼接，加入可学习的位置嵌入与模态嵌入，经 Transformer encoder 后读取融合 CLS 表示。

本轮为 1 层、4 个 attention heads、64 维嵌入、FFN 扩展比例 4、dropout 0.1；缺失临床 token 通过 padding mask 处理。

设计动机：允许影像与临床 token 在同一序列中学习模态内和模态间关系。相比 Concat，交互更灵活，但需要从当前有限样本中学习更多融合结构，是否有稳定收益必须通过重复评估确认。

#### 3.4 Cross-attention Adapter：临床查询影像

临床 token 作为 Query，影像 patch token 作为 Key/Value，影像 CLS 不参与该 cross-attention。临床信息沿残差路径保留，同时吸收查询到的影像信息：

```text
C1 = C + Dropout(CrossAttention(LN(C), LN(P), LN(P)))
C2 = C1 + FFN(LN(C1))
z  = LayerNorm(MaskedMean(C2))
```

其中 `C` 为临床 token，`P` 为影像 patch token。每个 block 后清零无效临床位置，最终仅对有效临床 token 池化。本轮为 1 层、4 heads、64 维嵌入、FFN 隐藏维度 256、dropout 0.1，启用 attention 和 FFN 残差。

设计动机：让每个临床变量选择相关影像区域，并通过残差保留临床表征。实现支持导出“临床变量 × 影像 patch”的 attention 权重用于诊断；attention 图本身不能证明临床因果关系或解释忠实性。本轮指标尚未支持该方案优于 Concat 或 Joint。

### 4. 训练与评测设置

| 项目 | v3 设置 |
| --- | --- |
| 随机种子 | 5212，单 seed |
| 优化器 | AdamW |
| 初始学习率 | 0.0003 |
| Weight decay | 0.1 |
| Batch size | 16 |
| 最大 epoch | 100 |
| 调度 | 5 epoch warmup + cosine |
| 梯度裁剪 | 1.0 |
| 损失函数 | BCE with logits；不额外增加阳性类别权重 |
| 样本权重 | 启用；本队列每次住院一张胸片 |
| 数据增强 / 模态 dropout | 均关闭 |
| Checkpoint 选择 | 最低验证 loss，允许从训练开始选择 |
| Early stopping patience | 15 |
| 分类阈值 | 各模型在验证集按 Youden J 确定，再用于测试集 |
| 评估单位 | 住院；按患者分组进行 bootstrap |
| 置信区间 | 2,000 次患者 bootstrap，95% CI，seed 5212 |

| 模型 | 实际训练 epoch 数 | 选中 epoch | 选中 checkpoint 验证 loss | 总参数量 |
| --- | ---: | ---: | ---: | ---: |
| Concat | 24 | 9 | 0.414799 | 85,863,681 |
| Joint attention | 21 | 6 | 0.414294 | 85,923,713 |
| Cross-attention Adapter | 26 | 11 | 0.417103 | 85,909,825 |

总参数量包含冻结的 ViT backbone，不等于可训练参数量。这里采用固定划分评估，不是 nested CV 或 out-of-fold 结果；“验证 AUROC 最高”也不代表 checkpoint 是按 AUROC 选取的。

### 5. v3 评测结果

#### 5.1 排序能力与概率误差

| 模型 | Val AUROC | Val AUPRC | Test AUROC（95% CI） | Test AUPRC（95% CI） | Test Brier ↓ |
| --- | ---: | ---: | --- | --- | ---: |
| Concat | 0.7063 | 0.3858 | **0.7952（0.6428–0.9226）** | **0.4881（0.2240–0.7619）** | 0.1439 |
| Joint attention | **0.7344** | 0.3423 | 0.7667（0.6000–0.9087） | 0.3570（0.1907–0.6777） | **0.1417** |
| Cross-attention Adapter | 0.6438 | **0.3859** | 0.6714（0.4483–0.8674） | 0.3256（0.1500–0.7023） | 0.1553 |

测试集阳性比例为 0.1892，可作为 AUPRC 的随机排序参考水平。Brier 越低越好，但不能仅凭此分数宣称概率已充分校准。

Concat 的测试排序指标最好，Joint 的验证 AUROC 最好且测试 Brier 最低。Adapter 的验证 AUPRC 仅略高于 Concat，差值约 0.00018，不应解读为实际优势；其测试 AUROC 区间包含 0.5。

#### 5.2 验证集所选阈值下的测试表现

| 模型 | 敏感度 | 特异度 | Precision | F1 | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Concat | 0.8571 | 0.5667 | 0.3158 | 0.4615 | 6 | 1 | 13 | 17 |
| Joint attention | 1.0000 | 0.4333 | 0.2917 | 0.4516 | 7 | 0 | 17 | 13 |
| Cross-attention Adapter | 0.1429 | 0.9333 | 0.3333 | 0.2000 | 1 | 6 | 2 | 28 |

Joint 在本测试集检出了全部 7 个阳性，但误报更多；这不意味着总体敏感度可稳定达到 100%。Adapter 在该阈值下漏掉 6 个阳性，较高特异度没有转化为好的召回。各模型阈值独立由验证集确定，因此该表体现的是当前阈值策略下的表现，而不是相同阈值或相同特异度下的比较。

### 6. 结论边界与后续验证

1. **当前结果支持保留 Concat 作为重要参照。** 更复杂的 attention 结构尚未呈现明确、稳定的性能优势。Concat 测试排名最高是本次观察，不能直接用测试集排名继续选模型和调参。
2. **融合增益尚未验证。** 需要在相同 ICU 转入队列、患者划分和输入定义下补充 clinical-only 与 CXR-only，并计算相对单模态的配对 ΔAUROC/ΔAUPRC。
3. **小样本与单次划分限制稳定性判断。** 验证集只有 8 个阳性、测试集只有 7 个阳性；后续可在开发集做患者分组交叉验证及多 seed 复核，并保留独立最终评估数据。
4. **历史数值不能直接横向比较。** 历史 244 次住院 / 186 位患者的实验采用患者分组嵌套交叉验证；本轮为新建的 280 次住院 / 218 位患者队列及固定划分。历史 AUROC 0.544/0.707 不应直接当作本轮单模态对照。
5. **Adapter 原因诊断仍待验证。** 当前结果不能确定是影像贡献不足、交互学习困难还是阈值不稳定。可在开发集进行图像置乱、模态遮蔽及临床单模态对照，结合 attention 分布检查；这些是建议实验，不是已经得到的结论。

适合汇报的表述：在当前固定划分的 ICU 转入预测任务上，Concat 获得最高测试 AUROC 0.795 和 AUPRC 0.488；Joint 获得最高验证 AUROC 0.734。小样本结果尚不能证明模型间存在稳定差异，也尚未验证相对单模态的融合增益。

### 7. 结果图与分析

以下嵌入本轮 v3 的 20 张结果图：14 张已有 PNG 原图，以及 6 张根据原预测重绘、修正任务标签的图。未重新训练或改变预测结果。训练曲线包含 train/validation 指标；ROC、PR、校准图和混淆矩阵来自同一测试集（37 次住院）；概率分布图来自验证集（48 次住院），attention 图来自验证集的一个 batch。精确指标以第 5 节表格为准。

#### 7.1 训练过程：后期过拟合与早停

三种模型均出现训练 loss 继续下降、验证 loss 在早期达到低点后升高的模式，提示后期过拟合。报告采用最低验证 loss 对应的 checkpoint，不能用训练结束时的权重或曲线上最高 AUROC 替代。

**Concat：选中第 9 epoch，共训练 24 epoch。**

![Concat：训练与验证 loss、验证 AUROC/AUPRC](../../../assets/bn5212_multimodal_v3/concat_training_training_curves.png)

图 1。第 9 epoch 后训练 loss 继续降低，但验证 loss 总体升高；后期验证 AUROC 也下降。继续降低训练误差没有带来更好的验证表现。

**Joint attention：选中第 6 epoch，共训练 21 epoch。**

![Joint attention：训练与验证 loss、验证 AUROC/AUPRC](../../../assets/bn5212_multimodal_v3/joint_training_training_curves.png)

图 2。验证 loss 在第 6 epoch 最低，后期与训练 loss 明显分离。图中验证 AUROC 的峰值不在所选 epoch，因为本轮 checkpoint 选择依据是验证 loss。

**Cross-attention Adapter：选中第 11 epoch，共训练 26 epoch。**

![Cross-attention Adapter：训练与验证 loss、验证 AUROC/AUPRC](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_training_training_curves.png)

图 3。后期训练 loss 接近 0，验证 loss 却显著升高，验证 AUROC 向随机水平靠近。这表明训练集拟合充分不能保证泛化，也支持保留早期 checkpoint。

#### 7.2 测试集 ROC 与 PR 曲线

每行对应一个模型，左侧展示不同阈值下的假阳性率与真阳性率，右侧展示 precision–recall 的权衡。PR 图中的虚线为测试集阳性比例 7/37 ≈ 0.189；图例 AP 对应本文报告的 AUPRC 计算口径。

| 模型 | 测试 ROC | 测试 PR |
| --- | --- | --- |
| Concat | ![Concat 测试集 ROC](../../../assets/bn5212_multimodal_v3/concat_benchmark_roc_curve.png) | ![Concat 测试集 PR](../../../assets/bn5212_multimodal_v3/concat_benchmark_precision_recall_curve.png) |
| Joint attention | ![Joint attention 测试集 ROC](../../../assets/bn5212_multimodal_v3/joint_benchmark_roc_curve.png) | ![Joint attention 测试集 PR](../../../assets/bn5212_multimodal_v3/joint_benchmark_precision_recall_curve.png) |
| Cross-attention Adapter | ![Cross-attention Adapter 测试集 ROC](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_benchmark_roc_curve.png) | ![Cross-attention Adapter 测试集 PR](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_benchmark_precision_recall_curve.png) |

图 4–9（按行从左至右）。Concat 的 AUROC/AUPRC 分别为 0.795/0.488，Joint 为 0.767/0.357，Adapter 为 0.671/0.326。Concat 的整体排序指标最好，但不能据此断言其在所有阈值下都更优；图中未显示置信区间，模型差异的不确定性仍应结合第 5、6 节理解。

#### 7.3 测试集校准曲线

横轴为分组后的平均预测概率，纵轴为对应组实际 ICU 转入比例；理想情况接近对角线。曲线高于对角线表示该组风险被低估，低于对角线表示被高估。

| Concat | Joint attention | Cross-attention Adapter |
| --- | --- | --- |
| ![Concat 测试集校准曲线](../../../assets/bn5212_multimodal_v3/concat_benchmark_calibration_curve.png) | ![Joint attention 测试集校准曲线](../../../assets/bn5212_multimodal_v3/joint_benchmark_calibration_curve.png) | ![Cross-attention Adapter 测试集校准曲线](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_benchmark_calibration_curve.png) |

图 10–12。三张图均使用测试集预测，按分位数分为 7 组，每组仅约 5–6 次住院，因此观测事件率容易波动。这些图用于检查概率偏差，不能凭折线形状认定校准可靠；对应 Brier 分别为 0.1439、0.1417、0.1553。

#### 7.4 Adapter 的临床变量—影像区域 attention

![Adapter：18 个临床变量与影像 patch 的 attention 矩阵](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_training_attention_matrix.png)

图 13。行为 18 个临床变量，列为影像 patch，颜色表示平均 attention 权重。导出程序只取验证集第一个 batch，并对样本和 attention heads 求平均，**不是整个验证集的汇总，也不是单个患者解释**。缺失变量的 attention 先被置零，再参与 batch 平均，因此某行颜色较浅可能反映观测缺失，不能直接解释为变量不重要。

![Adapter：前 12 个临床变量在影像 patch 网格上的 attention](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_training_attention_patch_maps.png)

图 14。将同一批次的平均 attention 重排到影像 patch 网格，仅展示前 12 个变量，所有子图共享色标。多个变量的空间模式看起来相近，可作为后续检验变量特异性是否充分的线索；尚不能单凭这张图判断 attention 退化。这不是叠加在原胸片上的病灶定位图，也不构成临床因果解释。

#### 7.5 测试集混淆矩阵（已修正任务标签）

原绘图函数沿用了死亡任务的 `Survived / Died` 标签。本节从 v3 保存的测试预测重新绘图，统一改为 `No ICU transfer / ICU transfer`（未转入 / 转入 ICU）；阈值仍为各模型在验证集上选定的原始阈值，使用完整精度计算，图标题仅做显示舍入。

| Concat | Joint attention | Cross-attention Adapter |
| --- | --- | --- |
| ![Concat：ICU 转入测试集混淆矩阵](../../../assets/bn5212_multimodal_v3/concat_confusion_matrix_test.png) | ![Joint attention：ICU 转入测试集混淆矩阵](../../../assets/bn5212_multimodal_v3/joint_confusion_matrix_test.png) | ![Cross-attention Adapter：ICU 转入测试集混淆矩阵](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_confusion_matrix_test.png) |

图 15–17。横轴为预测类别，纵轴为真实类别；左上 TN、右上 FP、左下 FN、右下 TP。重绘数值已逐项核对第 5.2 节的原始评测记录。Concat 检出 6/7 个阳性、误报 13 个；Joint 检出 7/7 个、误报 17 个；Adapter 检出 1/7 个、误报 2 个。该差异同时受模型排序能力和验证集选定阈值影响。

#### 7.6 验证集预测概率分布（已修正任务标签）

原图横轴误写为院内死亡概率；重绘后为 `P(future ICU transfer during this admission)`，图例区分未转入与转入 ICU。三个模型均使用原验证集的 48 次住院（40 个阴性、8 个阳性），保留原图 0–1 范围及 20 个等宽分箱。

| Concat | Joint attention | Cross-attention Adapter |
| --- | --- | --- |
| ![Concat：ICU 转入验证集概率分布](../../../assets/bn5212_multimodal_v3/concat_score_distribution_val.png) | ![Joint attention：ICU 转入验证集概率分布](../../../assets/bn5212_multimodal_v3/joint_score_distribution_val.png) | ![Cross-attention Adapter：ICU 转入验证集概率分布](../../../assets/bn5212_multimodal_v3/cross_attention_adapter_score_distribution_val.png) |

图 18–20。横轴为后续 ICU 转入概率，纵轴为住院次数。两组分布的重叠可用于观察区分难度；柱高是样本数，阴性组本来就更多，不能直接把阴性柱更高解释为模型偏好阴性。这里展示的是验证集，不能用其分布替代测试集结果，也不能仅凭直方图判定概率校准质量。

#### 7.7 图片来源与重绘核验

前 14 张图从 v3 已有文件原样复制到 assets 目录，后 6 张标签修正图保存于 [assets/bn5212_multimodal_v3](../../../assets/bn5212_multimodal_v3/)。重绘只调整任务文字与展示样式，未更改模型、数据划分、预测分数或分类阈值，原实验目录中的图片保持原样。

[重绘脚本](../../../assets/bn5212_multimodal_v3/redraw_task_labels.py)逐项检查混淆矩阵与原始 metrics 一致，并检查用于分布图的验证分数与训练导出文件一致；[核验记录](../../../assets/bn5212_multimodal_v3/redraw_verification.json)保存源文件哈希、阈值和汇总计数。

本文图片通过相对路径引用；移动或分享 Markdown 时，需同时保留 assets 图片目录；完整本地输出用于追溯预测、权重与评测记录。

### 8. 原始结果与实现索引

以下路径均相对于本 README 所在的训练项目目录；本总结由既有结果与配置整理，没有重新训练模型。

- [v3 完整结果目录](outputs/three_multimodal/transfer_multimodal_v3_20261030_161749/)
- [验证集汇总](outputs/three_multimodal/transfer_multimodal_v3_20261030_161749/validation_comparison.csv)
- [测试集排行榜](outputs/three_multimodal/transfer_multimodal_v3_20261030_161749/benchmark_leaderboard.csv)
- [本轮实际模型配置](outputs/three_multimodal/transfer_multimodal_v3_20261030_161749/configs/)
- [运行完成核验](outputs/training_launches/transfer_multimodal_v3_20261030_161749/completion_verification.json)
- [队列质量报告及划分数量](data/runs/icu_transfer_v1/qa_report.json)
- [ICU 转入任务说明](docs/ICU_TRANSFER.md)
- [历史队列差异核查](docs/ICU_TRANSFER_COHORT_AUDIT.md)
- [Concat 实现](src/fusion/concat.py)
- [Joint attention 实现](src/fusion/joint_self_attention.py)
- [Cross-attention Adapter 实现](src/fusion/cross_attention_adapter.py)

<a id="historical-icu-transfer"></a>

## 历史 ICU 转入单模态实验

> 来源：`BN5212_Final_Editable_v4.pptx`，包括正文、图片内文字及演讲备注。本节仅记录 PPT 中历史 ICU 转入单模态实验；院内死亡实验另列于后文，当前多模态结果见前文 v3。

### 1. 项目概述

本项目结合临床时间序列与胸部 X 光图像（CXR），预测患者在同次住院期间是否会在预测时点之后转入 ICU。核心问题是：**相比仅使用临床数据，加入胸片能否提高 ICU 转入预测能力？不同融合方式是否会影响增益？**

项目使用 MIMIC-IV 临床记录与 MIMIC-CXR 胸片，通过患者标识和时间窗口完成匹配。研究单位为一次住院，预测时点 t0 为入院后 48 小时内首张符合条件的胸片采集时间，患者在 t0 时尚未进入 ICU。输入限于 t0 及之前可获得的信息，标签为 t0 之后、同次住院内是否转入 ICU。

此处保留 PPT 中历史单模态实验背景。当前多模态实验已经完成，见 [v3 模型设计与结果](#multimodal-v3)。当前队列为 280 次住院，与本节历史 244 次住院实验不同。

### 2. 数据构建与预处理

#### 2.1 数据匹配和时间约束

MIMIC-IV 与 MIMIC-CXR 使用共同的 `subject_id`，同一患者的日期偏移也一致，因此可以根据胸片时间判断其所属住院记录。早期数据整合方案将匹配窗口起点设为急诊登记时间，以纳入住院前在急诊完成的胸片。

所有输入都必须在对应任务规定的观察窗口内可获得。历史记录仅来自此前就诊，训练与测试按患者分组，避免同一患者跨集合造成信息泄漏。

#### 2.2 实验队列

ICU 转入实验纳入 **186 名患者、244 次住院，其中 42 次后续转入 ICU，阳性比例为 17.2%**。患者可能有多次住院，因此必须按患者而不是住院记录划分数据，避免同一患者同时进入训练和测试。

#### 2.3 输入和缺失值处理

ICU 转入实验使用胸片前 48 小时的 16 项血液检查，以及年龄、性别。临床变量覆盖血常规、电解质、肾功能、血糖、乳酸、pH、INR、胆红素和白蛋白等。胸片取首张正位图像（AP/PA），输入尺寸为 224 × 224。

缺失值是主要难点：转入实验中乳酸缺失约 63%，pH 缺失约 90%。PPT 的任务示意图还提出使用 observation mask，区分未测量与真实数值。

单模态训练页明确采用按变量 z-score 标准化。预处理统计量应只从每折训练集拟合，再应用于验证和测试集。对于缺失值填补与 mask 的具体实现，需以 ICU 转入实验的最终代码为准。

PPT 的早期流程还展示过 log1p、分位数截尾、中位数填补等方法，可作为预处理参考，但不能直接当作当前实验已采用的全部步骤。任何预测时点之后的信息，例如出院去向、最终诊断和住院总时长，都应从输入中排除。

### 3. 模型与已有结果

#### 3.1 单模态基线

- **Clinical-only：**将各变量的 48 小时序列映射为 token，再通过平均池化和分类头预测。
- **CXR-only：**采用 ImageNet 预训练、冻结的 ViT-B/16，仅训练小型投影层和分类头，以适应有限样本。

训练采用 AdamW、二元交叉熵和验证集早停。评估强调患者分组的嵌套交叉验证，外层测试折不参与模型选择或训练轮数选择。

| ICU 转入预测指标 | Clinical-only | CXR-only |
| --- | --- | --- |
| AUROC（95% CI） | 0.544（0.439–0.642） | 0.707（0.619–0.791） |
| AUPRC | 0.226 | 0.335 |
| 敏感度 / 特异度 | 48% / 66% | 48% / 81% |

该队列的阳性比例为 0.172，可作为 AUPRC 的随机参考水平。结果提示胸片模型具有更好的区分能力；在 PPT 展示的阈值下，两者都检出 42 个阳性中的 20 个，而胸片模型的假阳性更少（39 对 68）。**这些结果尚不能回答融合是否带来增益。**


当前多模态的模型、训练设置、结果和后续验证建议统一见 [v3 报告](#multimodal-v3)。

<a id="mortality-baselines"></a>

## 院内死亡单模态基线

### 项目目标

用 MIMIC-IV 临床时序和 MIMIC-CXR 胸片预测 ICU 患者的院内死亡。本目录负责所有实验共用的
训练框架，以及两个单模态基线：**clinical-only** 和 **CXR-only**。

项目附带两个已抽取数据的 `.7z` 包及部分 embedding，`results/` 保存汇总指标和图。
数据包实际内容与原训练接口的对应关系见 [数据清单](docs/PROJECT_DATA_INVENTORY.md)。环境与运行细节见
[docs/USAGE.md](docs/USAGE.md)，cohort 与评估规则见 [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md)。

### 输入数据

| | |
|---|---|
| 数据 | MIMIC-IV 3.1、MIMIC-CXR 2.1.0（课程子集：5,534 张 DICOM、1,000 位患者） |
| 研究单位 | ICU 住院：198 次、159 位患者 |
| 观察窗口 | ICU 入住后的前 48 小时 |
| 预测时点 | ICU 入住 + 48 小时，此时患者存活且仍在院 |
| 标签 | 院内死亡：31 例（15.7%） |
| 划分（按患者） | train 140 / val 32 / test 26 次住院，死亡 23 / 4 / 4 例 |

### 方法

#### Clinical-only

```
17 个变量 × 48 个小时格 + 观测掩码 → 每个变量各一个线性层 → 17 个 token（64 维）
→ 带掩码的平均 → LayerNorm → 线性层 → P(死亡)
```

- 变量是 MIMIC benchmark 的 17 项（生命体征、GCS、血糖、pH、FiO2、身高、体重）。
  按变量做 z-score；缺失的小时填 0，掩码为 0。
- 一条测量只有在预测时点之前既已记录（charttime）又已入库（storetime）才会使用。
- **预训练。** 编码器先在 cohort 以外的 12,121 次 ICU 住院（1,100 例死亡）上训练，
  用另外 2,188 次外部住院选模（AUROC 0.848），之后冻结。cohort 的 159 位患者按
  `subject_id` 全部排除，入选规则与 cohort 相同。

#### CXR-only

```
ICU 住院的第一张 AP 胸片，224 × 224 → ViT-B/16（ImageNet 预训练，冻结）→ 线性层 768 → 64
→ CLS token → LayerNorm → 线性层 → P(死亡)
```

- 增强：随机裁剪（比例 0.85–1.0）、旋转 ±10°、亮度和对比度 ±0.15。不做水平翻转。

#### 训练与评估

- AdamW，学习率 1e-3，weight decay 0.1，5 个 epoch warmup 后 cosine 衰减，二元交叉熵，
  batch 16，early stopping（patience 15），seed 5212。
- **5 折患者分组 nested 交叉验证**，跑在 train + val 上（172 次住院、27 例死亡）。
  每个外层折里用 4 个内层折决定训练多少 epoch，再用全部拟合患者重训，留出折只评分一次。
- 指标：汇总 out-of-fold 预测后的 AUROC，95% 置信区间来自按患者 bootstrap 2,000 次。
- test split 只评一次，由 `benchmark-evaluation` 完成，用的是在 train + val 上重训的模型。

### 验证与结果

| 模型 | 交叉验证 AUROC（95% CI） | 交叉验证 AUPRC | Test AUROC（95% CI） |
|---|---|---:|---|
| Clinical-only，预训练编码器 | **0.725**（0.629–0.815） | 0.312 | 0.659（0.457–0.846） |
| Clinical-only，只用 cohort | 0.606（0.460–0.755） | 0.260 | 0.591（0.230–0.923） |
| CXR-only | 0.552（0.432–0.679） | 0.190 | 0.670（0.124–1.000） |

随机水平：AUROC 0.5，AUPRC 0.157。预训练出的临床模型不在 cohort 上拟合任何参数、
直接套用时是 0.748（0.655–0.833）。

![Out-of-fold AUROC 及 95% 区间](results/figures/unimodal_auroc.png)

![Out-of-fold ROC 曲线](results/figures/unimodal_roc.png)

- 编码器见过足够多的患者之后，临床数据有信号：区间下界 0.629。
- CXR-only 和随机分不开。
- 27 例死亡下，任意两个模型的差异都不显著。预训练对只用 cohort：+0.119（−0.051 到 +0.278）。
- test split 只有 4 例死亡，区间太宽，不能用来给模型排序。

#### 消融

| 选择 | 变体 | AUROC |
|---|---|---:|
| 影像编码器（交叉验证） | ViT-B/16 冻结（采用） | 0.552 |
| | ViT-Tiny，解冻最后 2 个 block | 0.495 |
| | DenseNet121 CheXpert，冻结 | 0.478 |
| | ViT-Tiny 冻结 | 0.462 |
| 临床编码器，只用 cohort（交叉验证） | 所有变量共用一个投影 | 0.606 |
| | 每个变量各一个投影 | 0.597 |
| | 汇总统计量 | 0.512 |
| 临床编码器，预训练（外部验证） | 每个变量各一个投影（采用） | 0.848 |
| | 所有变量共用一个投影 | 0.791 |

影像编码器做不了同样的预训练：课程子集里 cohort 以外的患者只有 7 例院内死亡。改用
「拍片后 180 天内死亡」作代理标签（844 张胸片），外部 AUROC 仍在 0.50–0.58 之间
（`scripts/probe_external_image_signal.py`）。

当前多模态结果统一见 [v3 结果索引](results/multimodal_v3/README.md)。

### 运行方式

多模态运行见 [v3 运行说明](docs/RUN_THREE_MULTIMODAL.md)。以下命令用于原有单模态实验：

```bash
# 1. cohort 的临床特征和影像缓存
bn5212-extract-clinical --run-dir <run> --mimic-root <mimic> --cohort-unit icu_stay \
    --max-hours 48 --output data/clinical_features.csv
bn5212-build-image-cache --run-dir <run> --output data/cache/icu_images.json

# 2. 在 cohort 以外的 ICU 住院上预训练临床编码器
python -m bn5212_training.pretrain cohort --run-dir <run> --mimic-root <mimic> --output data/pretrain/clinical_v1
bn5212-extract-clinical --run-dir data/pretrain/clinical_v1 --mimic-root <mimic> --cohort-unit icu_stay \
    --max-hours 48 --output data/pretrain/clinical_v1/clinical_features.csv
python -m bn5212_training.pretrain fit --cohort data/pretrain/clinical_v1 --encoder variable_projection \
    --config configs/icu/clinical_only.json --output data/pretrain/clinical_v1/variable_projection.pt

# 3. nested 交叉验证
bn5212-crossval --config configs/icu_pretrained/clinical_only.json --output-dir outputs/nested_cv --run-id v3
bn5212-crossval --config configs/icu/clinical_only.json --output-dir outputs/nested_cv --run-id v2
bn5212-crossval --config configs/icu/cxr_only.json --output-dir outputs/nested_cv --run-id v2

# 4. 导出表和图
python scripts/export_results.py
```

| 路径 | 内容 |
|---|---|
| `src/` | 训练循环、编码器、fusion 模块、交叉验证、预训练 |
| `configs/icu/`、`configs/icu_pretrained/` | 不带 / 带预训练临床编码器的实验配置 |
| `results/` | 本页用到的表和图 |
| `docs/` | [USAGE](docs/USAGE.md)、[DATA_STRATEGY](docs/DATA_STRATEGY.md)、[FUSION_API](docs/FUSION_API.md)、[CLINICAL_FEATURE_SPEC](docs/CLINICAL_FEATURE_SPEC.md) |

### 后续工作

v3 融合策略 `cross_attention_adapter`：临床 Q、胸片 patch K/V、残差与 FFN，
接入配置、冻结编码器及 ICU 转入任务模板见 [使用说明](docs/CROSS_ATTENTION_ADAPTER.md)。

- 交叉验证只有 27 例死亡，test 只有 4 例，且只跑了一个 seed。
- 影像 backbone 冻结在 224 px；MeTra 是在 6,125 位患者上以 384 px 全模型微调。
- 课程压缩包里的 `chartevents` 在约 72% 的患者处被截断。study cohort 完全覆盖，
  预训练 cohort 因此只有 14,309 次住院。
