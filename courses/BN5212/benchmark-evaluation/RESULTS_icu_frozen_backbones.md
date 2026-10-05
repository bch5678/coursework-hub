# 首次 ICU 真实数据运行：冻结的 ViT-B/16 vs 冻结的 ResNet-18

**注意：这份记录来自 `add-training` 加入交叉验证与 `project_to` 之前的配置**
（fusion 宽 768、无图像缓存、无增强、lr 5e-6）。它的价值是链路验证与一个负面结论，
**不是**当前推荐的对比方式。现在正确的做法是用 `configs/icu/metra_joint.json` 与
`configs/icu/metra_resnet.json` 跑 `bn5212-crossval`，两者除 `image_encoder` 外逐字段相同。

## 协议

| 项目 | 值 |
|---|---|
| 队列 | ICU 住院级，198 次 ICU 住院 / 159 患者 / 184 住院 / 31 死亡 |
| 观测窗口 | ICU `intime` 起 48 小时，预测时点 = `intime + 48h` |
| 不变式 | `intime + 48h` 严格早于 `min(deathtime, dischtime)`（防 immortal time） |
| 影像 | MIMIC-CXR 子集，224×224，3 通道，ImageNet 归一化，view 全为 AP |
| 临床 | 17 变量 × 48 小时（76 通道展开），按 `stay_id` 归类，`charttime` 与 `storetime` 双时钟截断 |
| 划分 | 患者级，train 140 / val 32 / test 26，seed 5212 |
| 影像 backbone | **冻结**（`freeze: true`，BN 钉在 eval） |
| 融合 | `joint_self_attention`（MeTra），768 宽 / depth 2 / heads 12 |
| 优化 | AdamW 5e-6 → 1e-7 cosine，warmup 5，200 epochs，patience 20，BCE，无类别加权 |
| 可训练参数 | ViT 臂 14,525,185 ／ ResNet 臂 14,808,577（相差 2%） |

两臂**除 `image_encoder` 外逐字段相同**，配置见
`bn5212-training/configs/icu/icu_vit_frozen_joint.json` 与
`.../icu_resnet18_frozen_joint.json`。冻结时 BatchNorm 被钉在 eval
（`TimmResNet.train`），否则 ResNet 臂会静默把归一化统计量重拟合到训练集胸片，
两臂的 "frozen" 就不是同义词。

## 测试集结果（26 样本 / 4 事件）

由 `bn5212-evaluate` 产生，阈值在验证集上用 Youden's J 选定后原样应用于测试集。

| 模型 | AUROC | AUPRC | Sensitivity | Specificity | F1 | Brier |
|---|---|---|---|---|---|---|
| icu-vit-b16-frozen | 0.7045 | 0.4006 | 0.250 | 0.909 | 0.286 | 0.1391 |
| icu-resnet18-frozen | 0.6250 | 0.2491 | 0.750 | 0.273 | 0.261 | 0.1382 |

患者级 bootstrap 95% CI：

| 模型 | AUROC CI | AUPRC CI |
|---|---|---|
| ViT-B/16 | [0.208, 0.967] | [0.108, 0.889] |
| ResNet-18 | [0.300, 0.893] | [0.089, 0.667] |

## 配对 bootstrap：没有可检测的差异

同一次患者级重采样下比较两臂（4000 次，seed 5212）：

```
ΔAUROC (ViT − ResNet) = +0.080    95% CI [−0.442, +0.599]    P(Δ>0) = 0.612
ΔAUPRC                = +0.151    95% CI [−0.169, +0.634]
```

**置信区间从 −0.44 跨到 +0.60，P(Δ>0) = 0.612 与抛硬币无异。**

在 26 样本 / 4 事件的测试集上，"冻结的 ImageNet ViT-B/16 特征"与"冻结的 ImageNet
ResNet-18 特征"对本任务**没有可检测的差别**。这不是"ResNet 更差"，而是这个测试集
没有能力分辨两者。

## 不能从这个结果得出的结论

- ❌ **不能**说 ViT 比 ResNet 好（或反之）。差值 0.5σ，在噪声内。
- ❌ **不能**用训练脚本输出的 `val_auroc` 作为结果。ViT 在验证集上是 0.839，到测试集
  掉到 0.705（−0.134），这个落差就是在 32 样本 / 4 事件上选 checkpoint 的代价。
- ❌ **不能**读依赖阈值的指标（sensitivity / specificity / F1）。四个事件上敏感度只能
  以 0.25 为步长跳动，两臂只是选中了不同的阈值（0.371 vs 0.180），才呈现出
  0.25/0.909 与 0.750/0.273 这种镜像式的分布。
- ❌ **不能**与其他组的数字并列比较。本项目的 index 由本组自建流水线生成，不是官方
  benchmark split。

## 为什么这恰好印证了数据策略

数据策略文档主张主结果用 5 折患者分组 CV 跑在 train+val 上，理由是"单一划分的 val 和
test 各只有 4 个事件，AUROC 只能以 1/(4×28) 为单位跳动，实测 per-fold 从 0.383 到
0.930"。本次运行给出的 CI 宽度（−0.44 到 +0.60）与该判断一致。

**在补上 5 折 CV 之前，两臂的胜负不可判定。** 本目录的 `bn5212-compare` 能给出排名，
但排名在这里没有统计意义 —— 它只是 26 个样本上的一个点估计。

## 这轮运行证明了什么

1. **整条链路能跑通** —— 287 GB MIMIC → ICU 队列 → 临床抽取 → 训练 → 评测框架，
   全程无人工干预，两臂各约 20 分钟。
2. **单一划分没有分辨能力** —— 实测 CI 宽达 [-0.44, +0.60]，与数据策略主张交叉验证的
   判断一致。现在 `bn5212-crossval` 已经就位。
3. **暴露了两个真实的性能问题，都已被后续工作解决**：
   - **DICOM 反复解码**：0.19 s/张，GPU 只有约 15% 时间在算 → 后来由 `image_cache` 解决
   - **融合过宽**：768 宽的 joint self-attention 单独就有 14.3M 参数，在 138 个训练样本上
     无法训练 → 后来由 `project_to` 把 token 投到 64 宽解决

## 用现行配置重跑

```bash
bn5212-build-image-cache --run-dir <run> --output data/cache/icu_images.json
bn5212-crossval --config configs/icu/metra_joint.json   --folds 5 --device cuda
bn5212-crossval --config configs/icu/metra_resnet.json  --folds 5 --device cuda
python scripts/compare_crossval.py <joint-dir> <resnet-dir>
```

`metra_resnet` 与 `metra_joint` 除 `image_encoder` 外逐字段相同，两个 backbone 都投到
同样的 64 宽融合，因此差异只剩冻结特征本身。

## 复现所需（当时的配置）

```bash
# 1. ICU 队列（约 40 分钟，含 DICOM 头扫描与逐张解码校验）
bn5212-data-pipeline: python run_pipeline.py --config config/icu_server.local.json --test-loader

# 2. 临床特征（约 13 分钟，流式扫 chartevents + labevents）
bn5212-extract-clinical --inspect-items --run-dir <run> --mimic-root <mimic>   # 先核对 itemid
bn5212-extract-clinical --run-dir <run> --mimic-root <mimic> \
  --output clinical_features.csv --cohort-unit icu_stay --max-hours 48

# 3. 两臂训练（各约 20 分钟，单张 A5000 足够）
bn5212-train --config configs/icu/icu_vit_frozen_joint.json --device cuda
bn5212-train --config configs/icu/icu_resnet18_frozen_joint.json --device cuda

# 4. 评测
bn5212-evaluate --run-dir <run> --val-predictions <vit>/predictions_val.csv \
  --test-predictions <vit>/predictions_test.csv --model-name icu-vit-b16-frozen \
  --model-version v1 --checkpoint <vit>/checkpoint_best.pt --output-dir outputs/icu_vit_v1
bn5212-compare outputs/icu_vit_v1 outputs/icu_resnet18_v1 \
  --output-csv outputs/leaderboard.csv --output-html outputs/leaderboard.html
```

数据本身不在仓库里。`index.csv`、`clinical_features.csv`、`predictions.csv` 与
checkpoint 都含患者级信息或由受限数据派生，按仓库规则不提交。

## 已知不一致

本次运行的 `index.csv` SHA-256 为
`5dc3442b89d43cd68dee3a6e450d0fb6c408833a353692056d54e0ca64ce3e04`，
而数据策略文档记录的是
`5aef965fd6ed83e017b18be769c030744387bcff47cc0caa2479c1387079e943`。

**队列本身逐项一致**：11 个筛选 stage 的进出数与排除数全部吻合
（5534→5391→3509→1640→1638→1383→438→436→410→198），
最终 198/159/184/31、prevalence 15.66%、view AP 100%、
train+val 27 事件、test 26 样本 4 事件全部相同。
哈希差异无法用序列化选项解释（枚举 10 种变体均不命中），推测来自比已提交代码更新的
流水线版本。**在任何对外报告里，本结果的来源哈希应记为上列 `5dc3442b…`。**
