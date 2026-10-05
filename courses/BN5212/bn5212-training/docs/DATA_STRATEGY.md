# BN5212 数据选择策略 v1.0

给组员：照这份文件做，产出的 cohort 会和已报告的结果完全一致。任何一步不同，
结果就不能放进同一张比较表。

一句话：**研究单位是一次 ICU 住院；观察窗口是入住后 48 小时；预测时点是窗口结束的那一刻；
标签是该次住院最终是否院内死亡。**

## 0. 先核对：你复现出来的应该是这些数字

```
ICU 住院（样本）   198        病人 159        住院 184
死亡事件            31        prevalence 15.7%
view               AP 198 (100%)
index.csv sha256   5aef965fd6ed83e017b18be769c030744387bcff47cc0caa2479c1387079e943
split_seed         5212       split_unit  subject_id
```

`index.csv` 的 SHA-256 对不上，就是某一步不同，**不要继续往下做**。
`dataset_spec.json` 里有这个值，`bn5212-train` 每次也会重新校验。

## 1. 数据版本

MIMIC-IV **3.1**、MIMIC-CXR **2.1.0**（课程提供的 DICOM 子集：5,534 张、1,000 位患者）。
流水线会拒绝其他版本组合。

## 2. Cohort：为什么是 ICU 级

流水线支持两种研究单位，我们用 `icu_stay`。**不要改回 `admission`**，原因不是偏好：

| | admission（住院级） | icu_stay（本方案） |
|---|---:|---:|
| 样本 | 413 | 198 |
| 死亡事件 | 29 | **31** |
| 临床数据覆盖 | 30.8% | **100%** |
| view | AP 314 / PA 99 | AP 198 |

住院级有两个会让结论失效的混杂：

1. **99 张 PA 片里零死亡**。单靠「是不是 AP 片」就有 AUROC 0.629 —— 模型可以靠
   「站着拍还是躺着拍」作弊，那是照护场所的代理变量，不是肺部所见。
2. **69% 的住院没有 chartevents**（它属于 icu 模块）。单靠「有没有临床数据」有
   AUROC 0.645。两种模态在住院级上都落到 0.807，因为都在检测「是不是 ICU 患者」。

ICU 级 cohort 里每个人都在 ICU、每张片都是 AP，两个混杂同时消失。

## 3. 配置

用 `bn5212-data-pipeline/config/server.local.json`，关键字段：

```json
"cohort":    {"unit": "icu_stay", "selection": "first_per_icu_stay",
              "min_age": 18, "views": ["AP", "PA"]},
"alignment": {"icu_observation_hours": 48,
              "max_hours_after_admission": null,
              "minimum_hours_before_end": 0},
"label":     {"kind": "in_hospital_mortality", "column": "hospital_expire_flag"},
"split":     {"seed": 5212, "train": 0.7, "val": 0.15, "test": 0.15, "stratify": true},
"loader":    {"image_size": 224, "channels": 3,
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
```

`max_hours_after_admission` 必须是 `null`。ICU 入住可能在住院第 5 天，留着 48 小时
的入院窗口会把 ICU 窗口再切一次。

## 4. 筛选序列（流水线自动执行，这里是它掉样本的地方）

```
5,534 张 DICOM
  → 5,391  study 元数据一致              (-143)
  → 3,509  只留 AP/PA                    (-1,882)
  → 1,640  落在住院区间内                (-1,869)   ← 过半的片子不在住院期间
  → 1,638  死亡标志与 deathtime 一致     (-2)
  → 1,383  能连到 ICU stay               (-255)
  →   438  落在 ICU 前 48 小时窗口内     (-945)     ← ICU 筛选的主要损失
  →   436  一张片只属于一次 ICU 住院     (-2)
  →   410  预测时点早于结局              (-26)      ← immortal time
  →   198  每次 ICU 住院取第一张         (-212)
```

最后一条不变式：**`intime + 48h` 必须严格早于 `min(deathtime, dischtime)`**。
它同时保证患者活过整个窗口、且住院尚未结束——记录提前中断本身会通过「没有数据」
泄漏存活。

`cohort_flow.csv` 里有完整计数，复现时逐行对一遍。

## 5. 临床特征

```bash
bn5212-extract-clinical --run-dir <run> --mimic-root <mimic> \
  --output clinical_features.csv --cohort-unit icu_stay --max-hours 48
```

**跑之前先核对 itemid**：`--inspect-items`。不同 MIMIC 版本的 itemid 会变（v3.1 的
发布说明就记载 labevents 的 itemid 在 v2.2→v3.0 之间非预期改变）。当前配置的 33 个
itemid 已对 v3.1 全部核对通过。

按 `stay_id` 而不是 `hadm_id` 归类：一次住院可能有多次 ICU 住院，各自的 hour 0 不同。

### 两条防泄漏规则

- **两个时钟都卡**：`charttime` 和 `storetime` 都必须早于预测时点。一条第 3 小时测量、
  第 50 小时才入库的记录，在预测当下拿不到。实测拦下 993 条。
- **标准化只在拟合数据上做**。交叉验证时是**每个 fold 各自拟合**，不是在整个
  train split 上拟合一次——否则每个 fold 都会泄漏它即将评分的患者的统计量。

### 已知缺口

身高覆盖率 **0.5%**（198 次住院只有 1 条），实质不可用，与 MeTra 剔除 2 个全缺变量
的情况一致。MIMIC-IV 的身高主要在 `omr` 表，当前抽取未接入。其余 16 个变量覆盖
67.7%–100%。

## 6. 划分与评估

**患者级隔离。** 同一 `subject_id` 绝不跨 split，也绝不跨 fold。流水线和
`make_folds` 都强制这一条。

**主结果用 5 折患者分组交叉验证，跑在 train+val 上；冻结的 test split 不参与。**
理由是数字：单一划分的 val 和 test 各只有 4 个事件，AUROC 只能以 1/(4×28) 为单位
跳动。交叉验证让 27 个事件全部被评分一次，置信区间从约 ±0.35 收到 ±0.15。

**被评分的那一折不参与任何选择（v2 协议，nested）。** 每个外层折里，拟合患者再切成
4 个内层折，各自 early stopping；取它们选中 epoch 的中位数作为预算，用全部拟合患者按
这个预算重训一次，再给外层留出折评分。交付模型用同一套做法：全部 train+val 患者、
内层选出的 epoch 中位数。

v1 直接在被评分的那一折上 early stopping，等于每折挑了最适合那批患者的 epoch。
实测 clinical-only 因此报出 0.720，而事先固定的 epoch 预算（8–60 个 epoch）只有 0.64–0.67。
**v1 的数字（clinical 0.720、CXR 0.600，以及当时的消融表）全部作废，不要再引用。**

当前结果（172 个 out-of-fold 样本，27 个事件；完整表在 [`results/`](../results/)，说明见 [README](../README.md)）。
v2 只用 cohort 训练；v3 的临床编码器先在 cohort 以外的 ICU 住院上预训练（第 10 节），
其余完全相同：

| 模型 | v3 OOF AUROC | 95% CI | v2（只用 cohort） |
|---|---:|---|---:|
| Clinical-only | 0.725 | [0.629, 0.815] | 0.606 |
| CXR-only | 0.552 | [0.432, 0.679] | 0.552 |
| Concat fusion | 0.686 | [0.591, 0.780] | 0.605 |
| MeTra joint self-attention | 0.531 | [0.415, 0.653] | 0.567 |
| Cross-attention | 0.671 | [0.564, 0.764] | 0.623 |

v3 里唯一排除 0 的配对差异是 MeTra joint 低于 clinical-only（−0.195，[−0.339, −0.035]）。
加入胸片没有提升；cross-attention 比 MeTra 高 0.140，但区间 [−0.026, +0.286] 包含 0。

**置信区间按患者 bootstrap**，不是按行——同一患者的多张片不独立。

**test split 只作协议合规检查。** 26 个样本、4 个事件，区间几乎覆盖全域，而且
它给出的模型排序与交叉验证相反。报告里要写出这一点，不要拿它下结论。

交给 `bn5212-evaluate` 的两份文件由 `bn5212-crossval` 直接写在运行目录下：
`predictions_val.csv` 是冻结 val split 的 **out-of-fold** 预测，`predictions_test.csv`
来自交付模型。不要用交付模型自己对 val 的预测——它拟合过这些患者，v1 因此在评测报告里
出现过 val AUROC 0.99，阈值也是在样本内选的。

## 7. 你该跑的命令

```powershell
# 1. 建 cohort
cd ..\bn5212-data-pipeline
.\.venv\Scripts\python.exe run_pipeline.py --config config\server.local.json --test-loader

# 2. 对数字（deaths 应为 31）
cd ..\bn5212-training
.\.venv\Scripts\python.exe scripts\summarize_run.py <run_dir>

# 3. 临床特征（先 --inspect-items）
.\.venv\Scripts\bn5212-extract-clinical.exe --run-dir <run_dir> --mimic-root <mimic> `
  --output data\clinical_features.csv --cohort-unit icu_stay --max-hours 48

# 4. 影像缓存（DICOM 解码约 400 ms/张，不缓存的话多折训练会被 I/O 拖死）
.\.venv\Scripts\bn5212-build-image-cache.exe --run-dir <run_dir> --output data\cache\icu_images.json

# 5. 训练
.\.venv\Scripts\bn5212-crossval.exe --config configs\icu\<实验>.json --folds 5 --device cuda
```

`configs/icu/` 里五个实验的 encoder、prediction head、optimizer、schedule、seed
完全相同，**只差 fusion 名字**。做多模态时请沿用这些配置，不要另起一套超参数，
否则比较的就不只是 fusion。

## 8. 不要动的东西

- **不要重建 split。** 换 seed 或换比例，所有已有结果就不可比。
- **不要用 admission 级 cohort 做多模态**（第 2 节的两个混杂）。
- **不要在交叉验证之外调超参数然后报告 CV 分数。** 27 个事件下试 6 种设置挑最好的，
  期望值会虚高 0.05–0.10，刚好是想检测的效应量。要调就用嵌套交叉验证。
- **不要用含 MIMIC 的预训练权重**（torchxrayvision 的 `-all` 和 `-mimic`）。
  它们训练时见过我们的留出图像。`xrv_densenet` 编码器会直接拒绝这类权重。

## 9. 已知偏离 MeTra 之处（报告要写）

| | MeTra | 本项目 |
|---|---|---|
| 队列规模 | 6,125 患者 | 159 患者 / 198 次 ICU 住院 |
| 影像 backbone | ViT-B/16 全模型微调 | ViT-B/16 **冻结** + 投影层 |
| 分辨率 | 384×384 | 224×224 |
| 评估 | 单一划分 | 5 折交叉验证 |
| 数据处理 | 官方 MeTra 代码 | 本组自建流水线 |

冻结不是偷懒：解冻 ViT-B 一个 block 就是 713 万可训练参数对 138 个训练样本。
实测（v2 协议）ViT-Tiny 冻结 0.462、解冻最后两层 0.495，都不如冻结的 ViT-B/16（0.552），
而且三者的配对差异区间都包含 0。

样本规模差 32 倍是影像分支低于 MeTra 的主因。若真实 AUROC 为 0.80，27 个事件下的
标准误约 0.053，观测到 0.55 的概率远低于 1%，所以这个差距是真的，不是噪声。

## 10. 临床编码器预训练（v3）

138 个训练样本学不出 17 个变量 × 48 小时的生理规律，换架构也没用（第 6 节的 v2 列）。
MIMIC-IV 里还有几万次 ICU 住院没有落在课程胸片子集里，所以从不进入 cohort。临床分支
先在这些住院上学，和影像分支先在 ImageNet 上学是同一个道理。

```powershell
.\.venv\Scripts\python.exe -m bn5212_training.pretrain cohort --run-dir <run_dir> `
  --mimic-root <mimic> --output data\pretrain\clinical_v1
.\.venv\Scripts\bn5212-extract-clinical.exe --run-dir data\pretrain\clinical_v1 --mimic-root <mimic> `
  --cohort-unit icu_stay --max-hours 48 --output data\pretrain\clinical_v1\clinical_features.csv
.\.venv\Scripts\python.exe -m bn5212_training.pretrain fit --cohort data\pretrain\clinical_v1 `
  --config configs\icu\clinical_only.json --encoder variable_projection `
  --output data\pretrain\clinical_v1\variable_projection.pt
.\.venv\Scripts\bn5212-crossval.exe --config configs\icu_pretrained\<实验>.json --folds 5 --device cuda
```

你复现出来的应该是：14,309 次 ICU 住院、10,007 位患者、1,313 个死亡事件；外部验证
AUROC 0.848（variable_projection）。

### 三条规则

- **study cohort 的患者一个都不能进来。** 按 `subject_id` 排除全部 159 位，train、val、
  test 都算。`cohort.json` 记录了排除人数和 study `index.csv` 的 SHA-256。
- **入选规则与 cohort 相同**：成人、死亡记录一致、ICU 入住 + 48 小时严格早于死亡或出院。
  特征用同一份抽取代码和同一份 itemid 配置。
- **关于预训练编码器的选择只看外部验证集。** 选哪种编码器、训练多少 epoch，都由外部的
  2,188 次住院决定。不要拿 cohort 的交叉验证分数回头挑预训练设置——那 27 个事件一旦
  用来调参，报出来的分数就不再是估计。

接入 study 实验时编码器冻结（`clinical_encoder.freeze: true`），标准化统计量随 checkpoint
一起带过来，不在 fold 上重新拟合。`configs/icu_pretrained/` 和 `configs/icu/` 只差这几个字段。

### 本机的 chartevents.csv 是截断的

文件在一行中间结束，按 `subject_id` 范围推算只有完整文件的约 72%。
study cohort 不受影响（患者 ID 都远在截断点之前，生命体征覆盖率 100%）。截断点之后
的患者只剩检验值、没有生命体征，`pretrain cohort` 会自动检测并剔除他们（20,001 →
14,309 次住院）。要用更多患者，先重新解压完整的 chartevents。

### 影像分支为什么没有做同样的事

试过，数据不支持。课程子集里非 cohort 患者的住院 AP 胸片有 299 张，院内死亡只有 7 位患者
（有标签且死亡的 41 位里，34 位已在 cohort 内）。改用代理标签「拍片后 180 天内死亡」
（844 张、59 位事件患者）后，冻结特征的外部交叉验证 AUROC 是 0.502–0.558，微调 ViT 最后
两个 block 不超过 0.576，都和随机分不开。没有信号的编码器不接入实验。
复现：`scripts/probe_external_image_signal.py`。

### 报告里要写

临床编码器在 12,121 次 cohort 以外的 MIMIC-IV ICU 住院上预训练，已按患者排除全部 cohort
成员。这是相对 MeTra 的又一处偏离（MeTra 的两个分支都只在配对数据上训练）。
