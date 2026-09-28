# 临床时序特征接口规格 v1.0

本文描述训练框架消费临床特征的格式。`bn5212-data-pipeline` 只读取 `admissions` 和
`patients`，不抽取临床时序数据，所以这一份是新增的产物。

**本规格已经有实现：`bn5212-extract-clinical`**（见第 7 节）。它读取冻结运行目录的
`index.csv` 拿到 cohort 和截止点，流式扫描 `chartevents` / `labevents`，
输出下面定义的长格式表和一份 QA 报告。它**不修改流水线的任何代码**，
也不碰 `index.csv` 或 Dataset —— 只产出一个 side-car 文件，
因此和数据负责人的模块互不冲突。

分工上，按 CONTRIBUTING，新增 MIMIC 表的清洗本属数据侧工作。
如果数据负责人接手，把 `src/bn5212_training/extract.py` 和
`configs/clinical_items.json` 整体搬进流水线项目即可，接口不变。

## 1. 产出什么

一个长格式表，每行一条观测：

```csv
hadm_id,hour,variable,value
20000001,0,Heart Rate,88
20000001,0,Oxygen saturation,97
20000001,1,Heart Rate,92
20000001,3,Glucose,141
20000002,0,Heart Rate,76
```

格式支持 `.csv`、`.csv.gz` 和 `.parquet`。真实数据衍生文件按课程数据规则保管，
**不提交到 Git 仓库**，通过课程允许的共享位置交接。

| 列 | 类型 | 含义 |
|---|---|---|
| `hadm_id` | string | 与 `index.csv` 的 `hadm_id` 一一对应，必须是同一份冻结 index 里的值 |
| `hour` | int ≥ 0 | 从 `admittime` 起算的整小时编号；`hour=0` 表示入院后第 1 小时内 |
| `variable` | string | 变量名，全表用统一拼写 |
| `value` | float | 该小时的数值 |

不需要提供缺失行：缺的就是缺的，框架会构造 mask。**不要用 0 填充缺失值**，
那会让"测得 0"和"没测"无法区分。

## 2. 时间窗口与截断（最重要的一条）

**所有观测必须严格早于该样本的预测时点 `study_time`。**

`index.csv` 已经提供 `hours_since_admission = study_time - admittime`（小时）。
对某次住院，只能包含 `hour < hours_since_admission` 的行。

训练框架会**再次强制**这条规则（`TableClinicalProvider` 会丢弃超出截止点的行），
但那是防御性措施，不是让抽取侧可以不做。理由：评测项目明确要求
"所有特征必须以胸片 `study_time` 为截止点"。

### 先看这个：`chartevents` 只覆盖进过 ICU 的住院

这一条比时间窗口更要紧，请先读。

MIMIC-IV 分成 `hosp` 和 `icu` 两个模块，**`chartevents` 属于 `icu` 模块**，
数据来自 MetaVision ICU 系统。没有 ICU 住院的患者在 `chartevents` 里**一行都没有**。

官方文档的数字（v3.0）：

- 有住院记录的患者：**223,452**
- 有 ICU 住院的患者：**65,366**（约 29%）

而 MIMIC-CXR 的选择偏倚会让情况更复杂：文档明确写着
"MIMIC-CXR is only available between 2011 - 2016 for patients who were admitted to
the emergency department"。经急诊入院的患者大多进普通病房，不是 ICU。

按模块清点本文第 3 节的 17 个变量：

| 来源模块 | 变量 | 覆盖 |
|---|---|---|
| `icu` / `chartevents` | 心率、呼吸、SpO2、收缩压/舒张压/平均压、体温、FiO2、GCS 三项、身高、体重、毛细血管再充盈、pH（223830） | **仅 ICU 住院** |
| `hosp` / `labevents` | Glucose（50931）、pH（50820） | 所有住院 |

**17 个里有 13 个是 ICU-only。** 在当前这个以住院为单位的 cohort 上，
非 ICU 患者的临床分支会几乎全是缺失值——模型不是"学不好"，而是根本没有输入。

抽取工具会把这件事直接报出来：QA 报告的 `source_coverage.chartevents.admission_coverage`
低于 50% 时会写一条 `warnings`。**拿到真实数据后第一件事就是看这个数字。**

### 本课程子集的实测结果

对课程提供的 CXR 子集（1,000 名患者，5,534 张 DICOM，3,351 个 study）
与 MIMIC-IV v3.1 的 `icustays` / `admissions` 交叉统计：

| 指标 | 数值 |
|---|---:|
| CXR 子集患者数 | 1,000 |
| 这些患者的住院次数 | 3,214 |
| **其中院内死亡** | **64** |
| 整体死亡率 | 2.0% |
| 有 ICU 住院的患者 | 300 / 1,000（30.0%） |
| **有 ICU 住院的住院次数** | **451 / 3,214（14.0%）** |
| ICU 住院的死亡率 | 13.5% |
| 非 ICU 住院的死亡率 | 0.1% |

三点结论：

1. **住院级 cohort 下 86% 的住院没有任何 `chartevents`**，13/17 个变量全缺。
2. **缺失本身是强预测因子。** 非 ICU 住院死亡率 0.1%，ICU 住院 13.5%，相差 135 倍。
   因此"这条记录有没有临床数据"几乎等价于"是否进过 ICU"，
   而 `missing_indicator` 通道会把这个信息直接交给模型。
   模型可能学到的是 ICU 收治与否，而不是生理状态，这会让 RQ2 失去意义。
   采用住院级 cohort 时必须在报告中讨论这一混杂。
3. **全子集只有 64 个死亡事件。** 按 70/15/15 划分，测试集约 9–10 个正例，
   AUROC 的 95% CI 宽度会远超 MeTra 论文中多模态与单模态之间 0.052 的差距。
   这一点与 A/B 案的选择无关，需要通过更大的子集、交叉验证，
   或改用事件更多的结局来缓解。

复现命令（`icu_coverage` 统计脚本不在仓库内，属于一次性核查）：
交叉 `icustays.hadm_id` 与 `admissions` 中属于 CXR 子集患者的住院记录即可。

可选的应对：

1. 改用 ICU 级 cohort（下面的 B 案），这也正是 MeTra 的做法；
2. 保留住院级 cohort，但把临床分支的变量集缩小到 `labevents` 能覆盖的项目
   （代价是丢掉 MeTra 最看重的生命体征）；
3. 补充 `hosp/omr` 表。它含有血压、身高、体重、BMI、eGFR，覆盖全体住院患者，
   而且常有入院前的 baseline 值。**注意**：`omr` 只有 `chartdate`（日期粒度，无时刻），
   套到小时级截止点上需要先定一个保守规则，否则同一天的值可能晚于 `study_time`。
   目前抽取工具**未**实现 omr，要用需要先定这条规则。

### 一个需要全组确认的问题

MeTra 用的是 **ICU 入住后前 48 小时**，得到完整的 `K×48`。
现在的 cohort 是**住院入院后 48 小时内的首张 AP/PA 胸片**，`study_time` 是唯一预测时点。
如果某位患者的胸片在入院后第 6 小时，那么可用的临床历史只有 0–6 小时，拿不到 48 格。

- **A 案（训练框架当前假设）**：窗口 = `[admittime, study_time)`，长度可变，
  用 mask 标记未覆盖的小时。无泄漏，现有 split 不需要重新冻结。
  报告中需说明这与 MeTra 的 ICU 48 小时窗口不同。
  **但要先解决上一节的覆盖率问题**，否则非 ICU 患者没有生命体征可用。
- **B 案**：改为 ICU 级，预测时点 = ICU 入住 + 48 小时，胸片从该窗口内选。
  更贴近 MeTra，并且**同时解决覆盖率问题**——cohort 里每个人按定义都有 `chartevents`。
  代价是需要引入 `icustays`、重建 cohort、重新冻结 split。

上一节的覆盖率问题使 B 案的性价比明显变高：它一次解决两个问题。
但 B 案有两个必须处理的坑：

- **Immortal time bias**：要求患者存活满 48 小时才进入 cohort，
  会系统性排除最早死亡的一批人，正类分布因此改变。MeTra 也有同样问题，
  报告中应当说明。
- **同一患者的多次 ICU 住院**：官方文档指出非连续的 ICU 住院保留为不同的 `stay_id`，
  且"it is up to the investigator to appropriately handle these cases"。
  需要明确规则（例如只取首次 ICU 住院），并保证同一 `subject_id` 不跨 split。

两种方案本文件的格式都不用改，只是 `hour` 的起点定义不同
（A 案从 `admittime` 起算，B 案从 ICU `intime` 起算）。**请在开始抽取前确认。**

### 其他从 MIMIC-IV 文档确认的点

- **itemid 会在版本之间变化。** v3.1 的发布说明记载：`d_labitems` 和 `labevents`
  的部分 itemid 在 v2.2 到 v3.0 之间发生了非预期的改变，v3.1 才修回。
  这正是抽取前必须跑 `--inspect-items` 的原因。
- **`labevents` 含院外/急诊检验。** 文档说明该表包含住院之外的记录，
  这类行的 `hadm_id` 可能为空。抽取工具会丢弃空 `hadm_id`，
  并按 cohort 与时间窗口过滤，因此院外检验不会混入。
- **官方不做数据清洗。** 文档明言保留真实世界的杂讯，
  "Implausible values may be present"。这就是本工具按生理范围过滤并计数的依据。
- **时间平移按 `subject_id` 一致。** 同一患者内部的时间差保持真实，
  所以 MIMIC-CXR 与 MIMIC-IV 按 `subject_id` 关联后直接比较时刻是正确的；
  但不同患者之间的年份不可比。
- **`anchor_age` 对 89 岁以上统一记为 91。** 流水线已用 `age_is_topcoded` 标记。

## 3. 变量清单

MeTra 基于标准 MIMIC benchmark 抽取的 17 个变量，其中 2 个因 100% 缺失被剔除，最终 K=15：

```text
Capillary refill rate            Glasgow coma scale verbal response   Respiratory rate
Diastolic blood pressure         Glucose                              Systolic blood pressure
Fraction inspired oxygen         Heart Rate                           Temperature
Glasgow coma scale eye opening   Height                               Weight
Glasgow coma scale motor response Mean blood pressure                 pH
Glasgow coma scale total         Oxygen saturation
```

请同时提供实际保留的变量清单和各自缺失率。框架不写死 K：
`variable_names` 从数据读取，因此剔除全缺失变量不需要改代码。

Glasgow coma scale 的分项在 MIMIC 中是文字类别（如 `Spontaneously`、`To speech`），
需要映射为有序数值并**把映射表一并交付**，否则模型学到的是任意编码。

## 4. 单位与异常值

- 同一变量全表统一单位（例如 Temperature 统一用摄氏或华氏，不要混用）。
- 生理不可能值（心率 0、体温 200）请在抽取阶段剔除并计数，不要留给训练侧。
- 请提供每个变量的剔除数量，写进 QA 报告，和现有 `qa_report.json` 风格一致。

## 5. 不要做的事

- **不要做标准化 / z-score。** 框架在 train split 上拟合 mean/std 并冻结到
  val/test，统计量保存为 `clinical_normalizer.json`。抽取侧如果先标准化，
  train 的统计量就会泄漏到 val/test。
- **不要做跨患者插补。** 前向填充（用同一次住院的上一次测量值）是可以的，
  但请说明是否做了，以及填充的最长跨度。
- **不要包含结局后信息**：`dischtime`、`deathtime`、出院诊断、出院用药等一律不能进。
- **不要重建 split。** 用数据流水线已冻结的那一份。

## 6. 交接清单

- [ ] 长格式表文件（路径通过课程允许的共享位置告知）
- [ ] 实际变量清单 + 每个变量的缺失率
- [ ] GCS 等类别变量的数值映射表
- [ ] 单位说明和异常值剔除统计
- [ ] `hour` 起点定义（A 案还是 B 案）
- [ ] 是否做了前向填充，及其最长跨度
- [ ] 对应的冻结 `run_dir` 和 `index.csv` 的 SHA-256

## 7. 用 `bn5212-extract-clinical` 生成这张表

### 先核对 itemid

不同 MIMIC 版本的 itemid 会变。抽取前先确认配置里的 itemid 在你的数据里指的是什么：

> **已对 MIMIC-IV v3.1 核对过**：`configs/clinical_items.json` 里的 33 个 itemid
> 在 `d_items` / `d_labitems` 中**全部命中，label 全部相符**，其中包括三组需要单位换算的项：
> `226707 = Height`（英寸）、`226531 = Admission Weight (lbs.)`、
> `223761 = Temperature Fahrenheit`。
> `224308 = Capillary Refill L` 与 `223951 = Capillary Refill R` 两侧都存在。
> 换用其他 MIMIC 版本时仍需重跑这一步。

```bash
bn5212-extract-clinical --inspect-items \
  --run-dir /srv/derived/bn5212/mortality_v1 \
  --mimic-root /srv/course/mimiciv/3.1
```

它会把每个 itemid 在 `d_items` / `d_labitems` 里的真实 label 打出来。
**逐行看一遍**：itemid 写错时这里会显示成不相干的 label 或 `<NOT FOUND>`，
比抽完才发现抽错信号便宜得多。需要调整就直接改 `configs/clinical_items.json`。

### 抽取

```bash
bn5212-extract-clinical \
  --run-dir /srv/derived/bn5212/mortality_v1 \
  --mimic-root /srv/course/mimiciv/3.1 \
  --output /srv/derived/bn5212/clinical_features.csv \
  --max-hours 48
```

`chartevents` 约 30 GB，按 `--chunksize`（默认 100 万行）流式读取，内存占用平稳。
因为一进来就按 cohort 的 `hadm_id` 和截止点过滤，输出通常只有几十 MB。

同时生成 `clinical_features_qa.json`，包含：每个变量的观测数、
覆盖到的住院数、缺失率，以及分类丢弃计数（不在 cohort / 晚于 study_time /
storetime 晚于 study_time / 超出生理范围 / 无法解析）。
**这份报告直接可以写进课程报告的数据章节。**

### 这个实现强制的规则

- 两个时钟都卡：`charttime` 和 `storetime` 都必须早于 `study_time`。
  一条第 3 小时测量、第 50 小时才入库的值，在拍片当下拿不到，会被丢弃并计数。
- 边界是 `study_time` 这个时刻本身，不是它所在的整点。
  所以包含 `study_time` 的那个小时格可能只被部分填充，这是对的。
- 一次住院有多张胸片时，按**最早**的 `study_time` 截断，保证该住院的任何样本都不会看到未来。
- 生命体征按小时取均值；GCS 等序数分数取该小时**最后一次**观测
  （取均值会造出从未出现过的分数）。
- GCS total 在三个分项同一小时都存在时求和得出；MIMIC-IV 不直接记录它。
- 不做标准化、不做跨患者插补 —— 这些留给训练侧在 train split 上拟合。

## 8. 训练侧怎么接

拿到文件后，训练侧只改配置，不改代码：

```bash
bn5212-train --config configs/clinical_only.json \
  --run-dir /srv/derived/bn5212/mortality_v1 \
  --set data.clinical_provider=table \
  --set data.clinical_source=/srv/derived/bn5212/clinical_features.csv.gz \
  --set data.clinical_timesteps=48
```

框架会自动校验：列是否齐全、`hour` 是否为非负整数、`value` 是否可解析、
是否出现未声明的变量名，并在读取时按 `hours_since_admission` 截断。
任何一项不符合会直接报错，不会静默丢样本。
