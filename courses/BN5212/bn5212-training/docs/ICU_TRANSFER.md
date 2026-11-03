# ICU 转入预测配置与运行

使用专用入口 `scripts/run_icu_transfer.sh`。默认只运行 concat、joint self-attention、
cross-attention adapter 三个多模态模型，不训练单模态。使用当前280次住院队列和现有患者划分。

## 队列与标签

- 研究单位：一次住院，按患者划分 train/val/test，seed=5212。
- t0：入院后 48 小时内首张符合条件的 AP/PA 胸片采集时间。
- 排除同次住院中在 t0 或更早已进入 ICU 的患者，包括已出 ICU 后再拍片的情况。
- 阳性：首次 ICU intime 严格晚于 t0，且严格早于同次住院的死亡/出院截止时间。
- 阴性：同次住院没有满足上述条件的未来 ICU 转入。
- ICU entry 来自 `icustays.csv`，不是 `hospital_expire_flag`。缺失/无法解析的 ICU intime
  不能自动当成阴性；会中止构建要求检查源数据。

预处理配置为 `configs/preparation/icu_transfer_pipeline.json`，输出独立的
`data/runs/icu_transfer_v1/`，`dataset_spec.json.task` 和 `label_name` 均为 `icu_transfer`。
不会修改死亡任务的 run 或把死亡标签改名。实际人数及事件数需构建后核对，
不强制凑到总结里的 244 次住院 / 42 个事件，也不直接使用 1054 次住院压缩包替代该队列。

## 临床输入

窗口为 `[t0-48h,t0)`，与“胸片在入院后48小时内”是两个不同约束。
检验按患者和时间匹配，支持急诊检验缺少 hadm_id 的情况；
charttime 和 storetime 均须严格早于 t0。storetime 未知的记录排除。
观测 mask 和标准化继续使用原框架，统计量仅拟合训练患者。

明确使用 18 个临床变量：

```text
White Blood Cells, Hemoglobin, Hematocrit, Platelet Count,
Sodium, Potassium, Chloride, Bicarbonate, Urea Nitrogen, Creatinine,
Glucose, Lactate, pH, INR, Bilirubin Total, Albumin, Age, Female
```

16 项检验从本机 `d_labitems.csv` 以 Blood fluid 和明确标签解析 itemid，不沿用另一任务的
17 项 ICU 生命体征。无法解析任一变量会明确报错。解析后的 itemids 保存在
`clinical_features.items.json`，覆盖率与缺失率在 `clinical_features.qa.json`。
年龄和性别在最后一小时格记录一次，Female=1、Male=0，未知则保留缺失。
固定变量清单确保全缺失的检验仍保留 token；不会随 split 改变输入维度。

训练配置为 `configs/icu_transfer/cross_attention_adapter.json`。
ViT-B/16 使用冻结 ImageNet backbone，投影到64维。临床 encoder 在第一阶段的当前训练患者上拟合，
**不加载死亡预测的预训练权重**。因此本任务无需 `--pretrain-clinical`；
继续传该参数会报错，避免把原外部死亡预训练流程混入新任务。
三个多模态模型使用同一份转入数据及公共训练配置，包含Joint attention。
默认冻结ImageNet ViT backbone，训练影像投影、临床encoder、融合模块和head。
Concat设置concat_image_pooling=patch_mean和concat_project=false，输出128维直接接MLP。
已有兼容单模态权重时，可通过clinical_checkpoint/image_checkpoint和freeze_encoders=true
直接复用并冻结，无需重训单模态。默认学习率为 0.0003，允许从训练开始选择最低验证 loss checkpoint。
完整设置见 [v3 运行说明](RUN_THREE_MULTIMODAL.md)。

## 下载解压完成后的命令（v3）

在训练目录激活环境后，先准备转入队列、18变量临床表和图像缓存：

```bash
bash scripts/run_icu_transfer.sh \
  --prepare-only --device cuda --run-id transfer_prepare_v1
```

此命令自动选择匹配的训练和 pipeline 配置。准备结果保存在
`data/prepared_configs/transfer_prepare_v1.json`。

再顺序运行三个多模态模型的训练、验证及 benchmark：

```bash
bash scripts/run_icu_transfer.sh \
  --base-config data/prepared_configs/transfer_prepare_v1.json \
  --device cuda --run-id transfer_multimodal_v3_rerun
```

也可合并为一条命令：

```bash
bash scripts/run_icu_transfer.sh --prepare --device cuda --run-id transfer_multimodal_v3_rerun
```

训练及评分结果在 `outputs/three_multimodal/<run-id>/`，续跑追加 `--resume`。
若追加单模态对照或修改配置，请使用新的 run-id。
默认生成三个多模态的leaderboard，不自动生成相对单模态的配对增益报告。
仅显式启用--include-unimodal时才训练单模态并生成paired_auroc_gains.csv。

任务校验会拒绝意外传入死亡预测的 dataset run。不要使用死亡任务的 index SHA-256
验证新的转入队列；新队列有自己的 hash 和患者划分。

先前单模态结果来自另一任务/输入定义，不能直接用于本任务的性能对比。
本次只训练多模态；已有单模态对照可在确认同队列及患者划分后单独接入比较。

## 与历史244次住院实验的边界

总结中的历史实验为186位患者、244次住院、42个事件，采用患者分组嵌套交叉验证。
本地当前转入run为218位患者、280次住院；当前入口是固定train/val/test。
因此当前三个多模态模型内部可以比较，但不能直接与历史AUROC 0.544/0.707作配对比较。
要严格复现，仍需原队列/外层和内层患者划分、确切变量顺序、预处理及单模态权重。
总结的16项检验加年龄性别与“17 token”本身不一致，获得原清单前不删除变量。
原设计的完整冻结需要已有临床encoder和影像投影权重。默认入口不重训单模态，
因此采用冻结影像backbone、联合训练其余模块的设置；已有权重时可直接加载并完整冻结。
当前结果见 [v3 结果索引](../results/multimodal_v3/README.md)。

## 验证

`tests/test_transfer.py` 覆盖转入队列、18 变量临床抽取、三个融合模型和可选单模态对照。
`tests/test_three_multimodal_script.py` 覆盖训练、benchmark 和续跑时保留权重。
测试使用合成数据；实验指标以 v3 原始评测及结果汇总为准。
