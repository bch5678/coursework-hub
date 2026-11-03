# 多模态 v3 结果索引

唯一保留批次：`transfer_multimodal_v3_20261030_161749`，2026-10-30 完成。
任务为早期胸片后的同次住院 ICU 转入。280 次住院、218 位患者，固定患者级 train/val/test 划分。

| 模型 | Val AUROC | Test AUROC | Test AUPRC |
| --- | ---: | ---: | ---: |
| Concat | 0.7063 | 0.7952 | 0.4881 |
| Joint attention | 0.7344 | 0.7667 | 0.3570 |
| Cross-attention Adapter | 0.6438 | 0.6714 | 0.3256 |

测试集仅 37 次住院、7 个阳性；当前结果未验证相对同队列单模态的融合增益。

- [完整模型设计、置信区间及图表](../../README.md#multimodal-v3)
- [验证汇总](validation_comparison.csv)与[测试汇总](benchmark_leaderboard.csv)：原始 v3 汇总表的原样副本。
- [训练与复现](../../docs/RUN_THREE_MULTIMODAL.md)及[v3 配置](../../configs/icu_transfer/)。
- [本地完整结果](../../outputs/three_multimodal/transfer_multimodal_v3_20261030_161749/)：权重、预测、图表、日志、配置和 benchmark 报告。
- [清理核验](../../outputs/training_launches/transfer_multimodal_v3_20261030_161749/cleanup_verification.json)：删除清单与保留产物 SHA-256。

2026-11-03 清理了旧多模态 v1/v2 输出、旧死亡任务多模态汇总/配置、重复启动器及旧 cross-attention 实现。
三个 v3 模型的训练、预测、权重、配置和 benchmark 原始文件均经 SHA-256 核对保持不变。
原启动记录作为实验来源保留，其中提及旧版本的路径仅代表历史记录。
本目录只有汇总指标；完整患者级输出和权重仅存在本地被 Git 忽略的 `outputs/`。

清理后验证：训练项目合成回归 210 项通过、1 项跳过；Bash 语法、v3 参数一致性与文档本地链接检查通过。
