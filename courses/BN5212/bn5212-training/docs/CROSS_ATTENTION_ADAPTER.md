# Cross-attention Adapter

当前多模态 v3 的交叉注意力实现为 `fusion/cross_attention_adapter.py`，
注册名为 `cross_attention_adapter`。另外两个融合模型是 Concat 与 Joint attention。

临床 token 作为 Query，影像 patch token 作为 Key/Value；排除影像 CLS。
每层使用 pre-norm 多头注意力、临床残差及 FFN，最后对有效临床 token 做 masked mean 和 LayerNorm。
v3 使用 18 个临床变量、196 个影像 patch、64 维嵌入、4 heads、1 层、dropout 0.1。
详细模型和结果见 [v3 报告](../README.md#multimodal-v3)。

## 输入与缺失值

影像输入为 `[B, 1+N, D]`，index 0 是 CLS/global summary，其余为真实 patch。
临床输入为 `[B, M, D]`，`clinical_mask[B,M]` 的 True 表示有效位置。
缺失变量不参与池化，记录的对应 attention 行为零；全缺失输入输出零向量，由分类头决定预测。
上游 provider 对空窗口的保护会保留一个有效位置；adapter 按收到的 mask 处理。
`embed_dim` 须被 `num_heads` 整除，`depth >= 1`，方向仅支持 `clinical_to_image`。

## 训练与注意力导出

三模型匹配比较使用 [统一入口](RUN_THREE_MULTIMODAL.md)。单模型可运行：

```bash
bn5212-train --config configs/icu_transfer/cross_attention_adapter.json --run-id adapter_v3_rerun
```

ViT backbone 冻结，768→64 影像投影、临床 encoder、融合与分类头参与训练。
临床分支在本队列训练患者上拟合，不加载死亡任务 checkpoint。

设置 `model.fusion.record_attention = True` 后，`last_attention()` 返回最终层
`[B, heads, M, N]` detached 权重，N 不含 CLS。关闭开关后下一次前向会清空记录。
评估模式下有效查询权重和为 1；训练 dropout 开启时不保证为 1。
训练器支持导出变量 × patch 热图；注意力图不能直接证明因果关系。

```bash
python -m pytest tests/test_cross_attention_adapter.py tests/test_fusion.py tests/test_training.py
```

测试覆盖 patch-only 注意力、缺失值隔离、全缺失输入、梯度及 checkpoint 预测一致性。
