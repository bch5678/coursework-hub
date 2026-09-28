# Fusion 模块接口 v1.0

**交给多模态负责人。** 本文说明怎么在训练框架里加一个新的融合策略。
目标是：你只写融合逻辑，不碰 encoder、训练循环、预测导出和评测对接，
这样 MeTra 和你的方法之间的差异就只有融合方式本身。

## 接口

所有融合模块实现同一个签名：

```python
forward(image_tokens:    torch.Tensor | None,   # [B, N, D]，CLS 在 index 0
        clinical_tokens: torch.Tensor | None,   # [B, M, D]
        clinical_mask:   torch.Tensor | None,   # [B, M]，True 表示有效 token
        ) -> torch.Tensor                       # [B, output_dim]
```

输出是一个 pooled 患者表示，交给统一的 prediction head。
`output_dim` 默认等于 `embed_dim`；双向融合等需要更宽输出时重写该属性。

两个类属性声明依赖，配置不匹配时框架会在建模型阶段就报错，而不是训练到一半炸：

```python
uses_image = True
uses_clinical = True
```

## 加一个新策略

在 `src/bn5212_training/fusion/` 下新建一个文件：

```python
from ..config import FusionConfig
from ..registry import FUSIONS
from .base import FusionModule, masked_mean, build_key_padding_mask


class MyFusion(FusionModule):
    uses_image = True
    uses_clinical = True

    def __init__(self, embed_dim: int, ...):
        super().__init__(embed_dim)
        ...

    def forward(self, image_tokens=None, clinical_tokens=None, clinical_mask=None):
        self.check_inputs(image_tokens, clinical_tokens)
        ...
        return pooled            # [B, self.output_dim]


@FUSIONS.register("my_fusion")
def build_my_fusion(cfg: FusionConfig, *, num_image_tokens, num_clinical_tokens, **_):
    return MyFusion(cfg.embed_dim, ...)
```

然后在 `fusion/__init__.py` 的 import 行里加上这个模块，让注册生效。复制一份
`configs/cross_attention.json`，把 `fusion.name` 改成 `my_fusion`，就可以跑：

```bash
bn5212-train --config configs/my_fusion.json --run-dir <frozen run>
```

**不需要改动** trainer、model、data、predict、cli 或任何配置解析代码。

## 必须遵守的几件事

### 1. 处理 `clinical_mask`

临床窗口是变长的（见 `docs/CLINICAL_FEATURE_SPEC.md` 第 2 节）。
被 padding 的 token 不能参与计算：

- pooling 用 `masked_mean(tokens, mask)`，不要用 `tokens.mean(dim=1)`；
- attention 的 key 用 `build_key_padding_mask(mask, length, batch, device)`，
  它返回 `nn.MultiheadAttention` 需要的 ignore-mask，并且会处理"整行都被 mask"
  的情况——否则 softmax 会出 NaN。

`tests/test_fusion.py::test_padded_clinical_tokens_do_not_change_other_samples`
会检查这一点。

### 2. 不要在融合模块里引入额外的容量差异

如果你的模块比 MeTra 多了几层 MLP，那么"cross-attention 更好"就可能只是参数更多。
`run_manifest.json` 会记录分支级参数量（`parameters.fusion`），
请在报告里一并给出，让对比可解释。

### 3. 记录 attention 权重供 RQ3 使用

如果你的模块产生跨模态 attention，实现 `last_attention()` 返回
`[B, heads, M, N]`（M = 临床 token 数，N = 影像 token 数），并用一个开关控制是否
记录——训练时记录会关掉 PyTorch 的 fast attention path，拖慢速度。
参考 `cross_attention.py` 里的 `record_attention`。

只要形状对，框架会自动画出两张 RQ3 图：
`attention_matrix.png`（变量 × patch）和 `attention_patch_maps.png`
（每个变量折回 patch 网格）。两张图共用同一色阶，所以变量之间可以直接比较。

> 前提是 `clinical_encoder.tokenization = "per_variable"`（默认值），
> 这时一个 token 就是一个临床变量。改成 `per_timestep` 的话，
> attention 的含义会变成"某个小时关注了哪里"，RQ3 的叙述要跟着改。

## 现有实现

| name | uses image | uses clinical | 说明 |
|---|---|---|---|
| `image_only` | ✓ | | CXR-only，取 ViT 的 CLS 或 mean pooling |
| `clinical_only` | | ✓ | Clinical-only，masked mean pooling |
| `concat_mlp` | ✓ | ✓ | 简单融合参照：各自 pool → concat → 一层 MLP |
| `joint_self_attention` | ✓ | ✓ | MeTra baseline |
| `cross_attention` | ✓ | ✓ | 提出的方法，**归你所有** |

`cross_attention.py` 目前是框架提供的参考实现，目的是让四个实验今天就能端到端跑通。
你可以直接在上面改，也可以整个换掉——只要保持签名和注册名不变，其他文件都不用动。

`fusion.direction` 已经支持项目计划的 Optional 1 和 Optional 2：

- `clinical_to_image`（默认）：`Q=C, K=V=I`
- `image_to_clinical`：`Q=I, K=V=C`
- `bidirectional`：两个方向都做，concat 后 `output_dim = 2 × embed_dim`

## 跑通之前请确认

```bash
.venv/bin/python -m pytest tests/test_fusion.py -q
```

`tests/test_fusion.py` 对每个注册的策略做参数化测试：输出形状、缺失模态的拒绝行为、
mask 语义、attention 权重形状与归一化。新策略加进注册表后会自动被这些测试覆盖。
