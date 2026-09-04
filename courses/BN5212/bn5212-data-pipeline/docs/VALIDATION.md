# 验证记录

验证日期：2026-09-04。输入全部由 `scripts/make_synthetic_data.py` 人工构造；没有使用课程患者数据、临床图像或凭证。

## 执行环境

Windows，Python 3.12.14。NumPy 2.3.5、pandas 3.0.1、Pillow 12.3.0、pydicom 3.0.2、PyTorch 2.14.0、pytest 9.1.1。依赖版本另列于 `requirements-tested.txt`。服务器可使用 `requirements.txt` 中声明的兼容范围；实际部署仍应运行验收脚本。

## 已执行的检查

自动测试 **23 项通过**。包含：

- PNG 元数据输入、DICOM 头扫描两种模式，均完成 cohort → 标签 → split → index → DataLoader。
- 两类图像均使用真实 PyTorch Dataset/DataLoader 读取；训练/验证/测试均能生成 float32 图像与 int64 标签。
- `num_workers=2` 的真实多进程读取、Dataset 序列化、相同随机种子的批次可复现性。
- 小数秒和丢失前导零的 StudyTime、非法时间、入院边界、死亡/出院边界。
- 重复源行、冲突 ID、缺失图片、损坏图像、未成年人、未知标签、死亡标志冲突。
- 一张影像匹配多个重叠住院时排除，不任意选择一次住院。
- 输入顺序打乱后的 patient-level split 一致性、患者/住院/研究/图像跨集合隔离。
- 自定义 admission_binary 标签、多图模式下住院权重合计为 1。
- DICOM MONOCHROME1/2 极性、1/3 通道输出。
- 完整产物 SHA-256、输出目录不可覆盖、非法 split 和样本不足的失败行为。
- ZIP 正常解压与复用、路径穿越拦截、tar 符号链接拦截、解压大小限制、下载校验和。
- 模拟 HTTP 的断点续传及凭证仅发送到指定主机。测试没有连接真实受限下载地址。

人工构造了 36 位患者、每位两次住院、不同视角/时间的影像，并添加边界与错误记录。默认过滤后，PNG 与 DICOM 两种模式均得到 35 位患者、68 次住院、68 张入选图像。68 个住院标签中 9 个为 1。这些数字只用于确认代码行为，**不是 MIMIC 的真实规模、标签分布或患病率**。

## 命令入口实跑

除 pytest 外，还独立运行了以下等效命令：

```bash
python scripts/make_synthetic_data.py --output <temporary-demo-directory>
python run_pipeline.py --config <temporary-demo-directory>/synthetic_config.json --test-loader
```

输出：train 46 条、val 12 条、test 10 条，三个集合首批形状均为 `(4, 1, 32, 32)`，完整索引时间约束、患者隔离、产物校验和及 DataLoader contract 检查通过。合成配置为加速测试将图像尺寸设为 32；正式配置默认 224。

## 验证边界

尚未获得课程数据，因此未运行真实下载、课程子集字段核对、全量性能测试或真实 cohort 统计。需要服务器运行者设置真实路径，运行 `run_pipeline.sh` / `run_pipeline.py` 和 `test_dataloader.py` 后读取实际 QA 报告。

Linux shell 封装已做静态检查；当前环境未执行 Linux Bash，也未验证特定服务器 CUDA 构建。跨平台 Python 入口、Windows PowerShell 一键封装与 Windows 多进程 DataLoader 均已执行通过。DICOM 测试覆盖未压缩单帧灰度图；压缩 Transfer Syntax 的解码器依赖取决于课程文件。

冻结下游实验时，应保留本次 `index.csv`、`split_assignments.csv`、`dataset_spec.json`、配置与依赖版本。真实结果不会随此代码项目包分发。
