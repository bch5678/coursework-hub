# tmux 下载课程原始数据

入口：`bash scripts/download_course_data.sh`。读取本机的
`data/dropbox_source.local.json`，其中已保存本次提供的共享链接、密码和解析后的两个文件地址。
此文件及登录 cookie 位于 Git 忽略的数据目录，权限为 600；密码不写进公共脚本或日志。
也可用环境变量 `BN5212_DROPBOX_PASSWORD` 覆盖私有配置中的共享密码。

本次清单经 Dropbox 目录元数据核对，且各用 1 KB Range 请求验证：

| 文件 | 字节数 | GiB |
| --- | ---: | ---: |
| `MIMIC-CXR/BN5212_MIMIC-CXR.zip` | 43,276,593,988 | 40.30 |
| `MIMIC-IV/MIMIC_IV.zip` | 9,404,880,624 | 8.76 |

合计约 52.7 GB / 49.06 GiB。不下载重复打包的 `dataset.zip`、四个 `data_subset*.zip` 或 NOTE。
不需要 Dropbox 账号，脚本走共享链接的正常密码验证流程。
另已只读检查真实 ZIP 的中央目录：CXR 包匹配 5534 个 DICOM，MIMIC-IV 包包含
脚本需要的全部 8 张表，两个 ZIP 均未加密。该检查没有下载完整数据包。

## 运行

```bash
tmux new -s bn5212-download
cd /home/coursework-hub/courses/BN5212/bn5212-training
source .venv/bin/activate
bash scripts/download_course_data.sh --extract
```

也可用 `PYTHON_BIN=/opt/conda/envs/qwen3/bin/python bash scripts/download_course_data.sh --extract`。
源码目录调整后，已有环境先执行 `python -m pip install --no-deps -e .` 更新包映射。
若当前 Python 缺少 requests，先 `python -m pip install -e '.[download]'`。

`--extract` 在每个 ZIP 下载完成后提取所需输入；只下载 ZIP 时省略该选项。
只想确认文件和大小：`bash scripts/download_course_data.sh --list-only`，这不会访问网络。
支持 `--only cxr`、`--only iv`、`--output-dir /path/to/downloads` 和 `--data-root /path/to/raw-data`。

## 断点、校验和日志

- 下载中断直接重跑同一命令，自动从 `.part` 续传；网络异常默认最多重试 20 次。
- 只有大小正确、ZIP 目录可读取并完成 SHA-256 记录后，才发布最终 `.zip` 文件。
- 已完成的 ZIP 重跑时先验证本地 SHA-256，再跳过下载。
- 提供者没有给出 SHA-256；本次记录的是下载后本地计算值，不能称为已与发布者校验值匹配。
  若后续拿到可信值，可在私有清单对应条目添加 `sha256` 字段，脚本会强制比对。
- 服务器不再支持续传或远端文件变化时保留 `.part` 并报错；确认后可用 `--restart` 重下未完成文件。
- 解压读取 ZIP 自身 CRC，按已完成文件续跑；未完成的单个解压文件会重新开始。
- 若 ZIP 本身加密，使用 `DATA_ARCHIVE_PASSWORD`；它与 Dropbox 共享密码属于不同配置。

实时日志：

```bash
tail -f data/download_logs/dropbox.log
```

按 `Ctrl-b` 再按 `d` 脱离 tmux，之后 `tmux attach -t bn5212-download` 返回。

## 输出与下一步

```text
data/downloads/MIMIC-CXR/BN5212_MIMIC-CXR.zip
data/downloads/MIMIC-IV/MIMIC_IV.zip
data/mimic-cxr/2.1.0/files/pXX/p<subject>/s<study>/*.dcm
data/mimiciv/hosp/{admissions,patients,d_labitems,labevents,omr}.csv
data/mimiciv/icu/{d_items,icustays,chartevents}.csv
```

默认输出与 `configs/preparation/icu_transfer_pipeline.json` 一致。
完成后再运行已有预处理/训练脚本：

```bash
bash scripts/run_icu_transfer.sh \
  --prepare --pipeline-config configs/preparation/icu_transfer_pipeline.json \
  --device cuda --run-id three_models_downloaded_v1
```

私有下载清单只在本机保存；把代码复制到其他机器时，还需单独配置该清单。
