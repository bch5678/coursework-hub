# Coursework Hub

课程作业与小组项目的共享仓库。课程项目使用独立目录，每个分工可以保留自己的 README、依赖和运行入口。

## 已有项目

| 课程 | 项目 | 入口 |
|---|---|---|
| BN5212 | MIMIC-IV + MIMIC-CXR cohort、时间对齐、标签、患者划分、数据清洗与 DataLoader | [项目说明](courses/BN5212/bn5212-data-pipeline/README.md) |

BN5212 数据流水线对应 MIMIC-IV v3.1 和 MIMIC-CXR v2.1.0。默认任务是从早期胸片预测住院死亡。项目包含完整数据接口、服务器一键脚本和 23 项合成数据测试；实际数据由每位获授权的成员通过自己的本地/服务器路径读取。

## 目录约定

```text
courses/
  BN5212/
    bn5212-data-pipeline/   # 当前已经实现的数据项目
  <其他课程代码>/
    <项目名称>/            # 按需要新增
templates/
  project-README.md        # 新项目说明模板
.github/
  workflows/              # 自动检查配置
```

同一门课可以有多个项目；每个项目单独管理依赖。BN5212 建模成员也可以在现有流水线项目内新增训练和评估模块，并共同使用一份冻结的 index 和患者划分。

## 开始参与

1. 由仓库所有者在 GitHub 中邀请组员，组员接受邀请后参与协作。
2. 下载或克隆仓库，进入对应项目目录，按该项目的 README 配置环境。
3. 为自己的改动建立分支，例如 `bn5212/add-training`，完成后提交 Pull Request，请组员查看再合入 main。
4. 新课程作业放到 `courses/BN5212/<分工角色>/`，复制 [README 模板](templates/project-README.md)，并更新本页项目表。

详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。每个项目保留自己的依赖，避免给一门课改环境时影响其他课程。

## BN5212 快速测试

```bash
cd courses/BN5212/bn5212-data-pipeline
python -m pip install -r requirements.txt
python -m pytest
```

配置的 GitHub Actions 会在 BN5212 代码或对应工作流变更时运行合成数据测试，也支持手动运行。它不会读取本地/服务器的课程受限数据，也不需要数据账号或凭证。工作流按 [GitHub 官方 Python 测试说明](https://docs.github.com/en/actions/tutorials/build-and-test-code/python) 配置；首次实际云端执行情况以仓库 Actions 页面为准。

## 数据与共享范围

仓库保存代码、说明、公开可共享材料和人工测试示例。BN5212 的原始图像、患者表、真实衍生 index/split、课程下载密码和私人配置不应提交。所有真实数据继续保存在课程允许的位置，组员分别使用配置指向自己的目录。

仓库可以先设为私有，供受邀组员协作。其他课程的作业按对应课程对合作与代码共享的要求加入。
