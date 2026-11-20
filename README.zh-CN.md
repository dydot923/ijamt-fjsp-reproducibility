# 三目标柔性作业车间调度复现代码

本仓库对应杜宇阳、张勇斌的论文《Objective-matched neighborhoods and exact prefix evaluation for three-objective flexible job-shop scheduling》。论文正在准备投稿，未登记文章 DOI。仓库软件版本为 0.1.0，实际公开日期与网址以发布后的仓库记录为准。

代码目录包含算法、解码器、前缀复用、实验与统计分析入口，以及六个人工构造小例。E0 工程输入、第三方 Mk 基准和完整运行记录已单独整理为 `ijamt-fjsp-data-results-v0.1.0.zip`，作为同版本 GitHub Release 的数据附件。数据再分发范围已于2026-11-20由用户确认。代码和小例可以独立运行；完整论文复现需要同时取得代码与该数据附件。

## 最快运行方法

安装 Python 3.12，在当前文件所在目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B scripts/run_demo.py
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

默认示例真实运行 T1、H-MVNS-2、种子 530001 和 200 次新编码评价，核对排程可行性、真前沿与可达前沿，并与原始小例运行结果比较。结果保存在 `outputs/`，该目录不提交到仓库。示例结果不能代替 678 次主确认实验。

只检查发布文件的 SHA256 和范围，不安装科学依赖也可以运行：

```powershell
python scripts/check_release.py
```

完整复现见 [REPRODUCTION.md](docs/REPRODUCTION.md)，公开数据范围见 [DATA.md](docs/DATA.md)，GitHub 发布步骤见 [PUBLISH.zh-CN.md](docs/PUBLISH.zh-CN.md)。

本仓库没有稿件 Word、作者声明、个人备份或本机安装目录。科学源文件保留原项目相对结构，未为本次发布修改算法、种子或性能数值。原始输入与新旧运行记录在本机额外数据包内；公开前需要确认相应再分发条件。

基金：国家重点研发计划（2023YFE0113800）。软件许可证尚未由权利人选定，本次未代替作者授予 MIT 等许可。
