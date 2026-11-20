# 将代码发布到仓库

以下按 GitHub 编写。Gitee/GitLab 的 Git 命令基本相同，远程地址换成对应平台实际仓库地址即可；网页按钮以实际界面为准。本说明不代表代码已经上传。推荐仓库名为 `ijamt-fjsp-reproducibility`。

## 哪些文件应该上传

上传当前代码目录的**全部内容**，让 `README.md` 直接位于仓库根目录；同时保留 `.gitignore`、`.gitattributes` 这两个以点开头的文件。代码 ZIP 应先解压，不能只把 ZIP 当成全部源代码上传。

上级目录的 `ijamt-fjsp-data-results-v0.1.0.zip` 包含 E0 工程输入、第三方 Mk 基准和完整运行记录，已按你确认的可公开再分发范围准备。**这份约27MB的数据包要上传到同版本 Release 的附件区**，与仓库内的代码配套。代码目录内也有六个人工验证实例，可以独立运行。软件及数据的额外复用许可证没有代作者选择；如需授予MIT等开放许可，由相应权利人确认后添加准确文本。

## 最简方法 用网页上传

本版本共82个发布文件，均远小于网页单文件大小限制，可以一次上传。

1. 登录 GitHub，选择 **New repository**，输入仓库名。确定发布范围后选择相应可见性；不勾选生成 README、`.gitignore` 或许可证。
2. 创建空仓库后点击 **uploading an existing file**。已有文件的仓库使用 **Add file → Upload files**。
3. 在资源管理器打开代码目录，选中里面的全部文件和子目录，包括 `.gitignore`、`.gitattributes`，拖入网页上传区域。不要只拖入压缩包；让 `README.md` 直接位于仓库根目录。
4. 填写提交说明，例如 `Add FJSP reproducibility code v0.1.0`，按页面按钮提交。完成后确认文件夹结构及 README 正常显示。
5. 重新下载仓库 ZIP，在另一个目录运行 README 中的示例。确认后再按下文创建 `v0.1.0` Release。

如果已有仓库包含其他项目，建议使用下方的克隆与分支方法，先审阅变更再合并。

## 使用 Git 命令发布到新建空仓库

1. 登录 GitHub，选择 **New repository**。名称可用 `ijamt-fjsp-reproducibility`。
2. 选择需要的可见性。若用于公开代码引用，应在确认发布范围后使用 Public；私有仓库链接不能供普通读者访问。
3. **不要勾选**初始化 README、`.gitignore` 或许可证，因为本地已经有相应文件。点击 Create repository，复制页面给出的 HTTPS 仓库地址。
4. 在本地代码文件夹空白处打开 PowerShell。确认 `Get-ChildItem` 能看到当前 `README.md` 和 `scripts`，然后执行：

```powershell
git init -b main
git add .
git status --short
git commit -m "Prepare reproducible FJSP research code"
git remote add origin https://github.com/YOUR_USERNAME/ijamt-fjsp-reproducibility.git
git push -u origin main
```

把 `YOUR_USERNAME` 和仓库名替换为你页面上复制的实际地址。如果第一次提交提示缺少身份，按 GitHub 帐号设置本地 Git 提交姓名和邮箱，或使用帐号的 noreply 邮箱，再重试提交。此处不替你猜测 GitHub 用户名或更改全局 Git 配置。

如果 Git 弹出浏览器登录，使用你自己的 GitHub 帐号完成正常登录。不把 token 写入源码、README 或远程 URL；不把密码/token 发到聊天里。

如已存在 origin，先用 `git remote -v` 查看；确认目标仓库后，使用 `git remote set-url origin 实际地址`。不要重复执行 remote add。后续更新通常只需要 `git add .`、`git commit -m "说明本次改动"` 和 `git push`。

## 上传到已经有内容的仓库

如果远程仓库已经有 README 或其他提交，先克隆它，不在这里强行推送覆盖：

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
Set-Location YOUR_REPOSITORY
git switch -c add-fjsp-reproduction
```

把准备好的代码目录**内容**复制到这个克隆目录中，保留原来的 `.git`。如 README 有冲突，先合并两份说明。若该仓库包含其他独立项目，可把本项目放在 `ijamt-fjsp-reproducibility/` 子目录中；后续运行命令先进入该子目录。检查文件后：

```powershell
git add .
git status --short
git commit -m "Add FJSP reproduction package"
git push -u origin add-fjsp-reproduction
```

回到 GitHub 创建 Pull Request，核对差异后合并到默认分支。不需要 `--force`。

## 用 GitHub Desktop

安装官方 GitHub Desktop 并登录自己的帐号。若仓库已存在，先选择 **Clone repository**，克隆后复制代码内容，在 Changes 中核对，填写提交说明并提交，再选择 **Push origin**。若创建新仓库，可先按上方Git方法在本地执行 `git init -b main`，再通过 **File → Add local repository** 选择当前代码目录，在界面完成首次提交和 **Publish repository**。发布时核对仓库名及可见性。

## 发布后核对

在仓库网页上确认能直接看到 `README.md`、`replication`、`examples`、`scripts`、`tests` 和 `CITATION.cff`。用 **Code → Download ZIP** 重新下载，解压到另一个目录，按 README 安装依赖并运行示例。确认仓库中没有本地完整数据 ZIP、Word 稿件、缓存或安装环境。

GitHub 网页上传每次最多100个文件，每个文件不超过25MiB。本项目适合使用 Git 或 Desktop，能完整保留文件结构。[GitHub 文件上传说明](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)

## 固定一个论文引用版本

上传并检查后，进入仓库 **Releases → Draft a new release**，建立标签 `v0.1.0`，标题可用 `FJSP reproducibility code and data v0.1.0`。使用根目录 `RELEASE_NOTES.md` 的正文。把 `ijamt-fjsp-data-results-v0.1.0.zip` 拖入附件区；代码 ZIP 也可作为附件一起放入。核对标签及两项文件后点击 Publish release。

完整数据包超过普通仓库网页上传的25MiB限制，所以不通过 **Add file → Upload files** 上传，而使用 **Release附件区**。Release支持附带文件供下载。[GitHub大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) · [Release创建步骤](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)

发布后重新下载代码和数据两项，将数据包解压到代码根目录。按 `docs/REPRODUCTION.md` 执行示例或完整派生重算，核对版本与文件可访问性，再把实际Release网址用于论文声明。

需要数据/软件 DOI 时，可以按官方说明将 GitHub 仓库连接到 Zenodo，并存档对应 Release。GitHub 的仓库地址或标签本身不是 DOI。只有获得真实 DOI 后才回填稿件及 `CITATION.cff`。

## 如何更新论文的数据代码声明

完整代码和数据附件都已发布并确认读者能够下载后，可使用下句，并把方括号替换为真实版本网址后再放入稿件：

> The source code, input datasets, experimental configurations, and computational results supporting this study are available in the versioned release at [actual release URL or DOI].

如果只上传了代码、尚未附上完整数据，则只能声明代码和人工小例已公开，待数据附件发布后再使用完整表述。提交前由两位作者核对实际共享内容。仓库发布前不把上述带方括号的模板当成已经公开的正式声明。

官方说明：[本地代码上传](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github) · [GitHub Desktop 发布](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop) · [创建 Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository) · [Zenodo 存档与 DOI](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
