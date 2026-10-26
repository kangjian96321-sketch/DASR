# 上传 GitHub：手动操作

本次仅整理本地代码，没有替你创建仓库、上传文件或公开论文资料。

**已有 DASR 仓库的本次更名更新：**不要重新初始化仓库，直接参考
[更名迁移说明](RENAME_20261026.md) 的分支/补丁流程。以下新建仓库流程仅适用于尚无仓库的情况。

## 上传前

1. 将完整 ZIP 解压到新的目录。不要覆盖正在训练的项目。
2. 查看 README_zh.md、protocol.md、TEST_REPORT.md；确认这一版就是计划公开的训练协议。
3. 确认作者/论文信息、代码权属和第三方许可，尤其是 CRT ALR 适配代码。
   本次没有擅自给全部源码添加 MIT 许可。建议先建 **Private** 仓库，确认后再公开。
4. 不要往待上传目录放数据集、模型权重、真实实验日志、微信文件、密钥或私人路径。
   `.gitignore` 不是保密保证，提交前仍需检查暂存区。

## 方法一：Git 命令（推荐）

在 GitHub 新建空仓库，例如 `DASR`，选择 Private。
**不要勾选初始化 README、.gitignore 或 License**，因为本地已包含文件。
打开解压后项目根目录的终端，按需将 YOUR_NAME 和仓库名换成自己的：

```bash
git init -b main
git add .
git status --short
git diff --cached --stat
git diff --cached
git commit -m "Release continual adversarial training implementation"
git remote add origin https://github.com/YOUR_NAME/DASR.git
git remote -v
git push -u origin main
```

第一次使用 Git 若提示身份未设置，先在此仓库配置自己的姓名和邮箱：

```bash
git config user.name "YOUR NAME"
git config user.email "YOUR EMAIL OR GITHUB NOREPLY EMAIL"
```

使用 GitHub 提示的浏览器登录、凭据管理器或 SSH；不要把 token 写入脚本、README 或 remote URL。
如果远端不是空仓库，不要用强制 push 覆盖，先检查远端历史。

上传后在 GitHub 检查 README、文件树和 Actions → CPU regression。
机器磁盘不足导致的测试跳过会显示为 skipped；正式实验需在自己的 GPU 环境重新运行。

## 方法二：GitHub Desktop

先在新的解压目录初始化本地 Git 仓库，再通过 Desktop 的 Add local repository 添加。
检查 Changes，只提交源码、配置和文档；发布时先保持 Private。
不要将整个旧工作目录作为仓库，也不要仅上传 ZIP 而不展开源码。

## 后续版本

```bash
git add 指定的修改文件
git diff --cached
git commit -m "Describe the protocol or implementation change"
git push
```

训练协议变化时更新 docs/changes.md，并新建实验目录。通过验证后再创建版本标签和 Release。
权重若以后需要发布，应单独审查后提供下载，不塞入 Git 提交历史。

依据：[GitHub 官方导入说明](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、
[GitHub 许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。
