# Git版本控制设置指南

本指南将帮助您完成Delta-ME13项目的Git版本控制设置。

## 第一步：安装Git（如果尚未安装）

### Windows系统

1. 访问 [Git官网](https://git-scm.com/download/win)
2. 下载并安装Git for Windows
3. 安装时选择默认选项即可
4. 安装完成后，重启命令行工具

### 验证安装

打开PowerShell或命令提示符，运行：
```bash
git --version
```

如果显示版本号（如 `git version 2.40.0`），说明安装成功。

## 第二步：配置Git（首次使用）

```bash
# 设置用户名（使用您的GitHub用户名）
git config --global user.name "Tim-He9"

# 设置邮箱（使用您的GitHub邮箱）
git config --global user.email "your-email@example.com"

# 验证配置
git config --list
```

## 第三步：初始化本地Git仓库

在项目目录（`D:\code\δ-me13`）中执行：

```bash
# 初始化Git仓库
git init

# 查看状态
git status
```

## 第四步：添加文件到Git

```bash
# 添加所有文件（.gitignore会自动排除不需要的文件）
git add .

# 或者分步添加
git add *.py
git add *.md
git add *.txt
git add config/
git add src/
```

## 第五步：创建初始提交

```bash
# 创建提交
git commit -m "Initial commit: MARL沙盘平台基础代码"

# 查看提交历史
git log
```

## 第六步：连接远程仓库

根据您在GitHub上创建的仓库信息：

```bash
# 添加远程仓库（使用您的仓库URL）
git remote add origin https://github.com/Tim-He9/Delta-ME13.git

# 验证远程仓库
git remote -v
```

**注意**：如果您的仓库名称不是 `Delta-ME13`，请替换为实际的仓库名称。

## 第七步：推送代码到GitHub

```bash
# 首次推送（设置上游分支）
git branch -M main
git push -u origin main
```

如果遇到认证问题，您可能需要：

1. **使用Personal Access Token**（推荐）：
   - 在GitHub设置中生成Token
   - 推送时使用Token作为密码

2. **或使用SSH**：
   ```bash
   # 更改远程URL为SSH
   git remote set-url origin git@github.com:Tim-He9/Delta-ME13.git
   ```

## 常用Git命令

### 查看状态
```bash
git status
```

### 查看更改
```bash
git diff
```

### 添加文件
```bash
git add <文件名>
git add .  # 添加所有更改
```

### 提交更改
```bash
git commit -m "提交说明"
```

### 推送更改
```bash
git push
```

### 拉取更改
```bash
git pull
```

### 查看历史
```bash
git log
git log --oneline  # 简洁版本
```

## 故障排除

### 问题1：Git命令不可用

**解决方案**：
- 确保Git已正确安装
- 重启命令行工具
- 检查PATH环境变量

### 问题2：推送时要求认证

**解决方案**：
1. 使用Personal Access Token（推荐）
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - 生成新token，勾选 `repo` 权限
   - 推送时使用token作为密码

2. 或配置SSH密钥
   ```bash
   # 生成SSH密钥
   ssh-keygen -t ed25519 -C "your-email@example.com"
   # 将公钥添加到GitHub
   ```

### 问题3：远程仓库已存在内容

如果远程仓库已经有README等文件：

```bash
# 先拉取远程内容
git pull origin main --allow-unrelated-histories

# 解决可能的冲突后
git push -u origin main
```

## 下一步

完成设置后，您可以：

1. 继续开发，定期提交更改
2. 创建分支进行功能开发
3. 使用Pull Request进行代码审查
4. 添加更多文档和示例

## 参考资源

- [Git官方文档](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Git命令速查表](https://education.github.com/git-cheat-sheet-education.pdf)

