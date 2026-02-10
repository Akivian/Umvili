# 版本控制使用指南

## 概述

本指南详细说明如何在Umvili项目中使用Git进行版本控制，包括日常操作流程、最佳实践和常见场景。

## 目录

1. [虚拟环境(.venv) - 需要提交吗？](#虚拟环境)
2. [日常版本控制流程](#日常流程)
3. [使用GitHub Desktop](#github-desktop)
4. [使用Git命令行](#git命令行)
5. [提交信息规范](#提交信息)
6. [常见场景](#常见场景)
7. [最佳实践](#最佳实践)

---

## 虚拟环境

### ❌ **不要提交.venv到仓库**

**为什么？**
- 虚拟环境是**机器特定的**（不同操作系统、Python版本）
- 包含**数千个文件**（不必要的体积）
- 可以通过`requirements.txt`**重新创建**
- 频繁变化，会在Git历史中产生**大量噪音**

### ✅ **正确的做法**

1. **保持`.venv`在`.gitignore`中**（已配置）
2. **提交`requirements.txt`**（这是别人需要的）
3. **在README中说明Python版本**

### 其他人如何获取依赖

```bash
# 克隆仓库
git clone https://github.com/Akivian/Umvili.git
cd Umvili

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

---

## 日常流程

### 标准工作流程（推荐）

```
1. 修改代码
   ↓
2. 查看更改
   ↓
3. 暂存更改 (git add)
   ↓
4. 提交更改 (git commit)
   ↓
5. 推送到GitHub (git push)
```

### 详细步骤

#### 步骤1：修改代码

编辑文件、添加功能、修复bug等。

#### 步骤2：查看更改

**在GitHub Desktop中：**
- 左侧面板显示所有更改的文件
- 点击文件名查看具体改动（diff）

**在命令行中：**
```bash
git status          # 查看哪些文件改变了
git diff            # 查看详细的改动内容
```

#### 步骤3：暂存更改

**在GitHub Desktop中：**
- 勾选要提交的文件旁边的复选框
- 或勾选"全选"来提交所有更改

**在命令行中：**
```bash
# 暂存特定文件
git add main.py

# 暂存所有更改
git add .

# 暂存特定目录
git add src/core/
```

#### 步骤4：提交更改

**在GitHub Desktop中：**
- 在左下角写提交信息
- 点击"Commit to main"（或您的分支名）

**在命令行中：**
```bash
git commit -m "您的提交信息"
```

#### 步骤5：推送到GitHub

**在GitHub Desktop中：**
- 点击"Push origin"按钮（右上角）

**在命令行中：**
```bash
git push origin main
```

---

## GitHub Desktop

### 基本工作流程

1. **修改代码**
   - 在代码编辑器中编辑文件
   - GitHub Desktop会自动检测更改

2. **查看更改**
   - 左侧面板：更改的文件列表
   - 右侧面板：差异视图（显示具体改动）
   - 点击文件名查看详情

3. **选择要提交的文件**
   - 勾选要提交的文件旁边的复选框
   - 取消勾选不想提交的文件

4. **编写提交信息**
   - 左下角：提交信息输入框
   - 写清楚、描述性的信息
   - 示例："添加：新的智能体类型配置"

5. **提交**
   - 点击"Commit to main"按钮
   - 更改保存在本地

6. **推送到GitHub**
   - 点击"Push origin"按钮（右上角）
   - 更改上传到GitHub

### 界面说明

```
GitHub Desktop界面：
┌─────────────────────────────────────┐
│  [仓库名称]  [Push origin]          │
├─────────────────────────────────────┤
│ 更改的文件（左侧面板）                │
│ ☑ main.py                           │
│ ☑ src/core/agents.py                │
│ ☐ temp_file.py  (未暂存)            │
├─────────────────────────────────────┤
│ 差异视图（右侧面板）                  │
│ 显示选中文件的具体改动                │
├─────────────────────────────────────┤
│ 提交信息：                           │
│ [添加：新的智能体配置]                │
│ [Commit to main]                    │
└─────────────────────────────────────┘
```

---

## Git命令行

### 基本命令

```bash
# 查看状态（哪些文件改变了）
git status

# 查看详细改动
git diff

# 暂存所有更改
git add .

# 暂存特定文件
git add main.py

# 提交并写信息
git commit -m "您的提交信息"

# 推送到GitHub
git push origin main

# 拉取最新更改
git pull origin main
```

### 完整示例

```bash
# 1. 查看更改了什么
git status

# 2. 查看详细改动
git diff

# 3. 暂存更改
git add .

# 4. 提交
git commit -m "修复：智能体初始化bug"

# 5. 推送
git push origin main
```

---

## 提交信息

### 好的提交信息

✅ **清晰且描述性强：**
```
添加：新的QMIX智能体配置
修复：模拟循环中的内存泄漏
更新：README安装说明
重构：智能体工厂模式
```

✅ **规范格式：**
```
类型: 简短描述

详细描述（如果需要）
- 要点1
- 要点2
```

### 提交类型

- `添加:` - 新功能
- `修复:` - 修复bug
- `更新:` - 更新现有功能
- `重构:` - 代码重构
- `文档:` - 文档更改
- `样式:` - 代码格式更改
- `测试:` - 测试更改
- `维护:` - 维护任务

### 不好的提交信息

❌ **避免：**
```
fix
updated
changes
test
asdf
```

---

## 常见场景

### 场景1：日常开发

**您修改了代码，想保存进度：**

```bash
# 快速流程
git add .
git commit -m "添加：功能描述"
git push
```

**在GitHub Desktop中：**
1. 勾选所有文件
2. 写提交信息
3. 提交
4. 推送

### 场景2：撤销最后一次提交（未推送）

**您提交了但想修改提交：**

```bash
# 撤销提交，保留更改
git reset --soft HEAD~1

# 修改后，重新提交
git add .
git commit -m "更好的提交信息"
```

**在GitHub Desktop中：**
- 右键点击最后一次提交 → "撤销提交"

### 场景3：丢弃本地更改

**您做了更改但想丢弃：**

```bash
# 丢弃所有未提交的更改
git checkout .

# 丢弃特定文件
git checkout -- main.py
```

**在GitHub Desktop中：**
- 右键点击文件 → "丢弃更改"

### 场景4：从GitHub更新

**别人推送了更改，您想拉取：**

```bash
# 拉取最新更改
git pull origin main
```

**在GitHub Desktop中：**
- 点击"Fetch origin"或"Pull origin"

### 场景5：创建新分支

**您想开发功能但不影响主分支：**

```bash
# 创建并切换到新分支
git checkout -b feature/new-agent

# 做更改，提交
git add .
git commit -m "添加：新的智能体类型"

# 推送新分支
git push origin feature/new-agent
```

**在GitHub Desktop中：**
- Branch → New branch
- 命名，创建
- 做更改，提交，推送

### 场景6：合并分支

**您完成了功能，想合并到主分支：**

```bash
# 切换到主分支
git checkout main

# 拉取最新
git pull origin main

# 合并功能分支
git merge feature/new-agent

# 推送
git push origin main
```

**在GitHub Desktop中：**
- 切换到主分支
- Branch → Merge into current branch
- 选择功能分支
- 推送

---

## 最佳实践

### 1. 频繁提交

✅ **好的做法：**
- 完成一个小功能就提交
- 修复一个bug就提交
- 提交逻辑单元的工作

❌ **不好的做法：**
- 一天结束时才提交所有内容
- 把不相关的更改一起提交

### 2. 写清晰的提交信息

✅ **好的：**
```
添加：QMIX智能体配置支持
修复：智能体更新循环中的内存泄漏
```

❌ **不好的：**
```
fix
update
changes
```

### 3. 提交前检查

- 始终检查您要提交的内容
- 不要提交临时文件
- 不要提交注释掉的代码
- 不要提交调试打印语句

### 4. 保持提交聚焦

✅ **好的：**
- 一次提交 = 一个逻辑更改
- 相关更改一起提交

❌ **不好的：**
- 混合bug修复和新功能
- 混合代码更改和文档

### 5. 推送前先拉取

```bash
# 推送前总是先拉取
git pull origin main
git push origin main
```

这样可以避免冲突。

### 6. 不要提交生成的文件

已在`.gitignore`中：
- `__pycache__/`
- `*.pyc`
- `.venv/`
- `*.log`

### 7. 提交前测试

- 运行您的代码
- 确保它能工作
- 然后提交

---

## 快速参考

### 日常命令

```bash
# 查看状态
git status

# 暂存所有
git add .

# 提交
git commit -m "您的信息"

# 推送
git push origin main

# 拉取
git pull origin main
```

### 有用命令

```bash
# 查看提交历史
git log

# 查看改动
git diff

# 撤销未提交的更改
git checkout .

# 创建分支
git checkout -b branch-name

# 切换分支
git checkout branch-name
```

---

## 故障排除

### 问题："Your branch is ahead of origin"

**解决方案：** 推送您的提交
```bash
git push origin main
```

### 问题："Your branch is behind origin"

**解决方案：** 拉取最新更改
```bash
git pull origin main
```

### 问题：合并冲突

**解决方案：**
1. 打开冲突的文件
2. 查找`<<<<<<<`标记
3. 手动解决冲突
4. 暂存已解决的文件
5. 提交

### 问题：意外提交了.venv

**解决方案：**
```bash
# 从Git中移除（保留本地）
git rm -r --cached .venv

# 提交移除
git commit -m "移除：.venv从仓库"

# 推送
git push origin main
```

---

## 总结

### ✅ 应该提交：
- 源代码（`.py`文件）
- 配置文件（`config/*.json`）
- 文档（`*.md`）
- `requirements.txt`
- `README.md`
- `LICENSE`

### ❌ 不应该提交：
- `.venv/`或`venv/`（虚拟环境）
- `__pycache__/`（Python缓存）
- `*.pyc`（编译的Python）
- `*.log`（日志文件）
- `.env`（环境变量）
- IDE设置（`.vscode/`，`.idea/`）

### 日常流程：
1. 做更改
2. 查看（`git status`）
3. 暂存（`git add`）
4. 提交（`git commit`）
5. 推送（`git push`）

---

**记住：** 频繁提交，提交有意义的更改，写清晰的信息！

