# GitHub仓库重命名指南

## 如何在GitHub上重命名仓库

### 步骤1：在GitHub上重命名

1. **访问您的仓库页面**
   - 访问：`https://github.com/Akivian/Umvili`（或您当前的仓库名）

2. **打开设置**
   - 点击顶部的 **"Settings"**（设置）标签

3. **找到仓库名称**
   - 向下滚动到 **"Repository name"**（仓库名称）部分
   - 通常在设置页面的顶部

4. **输入新名称**
   - 如果需要，更改仓库名称（当前：`Umvili`）
   - 点击 **"Rename"**（重命名）按钮

5. **确认**
   - GitHub会警告您关于重命名
   - 确认重命名

### 步骤2：更新本地仓库

在GitHub上重命名后，更新您的本地仓库：

#### 方法A：更新远程URL（推荐）

```bash
# 检查当前远程URL
git remote -v

# 更新远程URL到新的仓库名
git remote set-url origin https://github.com/Akivian/Umvili.git

# 验证更改
git remote -v
```

#### 方法B：删除并重新添加远程

```bash
# 删除旧远程
git remote remove origin

# 添加新远程
git remote add origin https://github.com/Akivian/Umvili.git

# 验证
git remote -v
```

### 步骤3：测试连接

```bash
# 从远程拉取以测试连接
git fetch origin

# 如果成功，就完成了！
```

### 步骤4：更新GitHub Desktop（如果使用）

如果您使用GitHub Desktop：

1. **打开GitHub Desktop**
2. **Repository** → **Repository Settings**（仓库设置）
3. **Remote**（远程）标签
4. 更新 **Primary remote repository**（主远程仓库）URL为：
   ```
   https://github.com/Akivian/Umvili.git
   ```
5. 点击 **Save**（保存）

### 步骤5：更新任何引用

#### 更新README.md

如果您的README包含仓库URL，更新它们：

```markdown
# 旧
git clone https://github.com/Akivian/旧仓库名.git

# 新
git clone https://github.com/Akivian/Umvili.git
```

#### 更新文档

检查并更新任何引用旧仓库名的文档文件。

## 重要提示

⚠️ **会改变的内容：**
- 仓库URL改变
- 克隆URL改变
- 所有指向仓库的链接需要更新

✅ **不会改变的内容：**
- Git历史（所有提交保留）
- Issues和Pull Requests
- Stars和Forks
- 本地仓库文件

## 重命名后

1. **更新任何书签**您保存的
2. **更新任何CI/CD配置**（GitHub Actions等）
3. **通知协作者**关于新的仓库名
4. **更新任何外部链接**指向您的仓库

## 故障排除

### 问题："Repository not found"（找不到仓库）

**解决方案：**
- 确保您已正确更新远程URL
- 验证GitHub上的新仓库名
- 检查您的GitHub用户名是否正确

### 问题：GitHub Desktop显示旧名称

**解决方案：**
- 关闭并重新打开GitHub Desktop
- 或在设置中手动更新远程URL

### 问题：无法推送到远程

**解决方案：**
```bash
# 检查远程URL
git remote -v

# 如果错误，更新它
git remote set-url origin https://github.com/Akivian/Umvili.git

# 再次尝试推送
git push origin main
```

## 快速检查清单

- [ ] 在GitHub上重命名了仓库
- [ ] 更新了本地远程URL
- [ ] 测试了连接（git fetch）
- [ ] 更新了GitHub Desktop（如果使用）
- [ ] 更新了README.md引用
- [ ] 更新了任何文档引用
- [ ] 通知了协作者（如果有）

---

**注意：** 旧的仓库URL会自动重定向到新的，但最好更新所有引用。
