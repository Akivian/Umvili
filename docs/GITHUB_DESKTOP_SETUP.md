# GitHub Desktop 正确设置指南

## 问题说明

如果您遇到以下情况：
- GitHub Desktop指向的是项目的一个**拷贝**（比如`D:\code\Umvili`）
- 但您实际开发在另一个文件夹（比如`D:\code\δ-me13`）
- 每次提交都要**复制粘贴文件**

这说明GitHub Desktop没有指向您实际工作的文件夹。

## 正确的做法

### 方案1：在GitHub Desktop中打开现有项目（推荐）

这是最简单的方法，直接让GitHub Desktop指向您正在工作的文件夹。

#### 步骤：

1. **打开GitHub Desktop**
   - 启动GitHub Desktop应用程序

2. **添加现有仓库**
   - 点击菜单：**File** → **Add Local Repository**
   - 或者点击左上角的 **"+"** 按钮 → **Add Existing Repository**

3. **选择您的项目文件夹**
   - 点击 **"Choose..."** 按钮
   - 浏览到您的实际项目文件夹：`D:\code\δ-me13`
   - 选择该文件夹，点击 **"选择文件夹"**

4. **确认添加**
   - GitHub Desktop会检测到这是一个Git仓库（如果已经初始化）
   - 如果没有初始化，会提示您初始化

5. **验证**
   - 在GitHub Desktop中，您应该能看到：
     - 仓库名称显示在顶部
     - 左侧显示所有文件
     - 可以正常查看更改、提交、推送

#### 如果文件夹还没有Git仓库

如果您的`D:\code\δ-me13`文件夹还没有Git仓库：

1. **在GitHub Desktop中初始化**
   - 选择文件夹后，GitHub Desktop会提示初始化
   - 点击 **"Create a Repository"**

2. **设置远程仓库**
   - Repository → Repository Settings
   - Remote标签
   - 添加远程仓库URL：`https://github.com/Akivian/Umvili.git`
   - 点击Save

### 方案2：将项目移动到GitHub Desktop克隆的位置

如果您已经在GitHub Desktop中克隆了仓库到某个位置（比如`D:\code\Umvili`），可以将您的项目文件移动到那里。

#### 步骤：

1. **确认GitHub Desktop中的仓库位置**
   - 在GitHub Desktop中，查看仓库路径
   - Repository → Show in Explorer（或Show in Finder）

2. **备份当前项目**（重要！）
   - 先备份您的`D:\code\δ-me13`文件夹

3. **移动文件**
   - 将`D:\code\δ-me13`中的所有文件
   - 复制到GitHub Desktop的仓库文件夹（比如`D:\code\Umvili`）
   - **注意**：不要覆盖`.git`文件夹（如果存在）

4. **在GitHub Desktop中查看**
   - 应该能看到所有文件
   - 可以正常提交和推送

### 方案3：重新克隆到正确位置

如果上面的方法都不方便，可以重新克隆仓库到您想要的位置。

#### 步骤：

1. **在GitHub Desktop中克隆**
   - File → Clone Repository
   - 选择GitHub.com标签
   - 找到`Akivian/Umvili`
   - 选择本地路径：`D:\code\`
   - 这样会创建`D:\code\Umvili`文件夹

2. **将项目文件复制过去**
   - 将`D:\code\δ-me13`中的所有文件
   - 复制到`D:\code\Umvili`
   - 注意排除`.git`文件夹

3. **在GitHub Desktop中提交**
   - 应该能看到所有文件
   - 提交并推送

## 推荐方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案1：打开现有项目** | 最简单，不需要移动文件 | 需要确保文件夹是Git仓库 | ⭐⭐⭐⭐⭐ |
| **方案2：移动文件** | 保持Git历史 | 需要移动文件 | ⭐⭐⭐ |
| **方案3：重新克隆** | 干净的开始 | 可能丢失本地Git历史 | ⭐⭐ |

## 详细步骤：方案1（推荐）

### 步骤1：检查您的项目文件夹是否有Git仓库

在PowerShell中，进入您的项目文件夹：

```powershell
cd D:\code\δ-me13
git status
```

**如果显示Git信息**：说明已经有Git仓库，可以直接用方案1。

**如果显示错误**：说明还没有Git仓库，需要先初始化。

### 步骤2A：如果已有Git仓库

1. 打开GitHub Desktop
2. File → Add Local Repository
3. 选择`D:\code\δ-me13`
4. 完成！

### 步骤2B：如果没有Git仓库

#### 方法1：在GitHub Desktop中初始化

1. 打开GitHub Desktop
2. File → Add Local Repository
3. 选择`D:\code\δ-me13`
4. 如果提示不是Git仓库，点击 **"Create a Repository"**
5. 填写仓库名称：`Umvili`
6. 选择本地路径：`D:\code\δ-me13`
7. 点击 **"Create Repository"**

#### 方法2：在命令行中初始化

```powershell
cd D:\code\δ-me13
git init
git remote add origin https://github.com/Akivian/Umvili.git
```

然后在GitHub Desktop中：
1. File → Add Local Repository
2. 选择`D:\code\δ-me13`

### 步骤3：设置远程仓库

1. 在GitHub Desktop中
2. Repository → Repository Settings
3. Remote标签
4. 确保Primary remote repository是：
   ```
   https://github.com/Akivian/Umvili.git
   ```
5. 如果不是，更新它
6. 点击Save

### 步骤4：验证设置

1. 在GitHub Desktop中，您应该能看到：
   - 仓库名称：`Umvili`
   - 当前分支：`main`（或其他）
   - 文件列表

2. 做一个测试更改：
   - 修改任意文件
   - 在GitHub Desktop中应该能看到更改
   - 可以正常提交和推送

## 常见问题

### 问题1：GitHub Desktop显示"这不是一个Git仓库"

**解决方案：**
- 在项目文件夹中初始化Git：
  ```powershell
  cd D:\code\δ-me13
  git init
  ```

### 问题2：GitHub Desktop找不到远程仓库

**解决方案：**
- Repository → Repository Settings → Remote
- 添加远程仓库URL：`https://github.com/Akivian/Umvili.git`

### 问题3：推送时提示"Repository not found"

**解决方案：**
1. 检查GitHub用户名是否正确（应该是`Akivian`）
2. 检查仓库名是否正确（应该是`Umvili`）
3. 确认您有推送权限

### 问题4：本地更改和远程不一致

**解决方案：**
1. 先拉取远程更改：
   ```bash
   git pull origin main
   ```
2. 解决可能的冲突
3. 然后推送

## 设置完成后的工作流程

设置完成后，您的工作流程应该是：

1. **在代码编辑器中工作**
   - 打开`D:\code\δ-me13`中的文件
   - 编辑、保存

2. **在GitHub Desktop中提交**
   - 自动检测到更改
   - 查看更改
   - 提交
   - 推送

**不再需要复制粘贴！**

## 检查清单

设置完成后，确认：

- [ ] GitHub Desktop显示正确的仓库路径（`D:\code\δ-me13`）
- [ ] 可以查看文件列表
- [ ] 修改文件后，GitHub Desktop能检测到更改
- [ ] 可以正常提交更改
- [ ] 可以正常推送到GitHub
- [ ] 远程仓库URL正确（`https://github.com/Akivian/Umvili.git`）

## 总结

**关键点：**
- GitHub Desktop应该指向您**实际工作的文件夹**
- 不需要复制粘贴，直接在原文件夹工作
- 确保文件夹是Git仓库（有`.git`文件夹）
- 确保远程仓库URL正确

**推荐流程：**
1. 在GitHub Desktop中打开现有项目文件夹
2. 设置远程仓库URL
3. 开始正常使用，不再需要复制粘贴

---

如果还有问题，请检查：
- 项目文件夹路径是否正确
- 是否有Git仓库（`.git`文件夹）
- 远程仓库URL是否正确
- GitHub Desktop是否已登录

