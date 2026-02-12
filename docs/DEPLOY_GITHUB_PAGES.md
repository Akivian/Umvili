# 通过 GitHub Pages 发布 Umvili 项目站

发布后访问地址：**https://\<你的 GitHub 用户名>.github.io/Umvili/**

---

## 方式一：用 GitHub Actions 自动部署（推荐）

已为你配置好工作流：推送到 `main` 分支时会自动构建并部署。

### 1. 在仓库里开启 GitHub Pages

1. 打开仓库 **Umvili** 的 GitHub 网页。
2. 点击 **Settings** → 左侧 **Pages**。
3. 在 **Build and deployment** 里：
   - **Source** 选 **GitHub Actions**（不要选 “Deploy from a branch”）。
4. 保存后无需再改分支或目录。

### 2. 推送代码触发部署

把包含 `.github/workflows/deploy-docs.yml` 的代码推到 `main`：

```bash
git add .
git commit -m "Add GitHub Pages deploy workflow"
git push origin main
```

### 3. 查看部署结果

- 打开仓库 **Actions** 页，点最新的 **Deploy Docs to GitHub Pages** 运行。
- 成功完成后，访问：**https://\<用户名>.github.io/Umvili/**  
  （把 \<用户名> 换成你的 GitHub 用户名）

若仓库默认分支不是 `main`（例如是 `master`），需把 `.github/workflows/deploy-docs.yml` 里的 `branches: [main]` 改成你的默认分支名。

---

## 方式二：本地构建后手动部署

不想用 Actions 时，可以本地构建后推送到 `gh-pages` 分支。

### 1. 本地构建

在项目根目录执行：

```bash
cd docs
npm ci
npm run build
```

会在 `docs/out` 得到静态文件。

### 2. 推送到 gh-pages 分支（任选一种）

**方法 A：用 gh-pages 包**

```bash
# 在项目根目录 D:\code\Web\Umvili
npx gh-pages -d docs/out
```

**方法 B：手动推**

```bash
# 进入 out，用当前目录作为新仓库推到一个孤立分支
cd docs/out
git init
git checkout -b gh-pages
git add .
git commit -m "Deploy docs"
git remote add origin https://github.com/<你的用户名>/Umvili.git
git push -f origin gh-pages
```

### 3. 在 GitHub 里用分支当 Pages 源

1. 仓库 **Settings** → **Pages**。
2. **Source** 选 **Deploy from a branch**。
3. **Branch** 选 **gh-pages**，目录选 **/ (root)**，保存。

几分钟后访问：**https://\<用户名>.github.io/Umvili/**

---

## 常见问题

- **打开是 404**  
  - 确认地址带结尾斜杠：`https://用户名.github.io/Umvili/`  
  - 确认 Pages 源是 **GitHub Actions**（方式一）或 **gh-pages** 分支（方式二）。

- **页面空白或资源 404**  
  - 项目站必须放在子路径 `/Umvili/`，不要改成根路径发布，否则资源路径会错。

- **想用自定义域名**  
  - 在 **Settings → Pages** 里填 **Custom domain**，并按提示在 DNS 加 CNAME 或 A 记录。

- **Actions 报错 “Resource not accessible by integration”**  
  - 到 **Settings → Actions → General**，确认 **Workflow permissions** 为 **Read and write**，保存后再跑一次 Actions。
