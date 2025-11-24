# 提交作业2到 GitHub 的步骤

## 仓库地址
https://github.com/173787247/model-retrain-f1

## 执行步骤

### 1. 打开 PowerShell 或 CMD，进入项目目录
```powershell
cd c:\Users\rchua\Desktop\AIFullStackDevelopment\model-retrain-f1
```

### 2. 初始化 Git 仓库（如果还没有）
```powershell
git init
```

### 3. 添加所有文件
```powershell
git add .
```

### 4. 提交更改
```powershell
git commit -m "完成模型再训练作业 - F1 Score 对比分析"
```

### 5. 添加远程仓库
```powershell
git remote add origin https://github.com/173787247/model-retrain-f1.git
```

### 6. 设置主分支为 main
```powershell
git branch -M main
```

### 7. 推送到 GitHub
```powershell
git push -u origin main
```

## 注意事项

- 如果第5步提示 "remote origin already exists"，先执行：
  ```powershell
  git remote remove origin
  ```
  然后再执行第5步

- 如果推送时要求输入用户名和密码，使用 GitHub Personal Access Token 作为密码

- 如果遇到网络问题，可以尝试多次执行 `git push -u origin main`

