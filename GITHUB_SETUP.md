# GitHub Setup Guide

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the "+" icon in the top right corner
3. Select "New repository"
4. Repository name: `ecommerce-churn-analysis` (or your preferred name)
5. Description: "E-Commerce Churn & Burn Analysis - Data Mining Project"
6. Choose **Public** or **Private**
7. **DO NOT** initialize with README, .gitignore, or license (we already have these)
8. Click "Create repository"

## Step 2: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these commands:

### Option A: If you haven't created the repository yet
```bash
# Replace YOUR_USERNAME with your GitHub username
# Replace REPO_NAME with your repository name
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

### Option B: If you already have a repository
```bash
# Copy the repository URL from GitHub (it will look like: https://github.com/username/repo-name.git)
git remote add origin <YOUR_REPOSITORY_URL>
git branch -M main
git push -u origin main
```

## Step 3: Update Git User Configuration (Optional but Recommended)

To set your GitHub identity globally:
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Troubleshooting

### If you get authentication errors:
- GitHub now requires Personal Access Tokens (PAT) instead of passwords
- Go to: Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate a new token with `repo` scope
- Use the token as your password when pushing

### If you need to change the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

## Quick Commands Reference

```bash
# Check current remote
git remote -v

# Push to GitHub
git push origin main

# Pull from GitHub
git pull origin main

# Check status
git status

# View commit history
git log --oneline
```

