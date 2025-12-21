# GitHub Authentication Guide

## Issue: Permission Denied (403 Error)

GitHub requires a **Personal Access Token (PAT)** instead of passwords for authentication.

## Solution: Create and Use a Personal Access Token

### Step 1: Create a Personal Access Token

1. Go to GitHub.com and sign in
2. Click your profile picture → **Settings**
3. Scroll down to **Developer settings** (left sidebar)
4. Click **Personal access tokens** → **Tokens (classic)**
5. Click **Generate new token** → **Generate new token (classic)**
6. Give it a name: `Data-Mining-G13-Push`
7. Set expiration (recommend: 90 days or custom)
8. Select scopes:
   - ✅ **repo** (Full control of private repositories)
9. Click **Generate token**
10. **COPY THE TOKEN IMMEDIATELY** (you won't see it again!)

### Step 2: Use the Token to Push

#### Option A: Use Token in Command Line (One-time)
```bash
git push -u origin main
# When prompted for username: muinul-official
# When prompted for password: PASTE_YOUR_TOKEN_HERE
```

#### Option B: Configure Git Credential Helper (Recommended)
```bash
# Windows (Git Credential Manager)
git config --global credential.helper manager-core

# Then push (will prompt for token once)
git push -u origin main
```

#### Option C: Use Token in URL (Temporary)
```bash
git remote set-url origin https://YOUR_TOKEN@github.com/muinul-official/Data-Mining-G13.git
git push -u origin main
```

### Step 3: Verify Push

After successful push, check your repository:
https://github.com/muinul-official/Data-Mining-G13

## Alternative: SSH Authentication

If you prefer SSH (more secure for long-term use):

1. Generate SSH key:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

2. Add SSH key to GitHub:
   - Copy public key: `cat ~/.ssh/id_ed25519.pub`
   - GitHub → Settings → SSH and GPG keys → New SSH key

3. Change remote URL:
```bash
git remote set-url origin git@github.com:muinul-official/Data-Mining-G13.git
git push -u origin main
```

## Troubleshooting

- **403 Forbidden**: Token doesn't have correct permissions or expired
- **401 Unauthorized**: Invalid token or username
- **Repository not found**: Check repository name and access permissions

