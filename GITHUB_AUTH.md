# GitHub Authentication Guide

## Quick Setup (Recommended: Personal Access Token)

### Step 1: Create a Personal Access Token on GitHub

1. Go to GitHub: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Give it a name: `CTscan_prognosis_VLM_push`
4. Select scopes:
   - ✅ **repo** (Full control of private repositories)
5. Click **"Generate token"**
6. **IMPORTANT**: Copy the token immediately (you won't see it again!)
   - It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 2: Push with Token

Run this command and when prompted:
- **Username**: `Areeb2735`
- **Password**: Paste your token (not your GitHub password!)

```bash
cd /share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main
git push -u origin main
```

The credentials will be saved for future use.

---

## Alternative: SSH Key Setup

If you prefer SSH:

### Step 1: Generate SSH Key
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location
# Optionally set a passphrase
```

### Step 2: Add SSH Key to GitHub
```bash
cat ~/.ssh/id_ed25519.pub
# Copy the output
```

1. Go to: https://github.com/settings/keys
2. Click **"New SSH key"**
3. Paste your public key
4. Click **"Add SSH key"**

### Step 3: Update Remote and Push
```bash
cd /share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main
git remote set-url origin git@github.com:Areeb2735/CTscan_prognosis_VLM.git
git push -u origin main
```

---

## Current Status

✅ Code is committed locally (commit: `cc316fd`)
✅ Remote is configured correctly
⏳ Waiting for authentication to push

