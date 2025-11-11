# Instructions for Pushing to GitHub

## ✅ What I've Done

1. **Updated `.gitignore`**: 
   - Excludes model weights (`.pth`, `.pt` files)
   - Excludes data files (`.nii.gz`, `.npz`, `.npy`)
   - Excludes cache files (`__pycache__/`, `.ipynb_checkpoints`)
   - Excludes logs and outputs (`wandb/`, `*.log`, `*.png`)
   - Excludes large CSV data files
   - Keeps only essential code and documentation

2. **Updated `README.md`**:
   - Comprehensive documentation with clear sections
   - Installation instructions
   - Usage examples for both modules
   - Methodology explanation
   - Project structure overview

3. **Cleaned `requirements.txt`**:
   - Removed commented lines
   - Kept only essential dependencies
   - Added clear comments for CT-CLIP installation

## 🚀 How to Push

### Option 1: Use the Automated Script (Recommended)

```bash
cd /share/sda/mohammadqazi/project/CTscan_prognosis_VLM-main
bash prepare_push.sh
```

This script will:
- Update the remote URL to the correct repository
- Stage all necessary files
- Show you what will be committed
- Ask for confirmation before committing
- Optionally push to GitHub

### Option 2: Manual Steps

1. **Update remote URL:**
```bash
git remote set-url origin https://github.com/Areeb2735/CTscan_prognosis_VLM.git
```

2. **Stage files:**
```bash
git add .gitignore
git add README.md
git add requirements.txt
git add CT-CLIP/
git add dataset/
git add processed_sampels.py
git add utils.py
git add selfattention.py
git add unetr.py
git add vit.py
git add docs/TNM_hector_prompts.csv
```

3. **Check what will be committed:**
```bash
git status
```

4. **Commit:**
```bash
git commit -m "Add MAFM³ implementation: Modular Adaptation of Foundation Models for Multi-Modal Medical AI

- Add prognosis module with MTLR survival prediction
- Add segmentation module with UNETR decoder
- Update README with comprehensive documentation
- Clean up requirements.txt
- Add comprehensive .gitignore"
```

5. **Push:**
```bash
git push -u origin main
```

## 📝 Files Excluded (by .gitignore)

The following files will **NOT** be pushed:
- Model weights: `*.pth`, `*.pt`, `docs/CT-CLIP_v2.pt`
- Data files: `*.nii.gz`, `*.npz`, `*.npy`
- Large CSV files: `final_hector*.csv`, `Patient_Data*.csv`
- Cache: `__pycache__/`, `.ipynb_checkpoints`
- Logs: `wandb/`, `*.log`
- Images: `*.png`, `*.jpg` (except in docs/)
- Checkpoints: `docs/weights_*/`

## ⚠️ Important Notes

1. **Model Weights**: The pre-trained CT-CLIP model (`CT-CLIP_v2.pt`) is excluded. Users need to download it separately from the CT-CLIP repository.

2. **Data Files**: Large data files are excluded. Only sample/example files like `TNM_hector_prompts.csv` are included.

3. **First Push**: If this is the first push to the repository, you might need to use:
   ```bash
   git push -u origin main --force
   ```
   (Use `--force` only if you're sure you want to overwrite the remote)

4. **Authentication**: Make sure you have GitHub authentication set up (SSH keys or personal access token).

## 🔍 Verify Before Pushing

Check what will be pushed:
```bash
git status
git diff --cached  # See staged changes
```

## 📦 Repository Size

After pushing, the repository should be relatively small (mostly code) since:
- Model weights are excluded (~GBs)
- Data files are excluded (~GBs)
- Only code, configs, and documentation are included

