#!/bin/bash

# Script to prepare and push code to GitHub repository
# Usage: bash prepare_push.sh

set -e

echo "🚀 Preparing repository for push to GitHub..."

# Update remote URL
echo "📡 Updating remote URL..."
git remote set-url origin https://github.com/Areeb2735/CTscan_prognosis_VLM.git

# Verify remote
echo "✅ Current remote:"
git remote -v

# Add all necessary files
echo "📦 Staging files..."
git add .gitignore
git add README.md
git add requirements.txt

# Add core code files
git add CT-CLIP/
git add dataset/
git add processed_sampels.py
git add utils.py
git add selfattention.py
git add unetr.py
git add vit.py

# Add docs (excluding model weights which are in .gitignore)
git add docs/TNM_hector_prompts.csv

# Show what will be committed
echo ""
echo "📋 Files to be committed:"
git status --short

echo ""
read -p "Do you want to commit these changes? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    # Commit changes
    echo "💾 Committing changes..."
    git commit -m "Add MAFM³ implementation: Modular Adaptation of Foundation Models for Multi-Modal Medical AI

- Add prognosis module with MTLR survival prediction
- Add segmentation module with UNETR decoder
- Update README with comprehensive documentation
- Clean up requirements.txt
- Add comprehensive .gitignore"
    
    echo ""
    echo "✅ Commit successful!"
    echo ""
    echo "📤 To push to GitHub, run:"
    echo "   git push -u origin main"
    echo ""
    echo "Or if you want to push now, run:"
    read -p "Push to GitHub now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        git push -u origin main
        echo "✅ Push successful!"
    fi
else
    echo "❌ Commit cancelled."
fi

