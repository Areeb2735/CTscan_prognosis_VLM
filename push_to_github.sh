#!/bin/bash

# Helper script to push code to GitHub
# This script will prompt you for your GitHub credentials

echo "🚀 Pushing MAFM³ code to GitHub..."
echo ""
echo "📋 Current commit:"
git log --oneline -1
echo ""
echo "📡 Remote repository:"
git remote -v
echo ""

# Check if we need authentication
echo "🔐 Authentication required..."
echo ""
echo "You have two options:"
echo ""
echo "Option 1: Personal Access Token (Recommended)"
echo "  1. Go to: https://github.com/settings/tokens"
echo "  2. Generate new token (classic) with 'repo' scope"
echo "  3. Copy the token (starts with 'ghp_...')"
echo ""
echo "Option 2: Use existing credentials"
echo "  If you've already set up credentials, we'll use them"
echo ""

read -p "Do you have a Personal Access Token ready? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "✅ Attempting to push..."
    echo "   When prompted:"
    echo "   - Username: Areeb2735"
    echo "   - Password: [paste your token]"
    echo ""
    git push -u origin main
else
    echo ""
    echo "📝 Please create a Personal Access Token first:"
    echo "   1. Visit: https://github.com/settings/tokens"
    echo "   2. Click 'Generate new token (classic)'"
    echo "   3. Select 'repo' scope"
    echo "   4. Copy the token"
    echo ""
    echo "Then run this script again or run:"
    echo "   git push -u origin main"
    echo ""
fi

