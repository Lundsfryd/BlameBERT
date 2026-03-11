#!/bin/bash

# This is an initialization script for use in UCloud, setting up Git credential manager and lfs

eval "$(/work/MarkusLundsfrydJensen#1865/miniconda3/bin/conda shell.bash hook)"
conda init
export MY_ENV=blameBERT
echo "conda activate $MY_ENV" >> ~/.bashrc


BASE_DIR="/work/MarkusLundsfrydJensen#1865"
 
# --- GitHub setup ---

cd "$BASE_DIR"


# Checking for existing git installations. No longer falls back to a new download each time
if ls amd 1> /dev/null 2>&1; then
    echo "Existing GCM installation found...configuring"
    git-credential-manager-core configure
else
    echo "No existing GCM found, downloading and installing..."
    wget https://github.com/GitCredentialManager/git-credential-manager/releases/download/v2.0.785/gcm-linux_amd64.2.0.785.deb
    sudo dpkg -i gcm-linux_amd64.2.0.785.deb
    git-credential-manager-core configure
fi

sudo apt-get update && sudo apt-get install -y git-lfs


git lfs install
cd "/work/MarkusLundsfrydJensen#1865/Bachelor_project"
git lfs pull