#!/bin/bash

# setup aws cli
echo "setting up aws cli"
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
rm -rf awscliv2.zip aws
aws --version
echo -e "aws cli setup complete\n-----"

# download data from s3
echo "downloading from s3"
FOLDERS=("checkpoints" "data")
for folder in ${FOLDERS[@]}; do
    SRC="s3://$S3_BUCKET/$folder/"
    DEST="./.cache/$folder/"
    mkdir -p "$DEST"
    aws s3 sync "$SRC" "$DEST"
done
echo -e "download complete\n-----"

# Exit immediately if any future command fails
set -e

# Set Git identity
echo "setting up git config"
git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"
echo -e "git config complete\n-----"

# install python reqs
echo "installing requirements\n-----"
pip install -r requirements.txt
echo -e "-----\nrequirements installed\n-----"

# set default shell working dir to mk-gpt repo
echo "setting default shell working dir"
echo 'cd /workspace/mk-gpt' >> ~/.bashrc
source ~/.bashrc
echo -e "default shell working dir set\n-----"