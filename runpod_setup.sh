# Set global Git identity
echo "setting up git config"
git config --global user.email "$GIT_EMAIL"
git config --global user.name "$GIT_NAME"
echo -e "git config complete\n-----"