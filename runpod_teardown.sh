# backup data to s3
echo "backing up data to s3"
FOLDERS=("test" "checkpoints" "data")
for folder in ${FOLDERS[@]}; do
    SRC="./storage/$folder/"
    DEST="s3://$S3_BUCKET/$folder/"
    aws s3 sync "$SRC" "$DEST"
done
echo "backup complete\n-----"