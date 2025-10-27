#!/bin/bash
# 上传Approximate文件夹和数据集到远程机器

echo "=== 上传文件到远程机器 ==="

# 设置远程主机
REMOTE_HOST="jiuzhou@amd272.utah.cloudlab.us"

# 上传Approximate文件夹
echo "上传Approximate文件夹..."
rsync -avz --progress Approximate/ $REMOTE_HOST:~/Approximate/

# 上传数据文件（从项目根目录）
echo "上传训练数据文件..."
rsync -avz --progress ../training_vectors_*.npy $REMOTE_HOST:~/

echo "上传测试数据文件..."
rsync -avz --progress ../testing_vectors_*.npy $REMOTE_HOST:~/

echo "=== 上传完成 ==="
