#!/bin/bash
# 从远程机器下载结果文件

echo "=== 从远程机器下载结果 ==="

# 设置远程主机
REMOTE_HOST="jiuzhou@amd272.utah.cloudlab.us"

# 创建本地结果目录
mkdir -p Result

# 下载所有结果
echo "下载Result目录..."
rsync -avz --progress $REMOTE_HOST:~/Approximate/Result/ Result/

echo "=== 下载完成 ==="
echo "结果保存在: Result/ 目录"
