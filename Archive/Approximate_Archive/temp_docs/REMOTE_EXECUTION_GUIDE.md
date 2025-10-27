# 远程执行指南

## 1. 环境准备（SSH连接到远程机后）

```bash
# SSH连接到远程机
ssh jiuzhou@amd272.utah.cloudlab.us

# 检查Python版本（需要Python 3.8+）
python3 --version

# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate

# 安装必要的包
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm
pip install faiss-cpu  # 或 faiss-gpu（如果有GPU）
pip install tensorflow
pip install umap-learn
pip install psutil  # 用于内存监控
```

## 2. 上传文件到远程机

在本地机器上执行：

```bash
# 方法1：使用scp（从本地到远程）
cd "D:\My notes\UW\HPDIC Lab\MPAD\PCA vs DW_PMAD\PCA vs DW_PMAD"

# 上传整个Approximate文件夹
scp -r Approximate jiuzhou@amd272.utah.cloudlab.us:~/

# 上传预处理的训练/测试数据（在项目根目录）
scp training_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
scp testing_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/

# 方法2：使用rsync（推荐，支持断点续传）
rsync -avz --progress Approximate/ jiuzhou@amd272.utah.cloudlab.us:~/Approximate/
rsync -avz --progress training_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
rsync -avz --progress testing_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
```

## 3. 在SSH会话中操作

```bash
# SSH连接到远程机
ssh jiuzhou@amd272.utah.cloudlab.us

# 进入工作目录
cd ~/Approximate

# 激活虚拟环境
source ../venv/bin/activate  # 如果venv在上级目录
# 或
source venv/bin/activate  # 如果venv在Approximate目录

# 检查数据文件是否上传成功
ls -lh training_vectors_*.npy
ls -lh testing_vectors_*.npy
```

## 4. 创建tmux会话并执行任务

```bash
# 创建并命名tmux会话
tmux new -s mpad_test

# 激活虚拟环境（在tmux session内）
source ../venv/bin/activate
cd ~/Approximate

# 方法1：执行快速测试（验证环境）
python3 test_complete.py

# 方法2：执行完整的Scalability测试（推荐）
# 这会运行1%, 5%, 10%三个Fasttext子样本
python3 scalability_test.py

# 方法3：执行主评估程序
python3 main_program.py

# 方法4：执行Ablation Study（会很耗时）
python3 ablation_study.py
```

### tmux常用操作

```bash
# 分离会话（会话继续运行）
Ctrl+B, 然后按 D

# 重新连接到会话
tmux attach -t mpad_test

# 查看所有tmux会话
tmux ls

# 在tmux中查看输出
# 可以按Shift+PageUp/PageDown滚动查看历史
```

## 5. 查看结果

```bash
# 在SSH会话中或tmux会话中查看
ls -lh Result/scalability_fasttext/
ls -lh Result/

# 查看最新的结果文件
ls -t Result/scalability_fasttext/*.csv | head -5
```

## 6. 下载结果到本地

在本地机器上执行：

```bash
# 下载Scalability结果
scp jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result/scalability_fasttext/*.csv ./

# 下载所有结果（包括cache）
scp -r jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result ./

# 下载所有CSV文件
scp jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result/**/*.csv ./results/

# 使用rsync同步整个Result目录
rsync -avz jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result/ ./Result/
```

## 7. 监控任务进度

```bash
# 在tmux会话中
tail -f nohup.out  # 如果使用了nohup

# 或者在另一个SSH会话中
ssh jiuzhou@amd272.utah.cloudlab.us
tmux attach -t mpad_test
```

## 8. 推荐的完整工作流程

### 在本地准备

```bash
# 1. 创建一个脚本用于上传
cat > upload.sh << 'EOF'
#!/bin/bash
# 上传Approximate文件夹
rsync -avz --progress Approximate/ jiuzhou@amd272.utah.cloudlab.us:~/Approximate/

# 上传数据文件
rsync -avz --progress training_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
rsync -avz --progress testing_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
EOF

chmod +x upload.sh
./upload.sh
```

### 在远程机上执行

```bash
# SSH连接
ssh jiuzhou@amd272.utah.cloudlab.us

# 创建tmux会话
tmux new -s mpad_test

# 激活环境和执行
cd ~/Approximate
source ../venv/bin/activate  # 或 source venv/bin/activate

# 执行测试（这会运行很久）
python3 scalability_test.py > scalability_output.log 2>&1 &

# 或者更安全的方式，使用nohup
nohup python3 scalability_test.py > scalability_output.log 2>&1 &

# 退出tmux（会话继续运行）
# Ctrl+B, 然后按 D
```

### 下载结果

```bash
# 在本地机器上
cat > download_results.sh << 'EOF'
#!/bin/bash
# 下载结果
rsync -avz jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result/ ./Result/
EOF

chmod +x download_results.sh
./download_results.sh
```

## 9. 完整命令序列

### 初始设置（只需执行一次）

```bash
# 在SSH连接中
ssh jiuzhou@amd272.utah.cloudlab.us
cd ~

# 创建虚拟环境（如果还没有）
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm faiss-cpu tensorflow umap-learn psutil

# 退出虚拟环境
deactivate
```

### 每次运行测试的流程

```bash
# 1. 在本地上传文件
rsync -avz --progress Approximate/ jiuzhou@amd272.utah.cloudlab.us:~/Approximate/
rsync -avz --progress training_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
rsync -avz --progress testing_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/

# 2. SSH连接
ssh jiuzhou@amd272.utah.cloudlab.us

# 3. 创建tmux会话（如果还没有）
tmux new -s mpad_test  # 如果会话已存在，用：tmux attach -t mpad_test

# 4. 在tmux中激活环境并运行
source venv/bin/activate
cd ~/Approximate
python3 scalability_test.py

# 5. 分离tmux（Ctrl+B, D）

# 6. 稍后查看进度
ssh jiuzhou@amd272.utah.cloudlab.us
tmux attach -t mpad_test

# 7. 完成后下载结果（在本地）
scp -r jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result ./
```

## 10. 常见问题

### Q: 如何检查任务是否还在运行？
```bash
tmux attach -t mpad_test  # 查看tmux会话
# 或者
ps aux | grep python  # 查看Python进程
```

### Q: 如何取消正在运行的任务？
```bash
tmux attach -t mpad_test
# 然后 Ctrl+C 中断
```

### Q: 如何查看输出日志？
```bash
# 如果重定向到文件
tail -f scalability_output.log

# 在tmux会话中
# Shift+PageUp 向上滚动查看历史
```

### Q: 磁盘空间不足？
```bash
df -h  # 查看磁盘使用
du -sh ~/Approximate  # 查看目录大小
# 可以删除旧的Result文件
rm -rf ~/Approximate/Result/cache  # 如果很大
```

## 11. 快速参考卡片

```bash
# === 上传文件 ===
rsync -avz Approximate/ jiuzhou@amd272.utah.cloudlab.us:~/Approximate/
rsync -avz training_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/
rsync -avz testing_vectors_*.npy jiuzhou@amd272.utah.cloudlab.us:~/

# === SSH连接 ===
ssh jiuzhou@amd272.utah.cloudlab.us

# === tmux操作 ===
tmux new -s mpad_test              # 创建会话
tmux attach -t mpad_test           # 连接会话
tmux ls                            # 列出所有会话
# Ctrl+B, D                        # 分离会话

# === 在远程执行 ===
source venv/bin/activate
cd ~/Approximate
python3 scalability_test.py        # 运行测试

# === 下载结果 ===
scp -r jiuzhou@amd272.utah.cloudlab.us:~/Approximate/Result ./
```

## 注意事项

1. **网络连接**：如果上传大量文件，确保网络稳定，使用rsync可以断点续传
2. **运行时间**：Scalability测试可能运行数小时，确保tmux会话稳定
3. **资源限制**：检查远程机的CPU/内存限制，避免任务被kill
4. **定期检查**：定期SSH连接查看任务进度
5. **备份结果**：及时下载结果，避免被覆盖或丢失
