#!/bin/bash
# 在远程机器上执行测试的脚本
# 这个文件需要上传到远程机器

echo "=== 远程执行测试 ==="

# 设置变量
SESSION_NAME="mpad_test"
WORK_DIR="$HOME/Approximate"

# 检查是否在tmux会话中
if [ -z "$TMUX" ]; then
    echo "不在tmux会话中，创建新会话..."
    tmux new -s $SESSION_NAME -d
    tmux send-keys -t $SESSION_NAME "cd $WORK_DIR && source ../venv/bin/activate && python3 scalability_test.py" C-m
    echo "测试已在tmux会话 '$SESSION_NAME' 中启动"
    echo "使用以下命令查看："
    echo "  tmux attach -t $SESSION_NAME"
else
    echo "已在tmux会话中，直接运行..."
    cd $WORK_DIR
    source ../venv/bin/activate
    python3 scalability_test.py
fi
