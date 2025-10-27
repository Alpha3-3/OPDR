#!/bin/bash
# Quick upload script for updated scalability parameters

REMOTE_USER="jiuzhou"
REMOTE_HOST="er074.utah.cloudlab.us"
REMOTE_DIR="~/Approximate"

echo "======================================================================"
echo "Uploading Updated Scalability Script (b=4%, alpha=0.4)"
echo "======================================================================"

echo ""
echo "[1/2] Uploading scalability_test_optimized.py..."
scp \
    scalability_test_optimized.py \
    SCALABILITY_PARAMETER_UPDATE.md \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upload files"
    exit 1
fi

echo ""
echo "[2/2] Verifying upload..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${REMOTE_DIR}
    echo 'Checking scalability_test_optimized.py parameters...'
    grep -A 2 'Base parameters' scalability_test_optimized.py
    echo ''
    echo 'File modification time:'
    ls -lh scalability_test_optimized.py
    echo ''
    echo '[OK] Files uploaded successfully'
"

echo ""
echo "======================================================================"
echo "Upload Complete!"
echo "======================================================================"
echo ""
echo "Parameter Changes:"
echo "  - b_percentage: 1.0% → 4.0%"
echo "  - alpha: 0.1 → 0.4"
echo ""
echo "Expected Impact:"
echo "  - Higher Recall@k (potentially 2-3× improvement)"
echo "  - Longer runtime (~4× slower, 4-8 hours total)"
echo ""
echo "Next steps on remote server:"
echo ""
echo "1. SSH into remote:"
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "2. Run updated scalability test:"
echo "   cd ~/Approximate"
echo "   source ~/venv_approximate/bin/activate"
echo "   tmux new -s scalability_b4_a04"
echo "   python scalability_test_optimized.py"
echo ""
echo "3. Monitor progress (in another terminal):"
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo "   tmux attach -t scalability_b4_a04"
echo ""
echo "4. Expected runtime: 4-8 hours"
echo "   Check back in ~6 hours"
echo ""
echo "======================================================================"

