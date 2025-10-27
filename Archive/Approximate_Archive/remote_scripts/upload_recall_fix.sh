#!/bin/bash
# Quick upload script for recall data extraction fix

REMOTE_USER="jiuzhou"
REMOTE_HOST="er074.utah.cloudlab.us"
REMOTE_DIR="~/Approximate"

echo "======================================================================"
echo "Uploading Recall Data Fix to Remote Server"
echo "======================================================================"

echo ""
echo "[1/3] Uploading fixed Python files..."
scp \
    main_program.py \
    scalability_test_optimized.py \
    ablation_study_optimized.py \
    run_sift_deep_optimized.py \
    rerun_all_experiments.py \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to upload Python files"
    exit 1
fi

echo ""
echo "[2/3] Uploading documentation..."
scp \
    RECALL_DATA_FIX.md \
    ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/

echo ""
echo "[3/3] Verifying upload..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${REMOTE_DIR}
    echo 'Files in Approximate directory:'
    ls -lh *.py *.md | tail -10
    echo ''
    echo 'Checking for syntax errors...'
    python3 -m py_compile main_program.py
    python3 -m py_compile scalability_test_optimized.py
    python3 -m py_compile ablation_study_optimized.py
    python3 -m py_compile run_sift_deep_optimized.py
    echo '[OK] All files uploaded and verified'
"

echo ""
echo "======================================================================"
echo "Upload Complete!"
echo "======================================================================"
echo ""
echo "Next steps on remote server:"
echo ""
echo "1. SSH into remote:"
echo "   ssh ${REMOTE_USER}@${REMOTE_HOST}"
echo ""
echo "2. Activate environment and re-run experiments:"
echo "   cd ~/Approximate"
echo "   source ~/venv_approximate/bin/activate"
echo "   tmux new -s rerun"
echo "   python rerun_all_experiments.py --all"
echo ""
echo "3. Or run individual parts:"
echo "   python rerun_all_experiments.py --scalability"
echo "   python rerun_all_experiments.py --large-datasets"
echo "   python rerun_all_experiments.py --ablation"
echo ""
echo "======================================================================"

