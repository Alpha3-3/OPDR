#!/bin/bash
# Quick script to re-run all experiments with correct ground truth
# Usage:
#   ./quick_rerun.sh                    # Run all (with prompts)
#   ./quick_rerun.sh --no-prompts       # Run all (no prompts, for remote)
#   ./quick_rerun.sh --scalability      # Run only scalability
#   ./quick_rerun.sh --large-datasets   # Run only large datasets
#   ./quick_rerun.sh --ablation         # Run only ablation studies

set -e  # Exit on error

echo "========================================"
echo "  EXPERIMENT RE-RUN - QUICK START"
echo "========================================"
echo ""
echo "This will re-run experiments with CORRECT ground truth (from original space)."
echo "Previous results (with wrong ground truth) are INVALID."
echo ""

# Check if in Approximate directory
if [ ! -f "rerun_all_experiments.py" ]; then
    echo "[ERROR] Please run this script from the Approximate/ directory"
    exit 1
fi

# Set optimal environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUMBA_THREADING_LAYER=omp
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "[ENV] Optimized environment variables set"
echo "  OMP_NUM_THREADS=1 (BLAS single-threaded to avoid oversubscription)"
echo "  NUMBA_THREADING_LAYER=omp"
echo ""

# Run the Python script with all arguments passed through
echo "[RUN] Starting: python rerun_all_experiments.py $@"
echo ""

python rerun_all_experiments.py "$@"

echo ""
echo "========================================"
echo "  EXPERIMENT RE-RUN COMPLETE"
echo "========================================"
echo ""
echo "Results saved to: Result/"
echo "Summary: Result/RERUN_SUMMARY.txt"
echo ""

