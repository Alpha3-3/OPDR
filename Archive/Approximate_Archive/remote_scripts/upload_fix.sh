#!/bin/bash
# Upload fixed files to remote server

echo "Uploading fixed files to remote server..."

scp main_program_optimized.py jiuzhou@er074.utah.cloudlab.us:~/Approximate/
scp mpad_optimized.py jiuzhou@er074.utah.cloudlab.us:~/Approximate/
scp scalability_test_optimized.py jiuzhou@er074.utah.cloudlab.us:~/Approximate/
scp ablation_study_optimized.py jiuzhou@er074.utah.cloudlab.us:~/Approximate/

echo "Upload complete! Now run on remote server:"
echo "  ssh jiuzhou@er074.utah.cloudlab.us"
echo "  cd ~/Approximate"
echo "  python3 scalability_test_optimized.py"

