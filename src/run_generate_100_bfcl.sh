#!/bin/bash
# Run the generate_100_datapoints_bfcl.py script in the background

cd /home/ishalyminov/data/APIGen-MT/src

# Log file
LOG_FILE="/home/ishalyminov/data/APIGen-MT/data/generated/generate_100_bfcl.log"

echo "Starting datapoint generation at $(date)" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Output file: /home/ishalyminov/data/APIGen-MT/data/generated/apigen_phase1_100_datapoints_bfcl.jsonl" | tee -a "$LOG_FILE"
echo "---" | tee -a "$LOG_FILE"

# Run the script
python3 generate_100_datapoints_bfcl.py 2>&1 | tee -a "$LOG_FILE"

echo "---" | tee -a "$LOG_FILE"
echo "Generation completed at $(date)" | tee -a "$LOG_FILE"