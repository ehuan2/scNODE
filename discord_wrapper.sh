#!/bin/bash

# Check that a command was given
if [ $# -eq 0 ]; then
    echo "Usage: $0 \"<command to run>\""
    exit 1
fi

# Capture the command string
CMD="$*"

# Optional: create logs directory
mkdir -p logs

# Generate timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/nohup_${timestamp}.out"

echo ">>> Running with Discord wrapper:"
echo "nohup bash ../discord.sh $CMD &"
echo "Logs: $log_file"

# Run the command inside nohup + discord.sh
nohup bash ../discord.sh bash "$CMD" > "$log_file" 2>&1 &
