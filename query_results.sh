#!/bin/bash
# Query results by job number or search pattern

LOGDIR="${1:-res-6}"

if [ ! -f "$LOGDIR/joblog.tsv" ]; then
    echo "Error: $LOGDIR/joblog.tsv not found"
    exit 1
fi

# If given a job number, show that job
if [[ "$2" =~ ^[0-9]+$ ]]; then
    job_num="$2"
    echo "=== Job $job_num ==="
    awk -v job="$job_num" 'NR==1 || $1==job' "$LOGDIR/joblog.tsv" | column -t -s $'\t'
    echo ""
    if [ -d "$LOGDIR/results/$job_num" ]; then
        echo "Results in: $LOGDIR/results/$job_num/"
        ls -lh "$LOGDIR/results/$job_num/"
    fi
# Otherwise search by pattern
elif [ -n "$2" ]; then
    pattern="$2"
    echo "=== Jobs matching '$pattern' ==="
    awk -v pat="$pattern" 'NR==1 || $0 ~ pat' "$LOGDIR/joblog.tsv" | column -t -s $'\t'
else
    # Show summary
    echo "=== Summary of $LOGDIR ==="
    echo "Total jobs: $(tail -n +2 "$LOGDIR/joblog.tsv" | wc -l)"
    echo ""
    echo "Jobs by method:"
    grep -oP 'cli.py \K\w+' "$LOGDIR/joblog.tsv" | sort | uniq -c
    echo ""
    echo "Usage:"
    echo "  $0 $LOGDIR <job_number>     # Show specific job"
    echo "  $0 $LOGDIR minimax          # Search for pattern"
    echo "  $0 $LOGDIR 'seed 5'         # Search for pattern with spaces"
fi

