#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${1:-$ROOT_DIR/configs/hidden_dim_grid.txt}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiment_logs}"
COMMAND_FILE="$LOG_DIR/experiment_commands.txt"

mkdir -p "$LOG_DIR"
mkdir -p "$ROOT_DIR/results"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Hidden-dimension config not found: $CONFIG_FILE" >&2
    exit 1
fi

datasets=(
    "aibl"
    "adni"
    "oasis"
)

view_combos=(
    "1"
    "2"
    "3"
    "1 2"
    "1 3"
    "2 3"
    "1 2 3"
)

# Note:
# The current codebase maps labels as 0,1,2 in src/data_utils.py.
# If you really want 1,2 / 1,3 / 2,3, keep this as-is.
# If you want the currently supported zero-based pairs, replace with:
#   "0 1" "0 2" "1 2"
class_combos=(
    "0 1"
    "0 2"
    "1 2"
)

run_count=0
: > "$COMMAND_FILE"

while read -r h1 h2 h3; do
    [[ -z "${h1:-}" ]] && continue

    for dataset in "${datasets[@]}"; do
        for views in "${view_combos[@]}"; do
            for classes in "${class_combos[@]}"; do
                view_slug="${views// /_}"
                class_slug="${classes// /_}"
                run_name="${dataset}_views_${view_slug}_classes_${class_slug}_hd_${h1}_${h2}_${h3}"
                log_path="$LOG_DIR/${run_name}.txt"

                printf 'python %q --dataset %q --epochs %q --batch-size %q --hidden-dim %q %q %q --view-list %s --class-ids %s > %q 2>&1\n' \
                    "$ROOT_DIR/train.py" \
                    "$dataset" \
                    "$EPOCHS" \
                    "$BATCH_SIZE" \
                    "$h1" "$h2" "$h3" \
                    "$views" \
                    "$classes" \
                    "$log_path" >> "$COMMAND_FILE"

                run_count=$((run_count + 1))
            done
        done
    done
done < "$CONFIG_FILE"

echo "Prepared $run_count experiments"
echo "Command list: $COMMAND_FILE"
echo "Logs directory: $LOG_DIR"
echo "Running sequentially from $COMMAND_FILE"

current_run=0
failed_runs=0
while IFS= read -r cmd; do
    [[ -z "$cmd" ]] && continue
    current_run=$((current_run + 1))
    echo "[$current_run/$run_count] $cmd"

    if bash -lc "$cmd"; then
        echo "[$current_run/$run_count] completed"
    else
        exit_code=$?
        failed_runs=$((failed_runs + 1))
        echo "[$current_run/$run_count] failed with exit code $exit_code; skipping to next experiment" >&2
    fi
done < "$COMMAND_FILE"

successful_runs=$((run_count - failed_runs))
echo "Finished experiments: $successful_runs succeeded, $failed_runs failed"
