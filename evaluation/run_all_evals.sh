#!/bin/bash

# Sequential evaluation runner script
# Each evaluation runs only after the previous one completes

set -e  # Exit immediately if a command fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Starting Sequential Evaluation Suite"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Track results
declare -a RESULTS

run_eval() {
    local script_name=$1
    local display_name=$2
    
    echo "----------------------------------------"
    echo "Running: $display_name"
    echo "Script: $script_name"
    echo "Started: $(date)"
    echo "----------------------------------------"
    
    if python3 "$script_name"; then
        echo "✓ $display_name completed successfully"
        RESULTS+=("✓ $display_name")
    else
        echo "✗ $display_name failed with exit code $?"
        RESULTS+=("✗ $display_name (FAILED)")
    fi
    echo ""
}

# Run each evaluation sequentially
run_eval "evaluate_s1k.py" "S1K Math Evaluation"
run_eval "evaluate_reclor.py" "ReClor Logical Reasoning Evaluation"
run_eval "evaluate_math_500.py" "MATH-500 Evaluation"
run_eval "evaluate_gsm8k.py" "GSM8K Evaluation"
run_eval "evaluate_aime2024.py" "AIME 2024 Evaluation"

# Summary
echo "========================================"
echo "Evaluation Suite Complete"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Execution Summary:"
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo ""

# Print consolidated results from JSON files
echo "========================================"
echo "Results Summary (from JSON files)"
echo "========================================"
printf "%-15s | %-10s | %-15s\n" "Dataset" "Accuracy" "Correct/Total"
echo "----------------|------------|----------------"

for json_file in *_eval_results.json; do
    if [ -f "$json_file" ]; then
        dataset=$(python3 -c "import json; d=json.load(open('$json_file')); print(d.get('dataset', 'Unknown'))" 2>/dev/null)
        accuracy=$(python3 -c "import json; d=json.load(open('$json_file')); print(f\"{d['accuracy']:.2%}\")" 2>/dev/null)
        correct=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['correct'])" 2>/dev/null)
        total=$(python3 -c "import json; d=json.load(open('$json_file')); print(d['total'])" 2>/dev/null)
        printf "%-15s | %-10s | %s/%s\n" "$dataset" "$accuracy" "$correct" "$total"
    fi
done

echo ""
echo "Individual JSON files saved in: $(pwd)"

