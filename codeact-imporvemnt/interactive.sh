#!/bin/bash
# Interactive MLX Demo Runner

cd "$(dirname "$0")"

# Activate the environment
source ./codeactImprovement/bin/activate 2>/dev/null || conda activate ./codeactImprovement/ 2>/dev/null

echo "Starting CodeAct Interactive Demo with MLX..."
python 4_interactive_feedback_demo.py
