#!/bin/bash

echo "======================================"
echo "CodeAct with Feedback Training Pipeline"
echo "======================================"
echo ""

# Check conda
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Please activate conda environment first:"
    echo "  conda activate codeact"
    exit 1
fi

echo "✓ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1
echo "Step 1: Creating Enhanced Dataset..."
python 1_create_dataset_with_feedback.py
if [ $? -ne 0 ]; then exit 1; fi
echo ""

# Step 2
echo "Step 2: Fine-tuning with Feedback..."
python 2_finetune_with_feedback.py
if [ $? -ne 0 ]; then exit 1; fi
echo ""

# Step 3
echo "Step 3: Testing Model..."
python 3_test_feedback_model.py
if [ $? -ne 0 ]; then exit 1; fi
echo ""

# Step 4
echo "Step 4: Interactive Demo..."
python 4_interactive_feedback_demo.py

echo ""
echo "======================================"
echo "Pipeline Complete!"
echo "======================================"