#!/bin/bash

echo "=============================="
echo "CodeAct Fine-tuning Pipeline"
echo "=============================="
echo ""

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "❌ Please activate conda environment first:"
    echo "   source activate codeact"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Step 1: Create dataset
echo "Step 1: Creating Dataset..."
python 1_create_dataset.py
if [ $? -ne 0 ]; then
    echo "❌ Dataset creation failed"
    exit 1
fi
echo ""

# Step 2: Fine-tune model
echo "Step 2: Fine-tuning Model..."
python 2_finetune_model.py
if [ $? -ne 0 ]; then
    echo "❌ Fine-tuning failed"
    exit 1
fi
echo ""

# Step 3: Test model
echo "Step 3: Testing Model..."
python 3_test_model.py
if [ $? -ne 0 ]; then
    echo "❌ Testing failed"
    exit 1
fi
echo ""

echo "=============================="
echo "Pipeline Complete!"
echo "=============================="
echo ""
echo "To run the interactive demo:"
echo "  ./interactive.sh"
