#!/bin/bash
# run.sh — One-command full pipeline reproduce
# Usage: bash run.sh

set -e  # Exit immediately on error

echo "============================================"
echo " Personalized Recommendation System"
echo " 6INTELSY Final Project AY2025-2026"
echo "============================================"

# Step 1: Download data
echo ""
echo "[1/4] Downloading and preparing data..."
python data/get_data.py

# Step 2: Train baselines
echo ""
echo "[2/4] Training baseline models..."
python src/train.py --config experiments/configs/baseline.yaml

# Step 3: Train CNN ranker
echo ""
echo "[3/4] Training CNN text ranker + bandit simulation..."
python src/train.py --config experiments/configs/cnn_ranker.yaml
python src/rl_agent.py --config experiments/configs/bandit.yaml

# Step 4: Evaluate all models
echo ""
echo "[4/4] Evaluating and saving results..."
python src/eval.py --all

echo ""
echo "============================================"
echo " Done! Check experiments/results/ for output"
echo "============================================"
