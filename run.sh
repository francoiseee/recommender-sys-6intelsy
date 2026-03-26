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
echo "[4/5] Evaluating and saving results..."
python src/eval.py --all

# Step 5: Run ablations
echo ""
echo "[5/7] Running ablation experiments..."
python src/run_ablations.py --config experiments/configs/cnn_ranker.yaml

# Step 6: Slice/error analysis
echo ""
echo "[6/7] Generating slice/error analysis outputs..."
python src/analyze_slices.py

# Step 7: Generate final report and defense assets
echo ""
echo "[7/7] Generating final report and defense assets..."
python src/generate_assets.py

echo ""
echo "============================================"
echo " Done! Check experiments/results/ for output"
echo "============================================"
