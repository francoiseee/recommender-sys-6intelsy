.PHONY: repro train eval data clean lint help

## repro: Full reproduce — download data, train, evaluate, generate results
repro:
	@echo "==> Downloading data..."
	python data/get_data.py
	@echo "==> Training baselines..."
	python src/train.py --config experiments/configs/baseline.yaml
	@echo "==> Training CNN text ranker..."
	python src/train.py --config experiments/configs/cnn_ranker.yaml
	@echo "==> Running RL bandit simulation..."
	python src/rl_agent.py --config experiments/configs/bandit.yaml
	@echo "==> Evaluating all models..."
	python src/eval.py --all
	@echo "==> Done! Results saved to experiments/results/"

## data: Download and preprocess data only
data:
	python data/get_data.py

## train: Train the main model
train:
	python src/train.py --config experiments/configs/cnn_ranker.yaml

## eval: Run evaluation on trained models
eval:
	python src/eval.py --all

## lint: Run code linting
lint:
	flake8 src/ --max-line-length=120

## clean: Remove generated files and caches
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf experiments/logs/*.log

## help: Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/## //'
