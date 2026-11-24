.PHONY: help install all pipeline train test evaluate demo presentation verify clean eda split download

PYTHON ?= uv run python

help:
	@echo "======================================================================"
	@echo "Next-Month Credit Card Default Prediction Pipeline (UCI 350)"
	@echo "======================================================================"
	@echo "Available commands:"
	@echo "  make install        - Install dependencies using uv sync"
	@echo "  make all            - Run the complete end-to-end pipeline (run.py)"
	@echo "  make pipeline-fast  - Run pipeline reusing frozen models"
	@echo "  make download       - Download and verify raw UCI dataset"
	@echo "  make split          - Create leak-free StratifiedGroupKFold partitions"
	@echo "  make eda            - Generate training-only EDA and audit plots"
	@echo "  make train          - Train classical models and PyTorch MLP (CUDA)"
	@echo "  make train-classical- Train Dummy, Logistic Regression, and Boosting"
	@echo "  make train-neural   - Train PyTorch MLP on GPU (RTX 4050 / CUDA)"
	@echo "  make freeze         - Lock validation winners and compute SHA-256 hashes"
	@echo "  make test           - Run automated unit tests"
	@echo "  make evaluate       - Evaluate on held-out test set & compute bootstrap CIs"
	@echo "  make demo           - Run interactive inference demo on test client #0"
	@echo "  make presentation   - Build 15-slide minimalist PowerPoint deck and speaker notes"
	@echo "  make verify         - Run comprehensive project integrity verifier"
	@echo "  make clean          - Remove temporary Python caches and bytecode"
	@echo "======================================================================"

install:
	uv sync

all:
	$(PYTHON) run.py

pipeline-fast:
	$(PYTHON) run.py --skip-training

download:
	$(PYTHON) -m src.data download

split:
	$(PYTHON) -m src.preprocessing prepare

eda:
	$(PYTHON) -m src.data eda

train-classical:
	$(PYTHON) -m src.classical

train-neural:
	$(PYTHON) -m src.neural train

train: train-classical train-neural

freeze:
	$(PYTHON) -m src.freeze

test:
	$(PYTHON) -m unittest discover -s tests

evaluate:
	$(PYTHON) -m src.evaluate

demo:
	$(PYTHON) -m src.demo --index 0 --reveal-label

presentation:
	$(PYTHON) scripts/build_presentation.py

verify:
	$(PYTHON) scripts/verify_project.py

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]"
