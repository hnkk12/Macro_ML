.PHONY: all data experiment tables figures test clean

all: data experiment tables figures

data:
	python scripts/download_data.py --config configs/experiments/main.yaml
	python scripts/build_dataset.py --config configs/experiments/main.yaml

experiment:
	python scripts/run_experiment.py --config configs/experiments/main.yaml
	python scripts/run_robustness_gap.py

tables:
	python scripts/make_tables.py --config configs/experiments/main.yaml

figures:
	python scripts/make_figures.py --config configs/experiments/main.yaml

test:
	python -m pytest tests/ -v --tb=short

clean:
	python -c "import glob, os; [os.remove(f) for f in glob.glob('outputs/**/*.csv', recursive=True)]; [os.remove(f) for f in glob.glob('outputs/**/*.png', recursive=True)]"
