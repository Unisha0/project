# CivicConnect

CivicConnect is a complaint classification project built with DistilBERT models.
It includes:

- An English complaint pipeline in `civicconnect/`
- A Nepali complaint pipeline in `nepali/`
- Trained checkpoints, best models, and analysis outputs committed for reproducibility

This repository is now clone-ready so others can directly build frontend and backend layers on top of the existing ML artifacts.

## Project Layout

```text
.
├── civicconnect/                # English pipeline
│   ├── train_distilbert.py
│   ├── predict_distilbert.py
│   ├── evaluate_hard_test.py
│   ├── analysis/
│   ├── best_model/
│   ├── data/
│   └── results/
├── nepali/                      # Nepali pipeline
│   ├── train_distilbert.py
│   ├── predict_distilbert.py
│   ├── generate_complaints.py
│   ├── merge_datasets.py
│   ├── analysis/
│   ├── best_model/
│   ├── data/
│   └── results/
└── README.md
```

## Requirements

- Python 3.10+ recommended
- `git-lfs` required (model weights are tracked with LFS)

Check LFS tracking in `.gitattributes`:

- `*.pt`
- `*.safetensors`

## Clone And Setup

1. Clone repository:

```bash
git clone https://github.com/Unisha0/project.git
cd project
```

2. Install Git LFS and download model artifacts:

```bash
git lfs install
git lfs pull
```

3. Create virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r civicconnect/requirements.txt
```

## English Pipeline (`civicconnect/`)

Run from repository root:

1. Train model:

```bash
cd civicconnect
python train_distilbert.py
```

Outputs:

- Best model in `civicconnect/best_model/`
- Checkpoints in `civicconnect/results/`
- Evaluation arrays and plots in `civicconnect/analysis/`

2. Predict interactively:

```bash
cd civicconnect
python predict_distilbert.py
```

3. Evaluate hard test set:

```bash
cd civicconnect
python evaluate_hard_test.py
```

## Nepali Pipeline (`nepali/`)

Run from repository root:

1. (Optional) Generate synthetic complaints:

```bash
cd nepali
python generate_complaints.py
```

2. Train multilingual DistilBERT classifier:

```bash
cd nepali
python train_distilbert.py
```

Outputs:

- Best model in `nepali/best_model/`
- Checkpoints in `nepali/results/`
- Reports and plots in `nepali/analysis/`

3. Run prediction script:

```bash
cd nepali
python predict_distilbert.py
```

Quick test mode:

```bash
python predict_distilbert.py --test
```

## Model Labels

Current class labels used by both workflows:

- `electricity`
- `water`
- `road`
- `garbage`

## Frontend/Backend Integration Notes

This repo currently provides model training and local inference scripts. To extend into a full application:

1. Backend layer:
- Wrap prediction logic from `civicconnect/predict_distilbert.py` and `nepali/predict_distilbert.py` into API endpoints.
- Recommended endpoints:
	- `POST /predict/en`
	- `POST /predict/ne`
- Suggested request body:
	- `{ "text": "..." }`
- Suggested response body:
	- `{ "label": "road", "confidence": 0.93, "uncertain": false }`

2. Frontend layer:
- Build a simple form to submit complaints.
- Add language toggle (English/Nepali).
- Display top prediction + confidence.
- Handle low-confidence responses as "Uncertain" and allow manual routing.

3. Deployment:
- Keep models on server disk or object storage.
- Ensure `git lfs pull` is part of deployment setup.

## Reproducibility Notes

- Training artifacts and checkpoints are intentionally committed.
- If clone appears incomplete, run:

```bash
git lfs pull
```

- If Python package conflicts occur, recreate virtual environment and reinstall from `civicconnect/requirements.txt`.

## License

No explicit license file is currently included.
Add a `LICENSE` file before public or commercial redistribution.
