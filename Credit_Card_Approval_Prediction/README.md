# Credit Card Approval Prediction

Machine learning web app that predicts credit card approval based on applicant data.

## Stack
- Python, Flask
- scikit-learn (RandomForestClassifier)
- HTML/CSS/JS frontend

## Setup
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open http://localhost:5000

## Structure
- `app.py` — Flask server
- `train_model.py` — trains and saves model + scaler
- `predict.py` — CLI prediction
- `dataset/` — Kaggle credit approval CSVs
- `templates/`, `static/` — frontend
- `reports/` — project report, presentation, ER diagram
