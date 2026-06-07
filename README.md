# ML Core to Production

Four machine-learning projects, each moving from notebook exploration to a reproducible training script and a Streamlit app.

## Projects

| # | Project | Type | App |
|---|---|---|---|
| 01 | Customer Churn Prediction | Classification | `01- Customer Churn Prediction/src/app.py` |
| 02 | Hourly Energy Consumption | Time-series forecasting | `02- Hourly Energy Consumption/src/app.py` |
| 03 | Book Recommendation System | Recommender system | `03-Book Recommendation System/src/app.py` |
| 04 | Flowers Classification | Image classification | `04-Flowers Classification/src/app.py` |

## Layout

- `notebooks/` for exploration
- `scripts/` for training
- `src/` for the app and shared code
- `models/`, `artifacts/`, `reports/`, or `data/` for local project assets

## Quick Start

Use a separate virtual environment for each project.

```powershell
cd "04-Flowers Classification"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run src/app.py
```

## Common Commands

```powershell
python src/train.py
streamlit run src/app.py
```

```powershell
python scripts/train_models.py
streamlit run src/app.py
```

## Notes

- Run commands from the matching project folder.
- Generated artifacts stay inside the project that created them.
- Each project manages its own dependencies through `requirements.txt`.
