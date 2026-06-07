# Book Recommendation System

Collaborative filtering recommender trained on the Book-Crossing dataset and served with Streamlit.

## Structure

- `src/app.py` - Streamlit app.
- `src/recommender.py` - data loading, filtering, model training helpers, and recommendation logic.
- `src/train.py` - reproducible training entry point.
- `data/` - local Book-Crossing CSV files.
- `artifacts/` - saved recommendation artifact.
- `notebooks/` - exploratory notebook.
- `requirements.txt` - Python dependencies.
- `README.md` - project overview and usage.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py
```

## Run

```bash
streamlit run src/app.py
```
