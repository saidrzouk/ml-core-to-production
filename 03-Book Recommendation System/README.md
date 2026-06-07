# Book Recommendation System

Collaborative filtering recommender trained on the Book-Crossing dataset and served with Streamlit.

## Structure

- `src/app.py` - Streamlit app.
- `src/recommender.py` - data loading, filtering, model training helpers, and recommendation logic.
- `src/train.py` - reproducible training entry point.
- `src/config.py` - environment-configurable data and artifact paths.
- `data/` - local Book-Crossing CSV files.
- `artifacts/` - saved recommendation artifact.
- `notebooks/` - exploratory notebook.
- `requirements.txt` - Python dependencies.
- `README.md` - project overview and usage.

## Configuration

Override default dataset and artifact locations with environment variables:

- `BOOKS_DATA_DIR` — directory containing `BX_Books.csv`, `BX-Users.csv`, and `BX-Book-Ratings.csv`.
- `BOOKS_ARTIFACT_DIR` — directory for artifact output.
- `BOOKS_ARTIFACT_PATH` — path to the saved recommendation artifact.

If unset, defaults use the local `data/` and `artifacts/` folders.

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
