# Flower Classification with Transfer Learning

Image classifier trained with an EfficientNetB0 backbone and served through a Streamlit upload interface.

## Dataset

Place the image dataset in:

```text
data/raw/flowers/
├── daisy/
├── dandelion/
├── rose/
├── sunflower/
└── tulip/
```

## Structure

- `scripts/train.py` — training and fine-tuning script.
- `src/config.py` — environment-configurable dataset and artifact paths.
- `src/app.py` — Streamlit image classification app.
- `models/flower_classifier_model.keras` — trained Keras model.
- `models/class_names.json` — label mapping.
- `notebooks/` — exploratory notebook.
- `data/raw/flowers/` — local image dataset.

## Configuration

The project supports overriding dataset and artifact file paths with environment variables:

- `FLOWERS_DATA_DIR` — image dataset root directory.
- `FLOWERS_MODEL_PATH` — output or load path for the Keras model.
- `FLOWERS_CLASS_NAMES_PATH` — path for the generated class name mapping.

If unset, defaults use `data/raw/flowers/`, `models/flower_classifier_model.keras`, and `models/class_names.json`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py
```

## Run

```bash
streamlit run src/app.py
```


