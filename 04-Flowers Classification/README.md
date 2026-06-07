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
- `src/app.py` — Streamlit image classification app.
- `models/flower_classifier_model.keras` — trained Keras model.
- `models/class_names.json` — label mapping.
- `notebooks/` — exploratory notebook.
- `data/raw/flowers/` — local image dataset.

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


