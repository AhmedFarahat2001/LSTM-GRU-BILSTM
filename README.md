# Sentiment Analysis with LSTM, GRU, and Bidirectional LSTM

A deep learning project for 3-class sentiment classification (Negative / Neutral / Positive) on social media comments, comparing three recurrent neural network architectures.

---

## Dataset

- **Source:** [Sentiment Analysis Dataset](https://www.kaggle.com/datasets/abdelmalekeladjelet/sentiment-analysis-dataset) via Kaggle
- **File:** `sentiment_data.csv`
- **Size:** 241,145 records (217 dropped due to null comments → 240,928 used)
- **Columns:** `Comment` (text), `Sentiment` (0 = Negative, 1 = Neutral, 2 = Positive)

---

## Project Structure

```
├── LSTM___GRU___BILSTM.ipynb   # Main notebook
└── README.md
```

---

## Requirements

```bash
pip install tensorflow keras scikit-learn pandas numpy matplotlib seaborn kagglehub
```

---

## Pipeline

### 1. Data Loading & Exploration
- Download dataset via `kagglehub`
- Inspect shape, null values, and class distribution
- Visualize sentiment class counts with `seaborn`

### 2. Text Preprocessing
- Remove URLs (`http`, `www`)
- Remove mentions (`@username`)
- Normalize whitespace

```python
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

### 3. Tokenization
- Keras `Tokenizer` with `vocab_size=50,000` and `oov_token="<OOV>"`
- Sequences padded/truncated to `maxlen=50` (`padding='post'`, `truncating='post'`)
- Input shape chosen from sequence length distribution (95th percentile ~50 tokens)

### 4. Train / Test Split
- 80% train / 20% test (`random_state=100`)
- Training set: **(192,742 × 50)**, Test set: **(48,186 × 50)**

### 5. Class Imbalance Handling
- `compute_class_weight('balanced')` applied during training

```
Class weights: {0: 1.46, 1: 0.97, 2: 0.78}
```

### 6. Callbacks (shared across all models)
```python
def get_callbacks():
    return [
        EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, min_lr=1e-6)
    ]
```
> ⚠️ `get_callbacks()` is called as a function for each model to avoid stale callback state between runs.

---

## Models

All three models share the same structure:

```
Embedding(vocab_size+1, 64)
→ SpatialDropout1D(0.2)
→ [Recurrent layers]
→ Dense(64, relu)
→ Dropout(0.3)
→ Dense(3, softmax)
```

| Model | Recurrent Layers |
|-------|-----------------|
| LSTM | LSTM(128) → LSTM(64) |
| GRU | GRU(128) → GRU(64) |
| BiLSTM | BiLSTM(128) → BiLSTM(64) |

**Compiled with:**
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam(lr=1e-3)`
- Metric: `accuracy`
- Epochs: 25, Batch size: 64, Validation split: 0.2

---

## Results

| Model | Accuracy | Negative F1 | Neutral F1 | Positive F1 |
|-------|----------|-------------|------------|-------------|
| LSTM | **84%** | 0.78 | 0.84 | 0.87 |
| GRU | **84%** | 0.79 | 0.84 | ~0.87 |
| BiLSTM | **84–85%** | 0.80 | 0.84 | 0.87 |

BiLSTM achieves the best Negative class recall (0.82) by reading sequences in both directions.

---

## Evaluation

Each model is evaluated with:
- `classification_report` (precision, recall, F1 per class)
- Confusion matrix (raw counts)
- Normalized confusion matrix (per-class recall rates)

---

## Notes

- `tf.keras.backend.clear_session()` is called before the GRU model to reset GPU memory state
- GRU is sensitive to learning rate; if training collapses (predicts only one class), reduce `lr` to `1e-4`
- `input_length` argument in `Embedding` is deprecated in Keras — can be safely removed
