from model import Transformer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import numpy as np
import matplotlib.pyplot as plt

def get_predictions(start, end):
    out= model(X_test[start:end])
    predictions = np.argmax(out, axis=-1)

    return predictions

if __name__  == "__main__":
    CLASS_LABELS = np.array(
    [
        "Stand",
        "Sit",
        "Talk-sit",
        "Talk-stand",
        "Stand-sit",
        "Lay",
        "Lay-stand",
        "Pick",
        "Jump",
        "Push-up",
        "Sit-up",
        "Walk",
        "Walk-backward",
        "Walk-circle",
        "Run",
        "Stair-up",
        "Stair-down",
        "Table-tennis"
    ]
    )

    # load dataset
    f = np.load('./new_dataset.npz')
    signals = f['signals']
    labels = f['labels']

    # split to train-test
    X_train, X_test, y_train, y_test = train_test_split(
        signals, labels, test_size=0.15, random_state=9, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=9, stratify=y_train
    )

    model = Transformer(
    num_layers=3,
    embed_dim=128,
    mlp_dim=256,
    num_heads=6,
    num_classes=18,
    dropout_rate=0.0,
    attention_dropout_rate=0.0,
    )

    out = model(tf.expand_dims(X_test[20], axis=0))
    print(out.shape)
    model.load_weights("model-best.h5")
    batch_size = 256  # set it by your GPU size

    full_predictions = []
    for i in range(X_test.shape[0] // batch_size):
        y = get_predictions(i * batch_size, (i + 1) * batch_size)
        full_predictions.append(y)

    y = get_predictions((i + 1) * batch_size, X_test.shape[0])
    full_predictions.append(y)

    full_predictions = np.concatenate(full_predictions, axis=0)
    print(full_predictions.shape)

    full_predictions = full_predictions.reshape(-1, full_predictions.shape[-1])
    print(full_predictions.shape)

    fig, ax = plt.subplots(figsize=(15, 15))
    cm = confusion_matrix(
        CLASS_LABELS[y_test.reshape(-1)],
        CLASS_LABELS[full_predictions.reshape(-1)],
        labels=CLASS_LABELS,
    )
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=CLASS_LABELS
    ).plot(cmap="Blues", xticks_rotation=70, ax=ax)

    print(
        classification_report(
            CLASS_LABELS[y_test.reshape(-1)],
            CLASS_LABELS[full_predictions.reshape(-1)],
            labels=CLASS_LABELS,
            digits=3,
        )
    )