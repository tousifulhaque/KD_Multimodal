from model import Transformer
from sklearn import train_test_split


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
