import argparse
from dataset import load_data
from train import train_model
from evaluate import evaluate_model
from tensorflow.keras.models import load_model

DATA_PATH = "data/intrusion.csv"

def main(mode):
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    if mode == "train":
        model = train_model(X_train, y_train, X_test, y_test)
    else:
        model = load_model("outputs/model.h5")

    metrics = evaluate_model(model, X_test, y_test)
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    main(args.mode)