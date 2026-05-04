from model import build_model
import json
import matplotlib.pyplot as plt

def train_model(X_train, y_train, X_test, y_test):
    model = build_model(X_train.shape[1:])

    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=64,
        validation_data=(X_test, y_test)
    )

    model.save("outputs/model.h5")

    # Save training plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.savefig("outputs/training_plot.png")
    plt.close()

    return model