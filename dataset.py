import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(path):
    df = pd.read_csv(path)

    df = df.dropna()
    df = df.drop_duplicates()

    label = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    encoder = LabelEncoder()
    y = encoder.fit_transform(label)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # reshape for CNN-LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test