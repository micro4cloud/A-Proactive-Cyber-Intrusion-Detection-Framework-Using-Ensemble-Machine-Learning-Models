from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM,
    Dense, Dropout, Flatten, Attention
)

def build_model(input_shape):
    inputs = Input(shape=input_shape)

    # CNN
    x = Conv1D(64, 3, activation='relu')(inputs)
    x = MaxPooling1D(2)(x)

    # LSTM
    x = LSTM(64, return_sequences=True)(x)

    # Attention
    attention = Attention()([x, x])
    x = Flatten()(attention)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model