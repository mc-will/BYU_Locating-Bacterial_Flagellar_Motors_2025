import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()
import tensorflow as tf
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

def train_model(
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        batch_size=32,
        patience=10,
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained with min val euclidean loss: {round(np.min(history.history['val_euclidean_loss']), 2)}")

    return model, history
