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
        monitor="val_euclidean_loss",
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


    print(f"✅ Model trained with min val Eucledian Loss: {round(np.min(history.history['val_euclidean_loss']), 2)}")



    return model, history


def evaluate_model(
        model: Model,
        #X: np.ndarray,
        #y: np.ndarray,
        test_ds: tf.data.Dataset,
        batch_size=32
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    print(model)

    metrics = model.evaluate(
        test_ds,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["euclidean_loss"]
    mse = metrics["euclidean_loss"]

    print(f"✅ Model evaluated, euclidean_loss: {round(mse, 2)}")

    return metrics
