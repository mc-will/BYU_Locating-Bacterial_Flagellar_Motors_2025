import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from src.params import *
from src.ml_logic.model import train_model
from src.ml_logic.registry import save_model, save_results
from src.ml_logic.registry import mlflow_run, mlflow_transition_model



@mlflow_run
def train(model,
        X_train,
        y_train,
        model_type,
        preprocess_type,
        model_name,
        validation_data=None, # overrides validation_split
        validation_split=0.3,
        batch_size = 256,
        patience = 2
    ) -> float:

    print(Fore.GREEN + "\n🏋️ Starting model training ..." + Style.RESET_ALL)

    model, history = train_model(
        model, X_train, y_train,
        validation_data=validation_data,
        validation_split=validation_split,
        batch_size=batch_size,
        patience=patience
    )

    val_mse = np.min(history.history['val_mse'])

    params = dict(
        model_type=model_type,
        preprocess_type=preprocess_type
        # rajouter le hash du commit
        # checker le cours pour d'autres paramètres à intégrer
    )

    save_results(params=params, metrics=dict(mse=val_mse))

    save_model(model_name, model=model)

    mlflow_transition_model("None", "staging", model_name=model_name)

    print("✅ train() done \n")
