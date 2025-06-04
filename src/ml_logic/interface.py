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
        model_type: str,
        preprocess_type: str,
        model_name: str,
        validation_data=None, # overrides validation_split
        validation_split=0.3,
        batch_size = 256,
        patience = 2
    ) -> float:
    '''
    Train a compiled model and saves it with mlflow on a server. Dependant on .env file to get the uri of the server,
    and to define the experiment name. The function asks for different string parameters to ensure a good monitoring of
    models produced.

            Parameters:
                    model: an initialized and compiled model
                    X_train: pictures array
                    y_train: could be any type of y (2D/3D position, true/false, etc.)
                    model_type (str): for monitoring purpose, the type of the model: 2D/3D, position or presence
                    preprocess_type (str): for monitoring purpose, the kind of preprocessed images used (mean, equaladaptX, best_slice)
                    model_name (str): for monitoring purpose, the name of the model, ideally a combinaition of the two previous parameters
                    validation_data: validation data, not necessary if validation split is given
                    validation_split: fraction used for the validation
                    batch_size: batch size
                    patience: patience

            Example:
                If you are running a model working on 2D picture to identify the position of motors, selecting the best
                slice of the tomogram, then call the function with model_type='pos2D', preprocess_type='best_slice',
                model_name='reg on x,y - best_slice':

                `from src.ml_logic.interface import train
                train(model, X_train, y_train, 'pos2D', 'best_slice', 'reg on x,y - best_slice')`
    '''
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
