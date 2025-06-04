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
        train_ds,
        val_ds,
        model_type: str,
        preprocess_type: str,
        model_name: str,
        batch_size = 32,
        patience = 10
    ) -> float:
    '''
    Train a compiled model and saves it with mlflow on a server. Dependant on .env file to get the uri of the server,
    and to define the experiment name. The function asks for different string parameters to ensure a good monitoring of
    models produced.

            Parameters:
                    model: an initialized and compiled model
                    train_ds: tf.data.Dataset object
                    val_ds: tf.data.Dataset object
                    model_type (str): for monitoring purpose, the type of the model: 2D/3D, position or presence
                    preprocess_type (str): for monitoring purpose, the kind of preprocessed images used (mean, equaladaptX, best_slice)
                    model_name (str): for monitoring purpose, the name of the model, ideally a combinaition of the two previous parameters
                    batch_size: batch size
                    patience: patience

            Example:
                If you are running a model working on 2D picture to identify the position of motors, selecting the best
                slice of the tomogram, then call the function with model_type='pos2D', preprocess_type='best_slice',
                model_name='reg on x,y - best_slice':

                `from src.ml_logic.interface import train
                train(model, train_ds, val_ds, 'pos2D', 'best_slice', 'reg on x,y - best_slice')`
    '''
    print(Fore.GREEN + "\n🏋️ Starting model training ..." + Style.RESET_ALL)

    model, history = train_model(
        model, train_ds,
        val_ds,
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
