import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from params import *
import mlflow
from mlflow.tracking import MlflowClient


def save_results(params: dict, metrics: dict) -> None:
    if params is not None:
        mlflow.log_params(params)
    if metrics is not None:
        mlflow.log_metrics(metrics)
    print("✅ Results saved on mlflow")


def save_model(model_name, model: keras.Model = None) -> None:
    mlflow.tensorflow.log_model(model=model,
                                artifact_path="model",
                                registered_model_name=model_name)
    print("✅ Model saved to mlflow")

    return None


def mlflow_transition_model(current_stage: str, new_stage: str, model_name:str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=model_name, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {model_name} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=model_name,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {model_name} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None



def load_model(model_name: str, stage='STAGING') -> keras.Model:
    """
    Return a saved model from MLFLOW (by "model_name")

    Return None (but do not Raise) if no model is found
    """
    print(Fore.BLUE + f"\nLoad model [{model_name}] from MLflow..." + Style.RESET_ALL)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    try:
        model_versions = client.get_latest_versions(name=model_name, stages=[stage])
        model_uri = model_versions[0].source
        assert model_uri is not None
    except:
        print(f"\n❌ No model found with name {model_name} in stage {stage}")
        return None

    model = mlflow.tensorflow.load_model(model_uri=model_uri)

    print("✅ model loaded from mlflow")

    return model


def mlflow_run(func):
    """
    Generic function to log params and results to MLflow along with TensorFlow auto-logging

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper
