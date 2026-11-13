# Build step, run before `docker build`: downloads the registered champion model into model/, so the image holds
# a fixed model version and the API never talks to MLflow. Needs mlflow, which the API's image doesn't have, so it
# runs with the training venv:
#   ..\give-me-some-credit\venv\Scripts\python fetch_champion.py
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
from mlflow import MlflowClient, MlflowException

HERE = Path(__file__).parent    # paths are relative to this file, so the script works from any folder
MODEL_DIR = HERE / "model"
DEFAULT_TRACKING_URI = f"sqlite:///{(HERE.parent / 'give-me-some-credit' / 'mlflow.db').as_posix()}"


def fetch_champion():
    """
    Download the champion's files into model/ and record which version they are in model/registered_model.json.
    Exits with code 1 if there is no champion, so a build that runs next doesn't package an old model or none.
    """
    mlflow_tracking_uri = os.environ.get("CREDIT_MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registered_model = os.environ.get("CREDIT_REGISTERED_MODEL", "credit-risk")
    alias = os.environ.get("CREDIT_ALIAS", "champion")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    try:
        champion_version = client.get_model_version_by_alias(registered_model, alias)
    except MlflowException:
        champion_version = None

    if champion_version is not None:
        print(f"Champion model: {registered_model}@{alias} (version {champion_version.version})")
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        MODEL_DIR.mkdir()
        # download the version looked up above, not @alias again: the alias could move in between
        mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{registered_model}/{champion_version.version}",
                                            dst_path=str(MODEL_DIR))
        # written last, so model/ has a version file only once the model files are there
        with open(MODEL_DIR / "registered_model.json", "w") as f:
            json.dump({"name": registered_model, "version": champion_version.version,
                       "run_id": champion_version.run_id}, f, indent=2)
    else:
        print(f"No champion model found for {registered_model}@{alias}")
        sys.exit(1)


if __name__ == "__main__":
    fetch_champion()
