import json
import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

# Offline mode avoids accidental uploads during development
wandb.init(mode="offline")

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]

# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    """
    Main pipeline orchestrator.
    Reads configuration via Hydra, runs selected steps, and passes parameters
    to each MLflow project step as defined in the config file.
    """

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/basic_cleaning",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "input_artifact": config["basic_cleaning"]["input_artifact"],
                    "output_artifact": config["basic_cleaning"]["output_artifact"],
                    "output_type": config["basic_cleaning"]["output_type"],
                    "output_description": config["basic_cleaning"]["output_description"],
                    "min_price": config["basic_cleaning"]["min_price"],
                    "max_price": config["basic_cleaning"]["max_price"]
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/data_check",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                },
            )

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            ##################
            # Implement here #
            ##################
            pass

        if "test_regression_model" in active_steps:
            ##################
            # Implement here #
            ##################
            pass


if __name__ == "__main__":
    go()
