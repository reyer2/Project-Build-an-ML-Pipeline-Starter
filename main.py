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
                ".",
                "main",
                env_manager="conda",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                version="main",
                env_manager="conda",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            # Run train_random_forest locally from src/train_random_forest folder (no git repo fetch)
            _ = mlflow.run(
                "src/train_random_forest",
                entry_point="main",
                env_manager="conda",
                version=none
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                },
            )

        if "test_regression_model" in active_steps:
            ##################
            # Implement here #
            ##################
            pass


if __name__ == "__main__":
    go()

