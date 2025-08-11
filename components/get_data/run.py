#!/usr/bin/env python
"""
This script downloads a URL to a local destination
"""

import os
os.environ["WANDB_DISABLE_SYSTEM_MONITOR"] = "true"

import argparse
import logging
import wandb

# Monkeypatch to fix pynvml UnicodeDecodeError in wandb GPU system monitor
try:
    import wandb.vendor.pynvml.pynvml as pynvml

    original_nvmlDeviceGetName = pynvml.nvmlDeviceGetName

    def safe_nvmlDeviceGetName(handle):
        try:
            raw_name = original_nvmlDeviceGetName(handle)
            # Try decode, fallback on ignoring errors
            if isinstance(raw_name, bytes):
                return raw_name.decode("utf-8", errors="ignore")
            return raw_name
        except Exception:
            return "Unknown GPU"

    pynvml.nvmlDeviceGetName = safe_nvmlDeviceGetName
except ImportError:
    # pynvml not available, skip monkeypatch
    pass

from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="download_file")
    run.config.update(vars(args))

    logger.info(f"Returning sample {args.sample}")
    logger.info(f"Uploading {args.artifact_name} to Weights & Biases")
    log_artifact(
        args.artifact_name,
        args.artifact_type,
        args.artifact_description,
        os.path.join("data", args.sample),
        run,
    )
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download")
    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")
    parser.add_argument("artifact_type", type=str, help="Output artifact type.")
    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()
    go(args)

