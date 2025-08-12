import wandb

api = wandb.Api()

entity = "reyer2-western-governors-university"
project = "nyc_airbnb"

# Get runs from the project
runs = api.runs(f"{entity}/{project}")

print(f"Runs in {entity}/{project}:")
for run in runs:
    print(f"Run ID: {run.id}")
    # List artifacts used or created in each run
    artifacts = run.logged_artifacts()
    for artifact in artifacts:
        print(f"  Artifact: {artifact.name} (Type: {artifact.type}, Version: {artifact.version})")
