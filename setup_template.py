import os

project_name = "heart_disease"

folders = [
    "data/raw",
    "data/processed",
    "data/artifacts",
    "notebooks",
    "src",
    "api",
    "models"
]

files = [
    "src/config.py",
    "src/data_loader.py",
    "src/feature_engineering.py",
    "src/preprocessing.py",
    "src/models.py",
    "src/train.py",
    "src/evaluate.py",
    "src/tune.py",
    "src/ensemble.py",
    "src/predict.py",
    "src/utils.py",
    "api/app.py",
    "main.py",
    "requirements.txt",
    "README.md"
]

for folder in folders:
    os.makedirs(os.path.join(project_name, folder), exist_ok=True)

for file in files:
    path = os.path.join(project_name, file)
    with open(path, "w") as f:
        f.write("")

print("Project structure created successfully.")
