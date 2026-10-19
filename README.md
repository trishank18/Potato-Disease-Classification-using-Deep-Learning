# Potato Leaf Disease Detection Using CNN and Flask

An educational image-classification application that identifies potato leaves as **Early Blight**, **Late Blight**, or **Healthy**. The trained TensorFlow/Keras model is exposed through a Flask web interface where users can upload a leaf image and view the predicted class and scores.

> **Important:** This is a closed-set classifier. It always chooses one of the three known classes, even when an uploaded image is not a potato leaf. Scores are model outputs, not guaranteed probabilities or agricultural advice.

## Features

- Reproducible, class-stratified 60/20/20 train/test/validation split.
- RGB image preprocessing at 128 × 128 pixels.
- Custom CNN, weighted CNN, and MobileNetV2 training configurations.
- Model selection using validation metrics and final evaluation on a held-out test set.
- Accuracy, precision, recall, F1-score, classification report, and confusion-matrix outputs.
- Flask upload interface with file validation, EXIF orientation handling, an 8 MB limit, and prediction results.

## Technologies used

- **Language:** Python 3.10–3.13 (64-bit)
- **Machine learning:** TensorFlow/Keras, NumPy
- **Data and evaluation:** Pandas, scikit-learn
- **Image processing:** Pillow
- **Visualization:** Matplotlib
- **Web application:** Flask, HTML, CSS, JavaScript

TensorFlow uses CPU on native Windows. For supported GPU workflows, use Linux or WSL2.

## Dataset

This project uses the potato classes from the PlantVillage image dataset. Download the dataset from Kaggle:

- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

Review the dataset's Kaggle license and attribution requirements before redistributing it. Only the following three class folders are used:

```text
data/raw/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

Place the downloaded images into those folders. The repository intentionally does not include the dataset.

## Folder structure

```text
.
├── app.py                    # Flask application and prediction endpoint
├── train.py                  # Train and compare model configurations
├── evaluate.py               # Evaluate the selected model on test data
├── split_dataset.py          # Build the reproducible train/test/val split
├── dataset_verify.py         # Validate dataset structure and image files
├── promote_experiment.py     # Promote a selected experiment checkpoint
├── requirements.txt          # Python dependencies
├── data/
│   ├── raw/                  # Downloaded images; not committed
│   └── processed/            # Generated train/test/val split; not committed
├── models/                   # Generated Keras models and labels; not committed
├── outputs/                  # Generated metrics, plots, and reports; not committed
├── static/
│   ├── css/style.css         # Application styles
│   ├── js/script.js          # Upload and preview interactions
│   └── uploads/              # Temporary prediction previews; not committed
├── templates/
│   ├── index.html            # Upload page
│   └── result.html           # Prediction result page
└── tests/                    # Automated tests
```

## Architecture

```text
PlantVillage images
        |
        v
data/raw/ --> split_dataset.py --> data/processed/{train,test,val}
                                      |
                                      v
                                  train.py
                                      |
                                      v
                         models/potato_cnn.keras
                         models/class_names.json
                                      |
                 +--------------------+--------------------+
                 |                                         |
                 v                                         v
          evaluate.py                                  app.py
                 |                                         |
                 v                                         v
       outputs/reports and plots              Flask upload and prediction UI
```

The training pipeline reads the raw class folders, creates a deterministic split with seed `42`, trains candidate models, and saves the selected model. The evaluation script measures the selected model only on the held-out test set. The Flask app validates and decodes an uploaded image, converts it to RGB, resizes it to 128 × 128, loads the saved model once, and renders the prediction.

## Setup and commands

Open the project in VS Code and use PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download and arrange the dataset as described above, then run the pipeline in order:

```powershell
python split_dataset.py
python dataset_verify.py
python train.py
python evaluate.py
python app.py
```

Open <http://127.0.0.1:5000> in a browser. If PowerShell blocks virtual-environment activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Run tests with:

```powershell
python -m pytest
```

## Evaluation

`evaluate.py` writes the classification report, confusion matrix, per-class counts, and other evaluation artifacts to `outputs/`. Existing local experiments recorded MobileNetV2 at 97.91% test accuracy and 0.9716 macro F1 on the fixed split; these results are dataset-specific and should not be treated as a guarantee for field imagery.

## Limitations

- The model does not reject unrelated images or identify unknown diseases.
- PlantVillage images may not represent field conditions, lighting, camera quality, or cultivar variation.
- Predictions should be reviewed by a qualified agricultural professional.
- Dataset files, trained models, generated outputs, and uploaded images are excluded from Git by `.gitignore`.

## License and attribution

Add the project's chosen source-code license and the required PlantVillage/Kaggle attribution before public redistribution. The dataset's terms are governed by its source license, not by this application.
