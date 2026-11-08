# CENG313 – MABe Challenge

## Project
Social Action Recognition in Mice

## Competition
https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview

## Group
GROUP19

## Members
- Efe Zeybek – 24118080089  GitHub username: EfeZeybek05
- Aytuğ Güler – 24118080077 GitHub username: aytug-guler

## Assignment 1
Assignment 1 focuses on exploratory data analysis (EDA) and the development of baseline models for mouse social behavior recognition using the MABe Challenge dataset.

### Dataset and Exploratory Analysis

The dataset contains 8,789 training videos with tracking data and associated metadata. Behavior annotation files are available for 847 videos.

The EDA investigates missing values, laboratory distributions, video durations, frame rates, behavior frequencies, and mouse movement patterns.

### Baseline Classification Task

For Assignment 1, we selected a focused binary classification task: detecting **mouse 1 → mouse 2 sniffbody behavior** at the frame level.

We selected 60 eligible videos from the `CalMS21_task1` dataset and divided them into independent video-level splits:

| Split        | Videos | Valid Frames |
|--------------|-------:|-------------:|
| Training     |   42   |    242,132   |
| Validation   |   9    |    41,451    |
| Test         |   9    |    54,192    |

Seven tracking-based features were extracted, including inter-mouse distances, movement speeds, and changes in relative distance. Invalid tracking coordinates and incomplete feature rows were excluded.

### Baseline Models

We evaluated two models:

- **Dummy Classifier:** A majority-class baseline.
- **Random Forest:** A classification model using 80 estimators, maximum depth 12, minimum leaf size 20, and balanced subsample class weights.

The classification threshold was optimized using the validation set only. The independent test set was reserved for final evaluation.

### Results

| Metric | Random Forest |
|---|---:|
| Validation F1-score (threshold 0.5) | 0.4793 |
| Optimized validation F1-score | 0.4883 |
| Selected threshold | 0.5896 |
| Test accuracy | 0.7918 |
| Test precision | 0.3707 |
| Test recall | 0.7520 |
| Test F1-score | 0.4966 |
| Test average precision | 0.4403 |

The model detects a substantial proportion of positive sniffbody frames, although false-positive predictions remain a limitation.

**Note:** These results represent a focused binary classification baseline, not an official Kaggle competition score.

### Repository Structure

```text
GROUP19_CENG313/
├── README.md
├── assignment1_eda_baseline.ipynb
├── requirements.txt
├── .gitignore
└── src/
    ├── data.py
    └── models.py
```

- `assignment1_eda_baseline.ipynb`: Exploratory analysis, baseline experiments, visualizations, and evaluation.
- `src/data.py`: Reusable feature extraction, annotation processing, and dataset construction.
- `src/models.py`: Baseline training, evaluation, threshold optimization, and feature importance functions.
- `requirements.txt`: Required Python libraries.
- `.gitignore`: Excludes local datasets, credentials, and unnecessary generated files.

### How to Run

1. Open `assignment1_eda_baseline.ipynb` in Google Colab.
2. Ensure that the Kaggle account has accepted the MABe Challenge competition rules.
3. Create a Colab secret named `KAGGLE_API_TOKEN` containing a valid Kaggle API token, and enable notebook access.
4. Run the notebook cells in order. The notebook clones the public GitHub repository and imports reusable functions from `src/`.
5. The notebook downloads the competition dataset, performs exploratory analysis, constructs the video-level splits, trains baseline classifiers, and evaluates the Random Forest model.

A Google Colab runtime with sufficient memory and storage is required to process the dataset.

### Limitations and Future Work

The baseline considers one directed social behavior and does not cover the full multi-behavior Kaggle prediction task. The frame-level samples are also temporally correlated within each video.

Assignment 2 will build upon this repository with improved modeling approaches, additional experiments, and comparisons against the Assignment 1 baseline.

## Progress Log
- 2026-10-11: Created the GitHub repository and initialized the project.
- 2026-11-07: Set up Google Colab and downloaded the MABe dataset.
- 2026-11-07: Inspected 8,789 training records and 38 metadata columns.
- 2026-11-07: Identified 8,789 training tracking files and 847 annotation files.
- 2026-11-07: Completed missing-value analysis of 38 metadata columns.
- 2026-11-07: Analyzed the distribution of 8,789 training videos across 21 laboratories.
- 2026-11-07: Found that MABe22_keypoints and MABe22_movies account for approximately 90.2% of training videos.
- 2026-11-07: Examined video durations and frame rates, identifying 404 recordings longer than 600 seconds and 43 recordings above 60 FPS.
- 2026-11-07: Added EDA visualizations and interpretations to the assignment notebook.
- 2026-11-08: Inspected tracking coordinates and behavior annotation intervals.
- 2026-11-08: Analyzed 83,620 annotated behavior events across 37 behavior categories.
- 2026-11-08: Identified substantial class imbalance, with sniff accounting for approximately 45.2% of annotated events.
- 2026-11-08: Explored mouse tracking data and visualized movement trajectories.
- 2026-11-08: Calculated inter-mouse distances and compared tracking measurements with a labeled chase event.
- 2026-11-08: Analyzed annotation coverage across 21 data sources, identifying two large sources without training annotation files.
- 2026-11-08: Completed the initial EDA stage of Assignment 1.
- 2026-11-08: Selected 60 CalMS21_task1 videos for directed sniffbody detection and created a 42/9/9 video-level split.
- 2026-11-08: Implemented frame-level labels, tracking-coordinate cleaning, and seven motion-related features.
- 2026-11-08: Trained Dummy and Random Forest baselines on 242,132 valid training frames.
- 2026-11-08: Random Forest achieved 0.4793 validation F1-score and 0.3976 average precision.
- 2026-11-08: Compared Dummy and Random Forest baselines using validation metrics and precision-recall analysis.
- 2026-11-08: Optimized the Random Forest decision threshold to 0.5896, achieving a validation F1-score of 0.4883.
- 2026-11-08: Analyzed feature importance and found that spatial distance features accounted for approximately 92.75% of impurity-based importance.
- 2026-11-08: Evaluated the fixed Random Forest on 9 independent test videos (54,192 valid frames), achieving F1-score 0.4966 and average precision 0.4403.
- 2026-11-08: Refactored reusable data processing and baseline model functions into src/data.py and src/models.py.
- 2026-11-08: Integrated GitHub modules into Google Colab and successfully reproduced the original training, validation, and independent test results.

## Kaggle Submission History
No submission yet.
