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

### Repository structure

    GROUP19_CENG313/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── assignment1_eda_baseline.ipynb
    ├── assignment2_solution.ipynb
    ├── src/
    │   ├── __init__.py
    │   ├── data.py
    │   ├── models.py
    │   └── utils.py
    ├── data/             local competition input
    └── submissions/      official score evidence

### Run Assignment 1 locally

1. Accept the competition rules and download the data from the [competition Data tab](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data).
2. Extract it to data/ so train.csv, test.csv, train_tracking/ and train_annotation/ are immediately under data/. Alternatively, set MABE_DATA_DIR to the extracted directory before launching Jupyter.
3. In a Python 3.12 environment, run the following from the project folder.

       python -m pip install -r requirements.txt
       python -m jupyter lab

4. Open assignment1_eda_baseline.ipynb and select Restart Kernel and Run All Cells. The checked-in notebook includes the original saved EDA, plots and model results. New summary cells populate on execution.

### Run in Colab

1. Accept the [MABe competition rules](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/rules) with the Kaggle account that owns your API token.
2. Upload the local `assignment1_eda_baseline.ipynb` to a fresh Colab CPU runtime. Under Secrets (the key icon), add `KAGGLE_API_TOKEN` and enable notebook access.
3. Select **Runtime → Run all**. The notebook installs the required libraries, downloads and locates the official competition data using KaggleHub, and loads the embedded `src/` snapshot. Allow space for the download and extraction.
4. The EDA, plots, 42/9/9 split and sniffbody evaluation run automatically.

### Run on Kaggle

1. Import the same local `assignment1_eda_baseline.ipynb` as a Kaggle notebook.
2. Under **Add Input**, attach the official [MABe competition data](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data).
3. Set **Accelerator** to **None** (CPU) and **Internet** to **Off**.
4. Use **Save Version → Save & Run All** and check that the run finishes. Inspect the EDA and evaluation in the saved notebook, and find `submission.csv` under **Output**. It is written to `/kaggle/working/submission.csv`.

### Competition model extension

The extension reuses the Assignment 1 Random Forest settings and the distance and motion feature ideas. Averaging available keypoints gives a per-mouse center across different laboratory tracking configurations. One binary forest per action trains on real annotation intervals and sampled frames within the labeling scope listed in `train.csv`. Sampling is capped per video, pair, action and class to limit CPU and memory use.

For each video, `test.csv` defines eligible mouse IDs and actions through `behaviors_labeled`. The pipeline intersects that scope with tracking coverage and merges consecutive predicted frames into half-open intervals. It validates the header against the provided sample and checks allowed triples, tracking coverage, interval validity and row IDs before reporting the CSV. The public input exposes a preview test set. The hidden set is provided by Kaggle during official scoring.

Set `RUN_EXTENSION_VALIDATION = True` in the setup cell to run the optional video-level holdout and per-action frame F1 evaluation. This adds a separate training pass. Set `RUN_SUBMISSION = True` in Colab or locally to exercise the CSV path against the public test input. The historical sniffbody test F1 of 0.4966 belongs to the original experiment. The extended model achieved an official Kaggle Public Score of **0.19871** and Private Score of **0.18551** through a successful late submission.

The notebook embeds exact copies of `src/__init__.py`, `src/data.py`, `src/models.py` and `src/utils.py`. Keep the setup cell's `MODULE_SOURCES` snapshot synchronized when updating these files.

### Limitations and Future Work

The historical baseline considers one directed social behavior and does not cover the full multi-behavior Kaggle prediction task. The frame-level samples are also temporally correlated within each video.

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
- 2026-11-13: Added offline Kaggle execution and automatic Colab setup to the original Assignment 1 notebook.
- 2026-11-13: Replaced the annotation-prior submission with supervised action models shared with Assignment 1 and added held-out behavior evaluation.
- 2026-11-13: Successfully submitted Assignment 1 to Kaggle as a late submission, achieving a Public Score of 0.19871 and Private Score of 0.18551.
## Kaggle submission status

The Assignment 1 notebook was successfully submitted to the **MABe Challenge – Social Action Recognition in Mice** as a late submission.


| Submission             | Public Score | Private Score | Status                     |
|------------------------|-------------:|--------------:|----------------------------|
| Group19(2) - Version 2 |      0.19871 |       0.18551 | Succeeded (after deadline) |


The submitted notebook performs exploratory data analysis, trains baseline models, and generates `submission.csv` using the competition's required format.

These scores are official Kaggle competition results and are distinct from the local binary `sniffbody` Random Forest test F1-score of approximately 0.4966.

Evidence: `submissions/kaggle_score_assignment1.png`.

Course requirements are documented in the [official CENG313 project guide](https://github.com/ceng313-2026/.github/blob/main/profile/README.md).

The Assignment 1 notebook embeds src/utils.py, src/data.py and src/models.py in that order, omitting relative imports because the cells share one namespace. Keep the embedded code in sync when changing the modules.
