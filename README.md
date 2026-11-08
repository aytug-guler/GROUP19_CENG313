# CENG313 – MABe Challenge

## Project
Social Action Recognition in Mice

## Competition
https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/overview

## Group
GROUP19

## Members
- Efe Zeybek – 24118080089
- Aytuğ Güler – 24118080077

## Assignment 1
Exploratory Data Analysis and baseline modeling.

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

## Kaggle Submission History
No submission yet.
