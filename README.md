# Loan-Approval-Prediction

Welcome to the 2024 Kaggle Playground Series! We plan to continue in the spirit of previous playgrounds, providing interesting an approachable datasets for our community to practice their machine learning skills, and anticipate a competition each month.

Your Goal: The goal for this competition is to predict whether an applicant is approved for a loan.

## Setup
This repo uses a Conda environment configured in `environment.yml`. Here are the steps to set these up properly from this repos home folder:
1. Create an new Conda environment `conda env create -f environment.yml`
2. Activate the environment `conda activate Regression-Of-Used-Car-Prices`

If changes are made to `environment.yml` then update by running `conda env update --file environment.yml --prune`

## File Manifest
All models are genetated in the `./model/ <model-type>` folders. The goal is to try to solve this with many different strategies. Models can predict on the data by running the `./model/model_predict.py` script (after changing the path to the model pickle file). Predictions are automatically formatted for Kaggle and stored in `./submission`.