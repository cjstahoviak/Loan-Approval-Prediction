# Import necessary libraries
import pandas as pd
import pickle
import os

# Step 1: Load the test data
test_data_path = './data/raw/test.csv'
test_data = pd.read_csv(test_data_path)

# Step 2: Load the saved model
model_path = './saved_model/xgboost_best.pkl'
with open(model_path, 'rb') as model_file:
    best_xgb_model = pickle.load(model_file)

# Extract the model name from the file (removing directories and extension)
model_name = os.path.splitext(os.path.basename(model_path))[0]

# Step 3: Preprocess the test data
# Retain the 'id' column for the submission file
test_ids = test_data['id']

# Drop unnecessary columns (same as in training) except 'id'
X_test = test_data.drop(columns=['id'])

# Apply the same preprocessing as the training data (e.g., get_dummies for categorical variables)
X_test = pd.get_dummies(X_test, drop_first=True)

# Ensure the test data has the same columns as training data
# (add missing columns as zeros to align with the model if necessary)
missing_cols = set(best_xgb_model.get_booster().feature_names) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0

# Align test set with training set columns
X_test = X_test[best_xgb_model.get_booster().feature_names]

# Step 4: Use the model to make predictions
# Predict the loan status (loan_status is the target column)
loan_status_predictions = best_xgb_model.predict(X_test)

# Step 5: Create a submission DataFrame with 'id' and 'loan_status'
submission_df = pd.DataFrame({
    'id': test_ids,
    'loan_status': loan_status_predictions
})

# Step 6: Save the submission file with the model name in the filename
submission_dir = './submission/'
submission_file_name = f'{model_name}_sub.csv'
submission_file_path = os.path.join(submission_dir, submission_file_name)

# Create the submission directory if it doesn't exist
os.makedirs(submission_dir, exist_ok=True)

# Save the submission file
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")
