# Import required libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
data = pd.read_csv('./data/raw/train.csv')

# Step 2: Preprocessing the Data
# Split data into features (X) and labels (y)
X = data.drop(columns=['loan_status', 'id'])
y = data['loan_status']

# Encode categorical variables if needed (get_dummies)
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the train and test sets
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Step 3: Set up XGBoost Classifier with GPU Support
xgb_model = xgb.XGBClassifier(
    tree_method='hist',  # Use GPU for training
    device='cuda',
    use_label_encoder=False, # Disable label encoder (to avoid warnings)
    eval_metric='mlogloss'   # Evaluation metric
)

# Step 4: Define the hyperparameter grid for tuning
param_grid = {
    'max_depth': [5, 7, 9, 15],           # Depth of trees
    'learning_rate': [0.07, 0.1, 0.12], # Step size shrinkage
    'n_estimators': [50, 100, 150],        # Number of trees
    'subsample': [0.95, 1.0],           # Fraction of samples to be used for fitting
    'colsample_bytree': [0.8, 1.0, 1.2],    # Fraction of features to be used
}

# Step 5: Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(
    estimator=xgb_model, 
    param_grid=param_grid, 
    scoring='accuracy',    # Evaluation metric for scoring
    cv=3,                  # 3-fold cross-validation
    verbose=1              # Print progress
)

# Step 6: Train the model with Grid Search
grid_search.fit(X_train, y_train)

# Step 7: Get the best model from Grid Search
best_xgb_model = grid_search.best_estimator_

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Step 8: Evaluate the best model on the test set
# Make predictions on the test set
y_pred = best_xgb_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 9: Save the best model to a file
model_save_path = './saved_model/xgboost_best.pkl'
with open(model_save_path, 'wb') as model_file:
    pickle.dump(best_xgb_model, model_file)

print(f"Best XGBoost model saved to {model_save_path}")
