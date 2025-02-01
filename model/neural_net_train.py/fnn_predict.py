# fnn_predict.py

import pandas as pd
import torch
import os
from sklearn.preprocessing import StandardScaler
import torch.nn as nn

# Define the Feedforward Neural Network (FNN)
class FeedforwardNeuralNet(nn.Module):
    def __init__(self, input_size):
        super(FeedforwardNeuralNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(input_size, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)          # Second hidden layer
        self.fc3 = nn.Linear(64, 1)            # Output layer (binary classification)
        self.relu = nn.ReLU()                  # Activation function (ReLU)
        self.sigmoid = nn.Sigmoid()            # Sigmoid for binary classification

    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Sigmoid to output a probability
        return x

# Step 1: Load the test data
test_data_path = './data/raw/test.csv'
test_data = pd.read_csv(test_data_path)

# Step 2: Preprocess the test data
test_ids = test_data['id']
X_test = test_data.drop(columns=['id'])

X_test = pd.get_dummies(X_test, drop_first=True)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

X_test = torch.tensor(X_test, dtype=torch.float32)

# Step 3: Initialize the model architecture
input_size = X_test.shape[1]  # Must match input size used during training
model = FeedforwardNeuralNet(input_size)

# Step 4: Load the saved model weights
model_weights_path = './saved_model/fnn_best_weights.pth'  # Updated path
model.load_state_dict(torch.load(model_weights_path))  # Load the saved state_dict
model.eval()  # Set model to evaluation mode

# Step 5: Make predictions
with torch.no_grad():
    test_outputs = model(X_test)
    loan_status_predictions = (test_outputs > 0.5).float().numpy()

# Step 6: Create a submission DataFrame
submission_df = pd.DataFrame({
    'id': test_ids,
    'loan_status': loan_status_predictions.flatten()
})

# Step 7: Save the submission file
submission_dir = './submission/'
submission_file_name = 'fnn_sub.csv'
submission_file_path = os.path.join(submission_dir, submission_file_name)

os.makedirs(submission_dir, exist_ok=True)
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")
