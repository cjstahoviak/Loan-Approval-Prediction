# train_fnn.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

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

# Step 1: Load the dataset
data = pd.read_csv('./data/raw/train.csv')

# Step 2: Preprocess the data
X = data.drop(columns=['loan_status', 'id'])
y = data['loan_status']

X = pd.get_dummies(X, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Initialize the model
input_size = X_train.shape[1]
model = FeedforwardNeuralNet(input_size)

# Loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 7: Save the model weights (state_dict)
model_save_path = './saved_model/fnn_best_weights.pth'
torch.save(model.state_dict(), model_save_path)  # Save only the model weights
print(f"Model weights saved to {model_save_path}")
