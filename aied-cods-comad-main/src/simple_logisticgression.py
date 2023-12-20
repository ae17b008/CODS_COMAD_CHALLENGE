import os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
# data = pd.read_csv('/kaggle/input/data-challenge-auto-req-video-pre-requisite-data/train.csv')  # Adjust the path based on your dataset's name
data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
# Separate features and target variable
X = data.drop(columns=['kfold', 'label', 'pre requisite', 'concept', 'pre requisite taxonomy', 'concept taxonomy'])
y = data['label']

# Standardize the features since Logistic Regression is sensitive to feature scales
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize a 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Logistic Regression model
clf = LogisticRegression(random_state=42, max_iter=1000)  # Increased max_iter for convergence

# Perform 5-fold cross-validation
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Predict on the validation set
    y_pred = clf.predict(X_val)
    
    # Calculate and print the accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

# Note: This is a basic example. In a real-world scenario, you'd likely preprocess the data, tune hyperparameters, etc.