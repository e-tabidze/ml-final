import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

# Load the dataset containing network traffic data
data = pd.read_csv("network_traffic_data.csv")

# Preprocess the data
# Encode categorical features using one-hot encoding
data = pd.get_dummies(data, columns=['protocol_type', 'service', 'flag'])

# Ensure that the labels are properly encoded as binary values
data['label'] = data['label'].map({'normal': 0, 'attack': 1})

# Split the data into features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the data for compatibility with Conv1D layer
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Design the CNN architecture
model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model specifying optimizer and loss function
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)

# Make predictions on unseen testing data
y_pred_proba = model.predict(X_test_reshaped)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))
