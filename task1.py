import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense

# Load the data from the CSV file
data = pd.read_csv("data.csv")

# Handle missing values (if necessary)
# data = data.dropna()  # Or use other methods to handle missing values

# Encode categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['protocol_type', 'service', 'flag'], drop_first=True)

# Separate features and target variable
X = data.drop(columns=['class'])
y = data['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Design the neural network architecture
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(y_train.unique()), activation='softmax')  # Use softmax for multiclass classification
])

# Compile the model specifying optimizer and loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train_scaled, pd.get_dummies(y_train), epochs=10, batch_size=32, verbose=1)

# Make predictions on unseen testing data
y_pred = model.predict(X_test_scaled)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Map predicted labels to original class names
class_names = ['attack', 'normal']
y_pred_mapped = [class_names[label] for label in y_pred_labels]

# Evaluate the model's performance
print(classification_report(y_test, y_pred_mapped))
