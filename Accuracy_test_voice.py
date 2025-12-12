import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Step 1: Load the dataset (use the test data for evaluation)
df = pd.read_csv('E:\\archive\\tripadvisor_hotel_reviews.csv')

# Check for missing values and drop them if any
df.dropna(inplace=True)

# Define the emotion mapping
emotion_mapping = {
    1: "sad",
    2: "angry",
    3: "neutral",
    4: "happy",
    5: "happy"
}

# Map the 'Rating' column to emotions
df['Emotion'] = df['Rating'].map(emotion_mapping)

# Encode the emotions as integers
emotion_labels = {'sad': 0, 'angry': 1, 'neutral': 2, 'happy': 3}
df['Emotion'] = df['Emotion'].map(emotion_labels)

# Split data into features and labels
X = df['Review']  # Text reviews
y = df['Emotion'] # Emotion label (0: sad, 1: angry, 2: neutral, 3: happy)

# Set parameters for text preprocessing
vocab_size = 10000  # Limit vocabulary size
max_length = 100    # Maximum length of each sequence

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Convert text to sequences and pad them
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# Convert labels to categorical (one-hot encoded) format
y_categorical = to_categorical(y, num_classes=4)  # 4 classes: sad, angry, neutral, happy

# Step 2: Load the saved model
model = load_model('emotion_prediction_model.keras')

# Step 3: Generate Predictions on the Test Set
y_pred = model.predict(X_padded)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert to labels (0, 1, 2, 3)
y_true = np.argmax(y_categorical, axis=1)  # Convert true labels to indices (0, 1, 2, 3)

# Step 4: Generate the Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['sad', 'angry', 'neutral', 'happy'],
            yticklabels=['sad', 'angry', 'neutral', 'happy'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Emotion Prediction')
plt.show()

# Step 5: Compute the Accuracy
accuracy = np.sum(y_true == y_pred_labels) / len(y_true)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# Step 6: Plot the Accuracy Change Over Test Cases

# Calculate cumulative accuracy as we go through the test cases
cumulative_correct = 0
cumulative_accuracy = []

for i in range(len(y_true)):
    if y_true[i] == y_pred_labels[i]:
        cumulative_correct += 1
    cumulative_accuracy.append(cumulative_correct / (i + 1))

# Plot the cumulative accuracy change
plt.figure(figsize=(10, 6))
plt.plot(range(len(cumulative_accuracy)), cumulative_accuracy, label='Cumulative Accuracy', color='blue')
plt.axhline(y=accuracy, color='red', linestyle='--', label='Final Accuracy')
plt.xlabel('Number of Test Cases')
plt.ylabel('Accuracy')
plt.title('Cumulative Accuracy Over Test Cases')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Compute and Plot Correlation Matrix for True and Predicted Labels

# Compute the correlation matrix between the true labels and predicted labels
# Convert the predictions and true labels into one-hot encoded format
y_true_one_hot = to_categorical(y_true, num_classes=4)
y_pred_one_hot = to_categorical(y_pred_labels, num_classes=4)

# Compute the correlation matrix for the true and predicted labels
correlation_matrix = np.corrcoef(y_true_one_hot.T, y_pred_one_hot.T)  # Correlation between true and predicted labels

# Plot the correlation matrix using Seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True,
            xticklabels=['sad', 'angry', 'neutral', 'happy'],
            yticklabels=['sad', 'angry', 'neutral', 'happy'])
plt.title('Correlation Matrix between True and Predicted Emotion Labels')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
