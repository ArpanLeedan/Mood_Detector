import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
# Load the dataset (assuming the file is named 'tripadvisor_hotel_reviews.csv')
df = pd.read_csv('E:\\archive\\tripadvisor_hotel_reviews.csv')

# Display basic information about the data
print(df.head())

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
embedding_dim = 16  # Embedding dimension

# Tokenize the text
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index

# Convert text to sequences and pad them
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post')

# Save the tokenizer for future use
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# Convert labels to categorical (one-hot encoded) format
y_categorical = to_categorical(y, num_classes=4)  # 4 classes: sad, angry, neutral, happy

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_padded, y_categorical, test_size=0.2, random_state=42)
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(4, activation='softmax')  # 4 classes for multi-class classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
# Save the model
# Instead of model.save('emotion_prediction_model.h5')
model.save('emotion_prediction_model.keras')  # This uses the new format
