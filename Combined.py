import os
import cv2
import time
import numpy as np
import pickle
import speech_recognition as sr
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load models and resources
face_model = load_model("fer.h5")
text_model = load_model('emotion_prediction_model.keras')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

emotion_labels_reverse = {0: 'sad', 1: 'angry', 2: 'neutral', 3: 'happy'}
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Initialize webcam
cap = cv2.VideoCapture(0)


def predict_text_emotion(review_text):
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')
    prediction = text_model.predict(padded_sequence)
    predicted_class = np.argmax(prediction)
    emotion = emotion_labels_reverse[predicted_class]
    confidence = prediction[0][predicted_class]
    return emotion, confidence


def get_voice_input(retries=3, lang="en-US"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say your review...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        for attempt in range(retries):
            try:
                print("Listening...")
                audio = recognizer.listen(source)
                review_text = recognizer.recognize_google(audio, language=lang)
                print(f"You said: {review_text}")
                return review_text
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand. Trying again...")
            except sr.RequestError:
                print("Speech recognition service error.")
                break
    print("Failed to capture voice input.")
    return None


def predict_face_emotion(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    for (x, y, w, h) in faces:
        roi_gray = gray_img[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = face_model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        return emotions[max_index], predictions[0][max_index]
    return None, None


# === Step 1: Capture facial emotion for 10 seconds ===
print("Capturing facial emotions for 10 seconds...")
start_time = time.time()
face_emotion_counts = {emotion: 0 for emotion in emotions}
face_confidences = {emotion: [] for emotion in emotions}

while time.time() - start_time < 10:
    ret, frame = cap.read()
    if not ret:
        continue

    emotion, confidence = predict_face_emotion(frame)
    if emotion and confidence is not None:
        face_emotion_counts[emotion] += 1
        face_confidences[emotion].append(confidence)

        display_text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial Emotion Detection (10s)', resized_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Determine most frequent facial emotion
final_face_emotion = max(face_emotion_counts, key=face_emotion_counts.get)
average_face_confidence = (
    sum(face_confidences[final_face_emotion]) / len(face_confidences[final_face_emotion])
    if face_confidences[final_face_emotion] else 0.0
)

print(f"\nFinal Facial Emotion: {final_face_emotion} (Avg Confidence: {average_face_confidence:.2f})")

# === Step 2: Voice input and emotion detection ===
review = get_voice_input()
if review:
    speech_emotion, speech_confidence = predict_text_emotion(review)
    print(f"Speech Emotion: {speech_emotion} (Confidence: {speech_confidence:.2f})")

    # === Step 3: Combined result ===
    if final_face_emotion == speech_emotion:
        combined_emotion = final_face_emotion
        combined_confidence = (average_face_confidence + speech_confidence) / 2
    else:
        if average_face_confidence > speech_confidence:
            combined_emotion = final_face_emotion
            combined_confidence = average_face_confidence
        else:
            combined_emotion = speech_emotion
            combined_confidence = speech_confidence

    print(f"\nCombined Emotion: {combined_emotion} (Confidence: {combined_confidence:.2f})")

# Cleanup
cap.release()
cv2.destroyAllWindows()
