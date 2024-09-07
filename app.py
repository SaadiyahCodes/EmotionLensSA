from flask import Flask, render_template, request, jsonify
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import base64
import random
import cv2
import numpy as np
import torch

app = Flask(__name__)

# Load pre-trained model and processor for facial expression analysis
processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")

# Predefined set of mock interview questions
questions = [
    "Can you explain the concept of OOP?",
    "What is a binary tree?",
    "What are hash tables used for?",
    "How does garbage collection work in Java?",
    "What is the time complexity of binary search?",
    "Explain the difference between a process and a thread.",
    "What are the four pillars of OOP?",
    "What is the difference between stack and heap memory?",
    "What are the advantages of using recursion?",
    "What is polymorphism in computer science?"
]

# Mock interview duration (in seconds)
INTERVIEW_DURATION = 180

# Define a feedback mechanism
def generate_feedback(expression_analysis, face_presence):
    expression_feedback = {
        "Happy": "You appeared confident and positive during the interview.",
        "Sad": "You seemed a bit down during the interview. Try to smile more!",
        "Angry": "There was some frustration detected. Stay calm and composed!",
        "Neutral": "You seemed neutral. Try to show more enthusiasm!",
        # More expressions can be added here.
    }

    feedback = []
    # Analyze the expressions from the interview session
    if expression_analysis:
        expression_counts = {label: expression_analysis.count(label) for label in set(expression_analysis)}
        most_common_expression = max(expression_counts, key=expression_counts.get, default="Neutral")
        feedback.append(expression_feedback.get(most_common_expression, "Your expressions were mixed."))
    else:
        feedback.append("No expressions detected.")

    # Analyze eye contact (based on face presence)
    if len(face_presence) > 0 and sum(face_presence) / len(face_presence) > 0.7:
        feedback.append("Good eye contact! You maintained focus during the interview.")
    else:
        feedback.append("Your eye contact could be improved. Try to face the camera more.")
    
    return " ".join(feedback)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/mock-interview')
def mock_interview():
    # Randomly select interview questions
    selected_questions = random.sample(questions, 5)
    return render_template('interview.html', questions=selected_questions)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    image_data = data['image'].split(",")[1]

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except Exception as e:
        print(f"Error loading image: {e}")  # Debugging
        return jsonify({'error': 'Invalid image data or format.'}), 400

    # Preprocess the image for facial expression analysis
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted expression
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]

    # Basic face detection for eye contact analysis
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

    # Determine if face is present
    face_present = len(faces) > 0

    print(f"Detected expression: {label}, Face present: {face_present}")  # Debugging

    return jsonify({'label': label, 'face_present': face_present})

@app.route('/end-interview', methods=['POST'])
def end_interview():
    expression_analysis = request.json.get('expressions', [])
    face_presence = request.json.get('face_presence', [])

    # Generate feedback based on analysis
    feedback = generate_feedback(expression_analysis, face_presence)
    
    print(f"Expression analysis: {expression_analysis}")  # Debugging
    print(f"Face presence: {face_presence}")  # Debugging
    
    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)
