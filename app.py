from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import markdown
from markupsafe import Markup
import requests
import json
from os import getenv
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import io
import base64
import cv2
import numpy as np
import torch
import random  # Import the random module

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    expression_analysis = db.Column(db.JSON, nullable=False)
    face_presence = db.Column(db.JSON, nullable=False)
    feedback = db.Column(db.Text, nullable=True)

with app.app_context():
    db.create_all()


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
        "Surprised": "You seemed surprised at times. Try to maintain a more consistent expression.",
        "Fear": "You appeared fearful at times. Try to relax and stay calm.",
    }

    feedback = []
    # Ensure expression_analysis is a dictionary
    if not isinstance(expression_analysis, dict):
        expression_analysis = {label: 0 for label in expression_feedback.keys()}

    # Analyze the expressions from the interview session
    expression_counts = {label: expression_analysis.get(label.lower(), 0) for label in expression_feedback.keys()}
    
    # If the most common expression is not "Happy", directly use that expression
    most_common_expression = max(expression_counts, key=expression_counts.get, default="Neutral")

    # If the expression is found in the expression feedback, add it, otherwise just use the label.
    feedback.append(expression_feedback.get(most_common_expression, most_common_expression))

    # Analyze eye contact (based on face presence)
    if len(face_presence) > 0 and sum(face_presence) / len(face_presence) > 0.7:
        feedback.append("/nGood eye contact! You maintained focus during the interview.")
    else:
        feedback.append("/nYour eye contact could be improved. Try to face the camera more.")

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

    # Initialize label and face_present variables
    label = "Neutral"
    face_present = False

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        print("Image received and decoded successfully.")
    except Exception as e:
        return jsonify({'error': f"Error decoding image: {e}"}), 400

    # Preprocess the image for facial expression analysis
    try:
        inputs = processor(images=image, return_tensors="pt")
        print("Image preprocessed successfully.")
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return jsonify({'error': f"Error preprocessing image: {e}"}), 500

    # Perform inference for facial expression prediction
    try:
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_idx]
        print(f"Predicted expression: {label}")
    except Exception as e:
        print(f"Error during inference: {e}")
        return jsonify({'error': f"Error during inference: {e}"}), 500

    # Basic face detection for eye contact analysis
    try:
        image_np = np.array(image)
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
        face_present = len(faces) > 0
        print(f"Face detected: {face_present}")
    except Exception as e:
        print(f"Error during face detection: {e}")
        return jsonify({'error': f"Error during face detection: {e}"}), 500

    return jsonify({'label': label, 'face_present': face_present})

@app.route('/end-interview', methods=['POST'])
def end_interview():
    expression_analysis = request.json.get('expressions', {
        "happy": 0,
        "sad": 0,
        "angry": 0,
        "neutral": 0,
        "fear": 0,
        "surprised": 0
    })
    face_presence = request.json.get('face_presence', [])

    # Generate feedback based on analysis
    feedback = generate_feedback(expression_analysis, face_presence)
    interview = Interview(expression_analysis=expression_analysis, face_presence=face_presence, feedback=feedback)
    db.session.add(interview)
    db.session.commit()

    #generate ai feedback
    ai_feedback = generate_ai_feedback(expression_analysis)
    
    return jsonify({'feedback': feedback, 'ai_feedback': ai_feedback})

def generate_ai_feedback(expression_counts):
    #to avoid KeyError
    happy_count = expression_counts.get('happy', 0)
    sad_count = expression_counts.get('sad', 0)
    angry_count = expression_counts.get('angry', 0)
    neutral_count = expression_counts.get('neutral', 0)
    fear_count = expression_counts.get('fear', 0)
    surprised_count = expression_counts.get('surprised', 0)

    message = (
        f"The user had the following emotional states during the interview: "
        f"Happy: {happy_count}, Sad: {sad_count}, Angry: {angry_count}, "
        f"Neutral: {neutral_count}, Fear: {fear_count}, Surprised: {surprised_count}."
        f" Prepare feedback based on their facial expressions and give tips to improve. "
        f"Keep the performance summary brief around 3 to 4 lines and tips brief around 6-7 lines."
    )

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {getenv('OPENROUTER_API_KEY')}",
        },
        data=json.dumps({
            "model": "meta-llama/llama-3.1-8b-instruct:free",
            "messages": [
                {"role": "user", "content": message}
            ]
        })
    )
    ai_feedback = response.json().get("choices")[0].get("message").get("content")
    html_feedback = markdown.markdown(ai_feedback) # Convert to HTML
    html_feedback_safe = Markup(html_feedback) # Mark as safe

    return f"Your performance summary:\n{html_feedback_safe}"


if __name__ == '__main__':
    app.run(debug=True)
