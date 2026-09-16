# ml_service/model.py
print("[ml_service] module import", "__name__=", __name__)

from dotenv import load_dotenv
load_dotenv()
print("ENV FILE LOADED")

import os
import re
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from flask import Flask, request, jsonify
from flask_cors import CORS
import openai

# --------------------------------------------------
# OpenAI setup (chat only, NOT diagnosis)
# --------------------------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key:
    print("[ml_service] OPENAI_API_KEY loaded")
else:
    print("[ml_service] WARNING: OPENAI_API_KEY not set")

# --------------------------------------------------
# Flask App
# --------------------------------------------------
app = Flask(__name__)
CORS(app)

# ==================================================
# ML MODEL
# ==================================================
class SymptomMLModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=300,
            ngram_range=(1, 3),
            stop_words="english",
            min_df=2,
            max_df=0.9
        )

        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            class_weight="balanced",
            bootstrap=True,
            max_features="sqrt"
        )

        self.encoder = LabelEncoder()
        self.trained = False
        self.symptom_patterns = {}

    # --------------------------------------------------
    # DISEASE + SYMPTOMS (FROM FIRST CODE ONLY)
    # --------------------------------------------------
    def _create_training_data(self):
        print("[ML] Creating training data...")

        disease_patterns = {
            "Common Cold": {
                "core": [
                    "runny nose", "sneezing", "sore throat",
                    "blocked nose", "mild cough", "watery eyes"
                ],
                "common": [
                    "cough", "slight fever", "tiredness",
                    "headache", "reduced taste"
                ],
                "rare": ["body pain"]
            },
            "Influenza (Flu)": {
                "core": [
                    "high fever", "body pain", "muscle pain",
                    "dry cough", "shivering", "weakness"
                ],
                "common": [
                    "headache", "runny nose",
                    "poor appetite", "chest discomfort"
                ],
                "rare": ["vomiting"]
            },
            "Migraine": {
                "core": [
                    "strong headache", "head pain on one side",
                    "sensitivity to light", "sensitivity to sound"
                ],
                "common": [
                    "nausea", "vomiting",
                    "blurred vision"
                ],
                "rare": ["dizziness"]
            },
            "Gastroenteritis": {
                "core": [
                    "vomiting", "loose motion",
                    "stomach pain", "stomach cramps"
                ],
                "common": [
                    "fever", "poor appetite",
                    "dehydration"
                ],
                "rare": ["body pain"]
            },
            "Urinary Tract Infection": {
                "core": [
                    "burning while urinating",
                    "frequent urination",
                    "pain while passing urine"
                ],
                "common": [
                    "lower stomach pain",
                    "cloudy urine"
                ],
                "rare": ["blood in urine"]
            },
            "COVID-19": {
                "core": [
                    "loss of taste", "loss of smell",
                    "fever", "dry cough"
                ],
                "common": [
                    "tiredness", "body pain",
                    "headache"
                ],
                "rare": ["breathing difficulty"]
            },
            "Diabetes": {
                "core": [
                    "frequent urination",
                    "excessive thirst",
                    "sudden weight loss"
                ],
                "common": [
                    "tiredness",
                    "blurred vision"
                ],
                "rare": ["slow healing wounds"]
            },
            "Asthma": {
                "core": [
                    "shortness of breath",
                    "wheezing",
                    "chest tightness"
                ],
                "common": [
                    "cough",
                    "breathing difficulty"
                ],
                "rare": ["bluish lips"]
            },
            "Appendicitis": {
                "core": [
                    "pain in lower right stomach",
                    "severe stomach pain",
                    "vomiting"
                ],
                "common": [
                    "loss of appetite",
                    "fever"
                ],
                "rare": ["severe infection"]
            },
            "Pneumonia": {
                "core": [
                    "cough with mucus",
                    "fever",
                    "breathing difficulty"
                ],
                "common": [
                    "chest pain",
                    "tiredness"
                ],
                "rare": ["confusion"]
            },
            "Ortho related Injury": {
                "core": [
                    "joint pain",
                    "swelling",
                    "limited movement",
                    "back pain"
                ],
                "common": [
                    "muscle ache",
                    "stiffness"
                ],
                "rare": ["numbness"]
            }
        }

        samples = []

        for disease, groups in disease_patterns.items():
            all_symptoms = groups["core"] + groups["common"] + groups["rare"]
            self.symptom_patterns[disease] = all_symptoms

            for symptom in all_symptoms:
                samples.append({
                    "symptoms": f"having {symptom}",
                    "disease": disease
                })
                samples.append({
                    "symptoms": f"experiencing {symptom} for few days",
                    "disease": disease
                })
                samples.append({
                    "symptoms": f"{symptom}",
                    "disease": disease
                })

        df = pd.DataFrame(samples)
        print(f"[ML] Training samples generated: {len(df)}")
        return df

    # --------------------------------------------------
    # TEXT PROCESSING (FROM SECOND CODE)
    # --------------------------------------------------
    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_features(self, text):
        words = text.split()
        features = set(words)

        for i in range(len(words)):
            if i < len(words) - 1:
                features.add(f"{words[i]}_{words[i+1]}")
            if i < len(words) - 2:
                features.add(f"{words[i]}_{words[i+1]}_{words[i+2]}")

        return " ".join(features)

    # --------------------------------------------------
    # TRAINING (NO ACCURACY TABLE)
    # --------------------------------------------------
    def train(self):
        print("[ML] Training model...")
        df = self._create_training_data()

        df["cleaned"] = df["symptoms"].apply(self._clean_text)
        df["enhanced"] = df["cleaned"].apply(self._extract_features)

        X = self.vectorizer.fit_transform(df["enhanced"])
        y = self.encoder.fit_transform(df["disease"])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)
        self._save_model()
        self.trained = True

        print("[ML] Training completed")
        return True

    def _save_model(self):
        joblib.dump(self.vectorizer, "symptom_vectorizer.joblib")
        joblib.dump(self.encoder, "disease_encoder.joblib")
        joblib.dump(self.model, "diagnosis_ml_model.joblib")
        with open("symptom_patterns.json", "w") as f:
            json.dump(self.symptom_patterns, f)

    def load_model(self):
        try:
            self.vectorizer = joblib.load("symptom_vectorizer.joblib")
            self.encoder = joblib.load("disease_encoder.joblib")
            self.model = joblib.load("diagnosis_ml_model.joblib")
            with open("symptom_patterns.json", "r") as f:
                self.symptom_patterns = json.load(f)
            self.trained = True
            print("[ML] Pre-trained model loaded")
            return True
        except Exception:
            return False

    # --------------------------------------------------
    # PREDICTION (CONFIDENCE >= 0.30)
    # --------------------------------------------------
    def predict(self, text):
        cleaned = self._clean_text(text)
        enhanced = self._extract_features(cleaned)

        X = self.vectorizer.transform([enhanced])
        if X.sum() == 0:
            return None, 0.0, [], "unknown"

        probs = self.model.predict_proba(X)[0]
        idx = np.argmax(probs)

        confidence = float(probs[idx])
        if confidence < 0.30:
            return None, confidence, [], "unknown"

        disease = self.encoder.inverse_transform([idx])[0]
        identified = self._identify_symptoms(cleaned)
        urgency = self._get_urgency(disease, confidence, len(identified))

        return disease, confidence, identified, urgency

    def _identify_symptoms(self, text):
        identified = []
        for symptoms in self.symptom_patterns.values():
            for symptom in symptoms:
                if symptom in text:
                    identified.append(symptom)
        return list(set(identified))

    # --------------------------------------------------
    # URGENCY LOGIC (FROM FIRST CODE)
    # --------------------------------------------------
    def _get_urgency(self, disease, confidence, symptom_count):
        urgent = ["Pneumonia", "COVID-19", "Urinary Tract Infection"]
        emergency = ["Appendicitis"]

        if disease in emergency and confidence > 0.4:
            return "emergency"
        elif disease in urgent and confidence > 0.4:
            return "high"
        elif symptom_count >= 3 and confidence > 0.6:
            return "medium"
        else:
            return "low"

    # --------------------------------------------------
    # RESPONSE MESSAGES (FROM FIRST CODE)
    # --------------------------------------------------
    def get_recommendation(self, disease, confidence):
        recommendations = {
            "Common Cold":
                "Rest, stay hydrated, use basic cold medicines. Consult a General Physician.",
            "Influenza (Flu)":
                "Rest, drink fluids, use fever reducers. Consult a General Physician.",
            "Migraine":
                "Rest in a dark quiet room, stay hydrated. Consult a Neurologist.",
            "Gastroenteritis":
                "Drink fluids and eat light food. Consult a Gastroenterologist.",
            "Urinary Tract Infection":
                "Increase water intake. Treatment may be required. Consult a Urologist.",
            "COVID-19":
                "Isolate, rest, monitor symptoms. Consult a General Physician.",
            "Diabetes":
                "Monitor blood sugar levels and diet. Consult an Endocrinologist.",
            "Asthma":
                "Use prescribed inhaler and avoid triggers. Consult a Pulmonologist.",
            "Appendicitis":
                "This may require urgent medical evaluation. Go to the hospital immediately.",
            "Pneumonia":
                "Medical treatment is required. Consult a Pulmonologist immediately.",
            "Ortho related Injury":
                "Rest, ice, compression, and elevation. Consult an Orthopedic Specialist."
        }

        base = recommendations.get(
            disease,
            "Consult a healthcare professional."
        )

        if confidence > 0.7:
            base = f"High confidence prediction. {base}"
        elif confidence < 0.4:
            base = f"Low confidence prediction. {base}"

        return base

    def needs_doctor(self, urgency):
        return urgency in ["high", "emergency"]


# ==================================================
# MODEL INIT
# ==================================================
ml_model = SymptomMLModel()
if not ml_model.load_model():
    ml_model.train()

# ==================================================
# OPENAI CHAT (SUPPORT ONLY)
# ==================================================
def chat_with_openai(message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content":
                        "You are a helpful medical assistant. "
                        "Provide general health information ONLY!, if the query is not related to health, then just handle the situation yourself"
                        "Do not diagnose symptoms."
                },
                {"role": "user", "content": message}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "I can help with general health questions. Please use the symptom checker for diagnosis."

# ==================================================
# ROUTES
# ==================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "trained": ml_model.trained,
        "diseases": list(ml_model.encoder.classes_)
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json() or {}
    symptoms = str(payload.get("symptoms", "")).strip()

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    disease, confidence, identified, urgency = ml_model.predict(symptoms)

    if disease is None:
        return jsonify({
            "diagnosis": "No clear diagnosis identified",
            "confidence": float(confidence),
            "recommendation":
                "Please provide more specific symptoms or consult a healthcare professional.",
            "needs_doctor": True,
            "urgency": "unknown"
        })

    return jsonify({
        "diagnosis": f"Possible {disease}",
        "identified_symptoms": identified,
        "confidence": float(confidence),
        "recommendation": ml_model.get_recommendation(disease, confidence),
        "needs_doctor": ml_model.needs_doctor(urgency),
        "urgency": urgency
    })

@app.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json() or {}
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please enter a message."})
    return jsonify({"reply": chat_with_openai(message)})

@app.route("/retrain", methods=["POST"])
def retrain():
    ml_model.train()
    return jsonify({"success": True, "message": "Model retrained successfully"})

# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    port = int(os.environ.get("ML_PORT", 5000))
    print(f"[ml_service] Running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
