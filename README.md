# 🩺 Smart Medical Assistant: Disease Prediction System

A professional-grade web application that leverages **Machine Learning (Random Forest)** to predict potential medical conditions based on user-reported symptoms. Designed for healthcare informatics and preliminary diagnostic support.

---

## 🚀 Live Demo
**http://localhost:8501**

---

## 📊 Project Overview
This project addresses the challenge of preliminary medical screening by using a supervised learning approach. It maps a high-dimensional symptom input vector to specific disease categories and provides immediate, formatted medication suggestions.

### Key Features:
* **Predictive Modeling:** Uses a Random Forest Classifier for robust, multi-class disease prediction.
* **Interactive UI:** Built with **Streamlit**, featuring a custom CSS-styled "Diagnostic Report" card.
* **Exportable Reports:** Users can download a timestamped `.txt` report of their results.
* **Data Sanitization:** Custom Python logic to handle and format complex medication strings.

---

## 🛠️ Tech Stack & Tools
* **Language:** Python 3.10+
* **ML Library:** Scikit-Learn (Random Forest)
* **Data Handling:** NumPy, Pandas
* **Frontend:** Streamlit (Custom HTML/CSS injection)
* **Model Deployment:** Joblib (Serialization)

---

## 📂 Project Structure
```text
├── app.py                           # Main Streamlit application code
├── requirements.txt                 # Project dependencies
├── random_forest_medical_model.pkl  # Trained ML Model
├── symptoms_list.pkl                # Serialized list of symptom features
└── README.md                        # Project documentation


## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Pranavraj3/Smart-Medical-Assistant.git
   cd Medical-Diagnostic-Assistant

