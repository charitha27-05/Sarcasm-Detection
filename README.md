# Sarcasm Detection using NLP & Machine Learning 🧠😏

> An interactive fullstack NLP application that detects sarcasm in news headlines, tweets, and social comments using TF-IDF vectorization and supervised machine learning, complete with a Python Flask REST API and web user interface.

---

## 🌐 Live Application Deployment

- **Permanent Cloud Deployment:** Deployable 24/7 on [Render](https://render.com/) with one click using `render.yaml` or as a Docker container.
- **Backend Health Check Endpoint:** `/api/health`
- **Prediction API Endpoint:** `/api/predict`

> **Note on LocalTunnel:** The previous temporary tunnel URL (`metal-socks-think.loca.lt`) was ephemeral and expires whenever local tunneling stops. For permanent 24/7 access without requiring a local machine running, follow the Render deployment steps below.

---

## 📓 Google Colab Notebook

You can also run the original model experimentation and dataset training directly on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/charitha27-05/Sarcasm-Detection/blob/main/Sarcasm_Detection.ipynb)

---

## ✨ Features

- **Real-Time Sarcasm Prediction:** Instant classification of sentences into **Sarcastic 😏** or **Genuine / Not Sarcastic 🙂**.
- **Probability & Confidence Score:** Outputs percentage probabilities for both classes.
- **Pre-trained ML Model:** TF-IDF feature extractor (10,000 n-gram features) paired with a trained classifier achieving **80.26% test accuracy**.
- **RESTful API:** Clean JSON endpoints for seamless integration with other web/mobile apps.
- **Modern Interactive UI:** Glassmorphism web frontend with sample click-to-test prompts and responsive design.

---

## 📊 Model Performance

Trained on over 28,000 news headlines from The Onion (sarcastic) and HuffPost (genuine):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **80.26%** |
| **Macro Average F1** | **0.80** |
| **Sarcastic Precision** | **0.82** |
| **Not Sarcastic Recall** | **0.84** |

---

## 📡 API Reference

### 1. Health Check
* **Endpoint:** `GET /api/health`
* **Response:**
  ```json
  {
    "status": "online",
    "service": "Sarcasm Detection API",
    "algorithm": "TF-IDF + Logistic Regression",
    "accuracy": "80.26%"
  }
  ```

### 2. Predict Sarcasm
* **Endpoint:** `POST /api/predict`
* **Request Body:**
  ```json
  {
    "text": "Oh great, another rainy day when I forgot my umbrella!"
  }
  ```
* **Response:**
  ```json
  {
    "success": true,
    "text": "Oh great, another rainy day when I forgot my umbrella!",
    "is_sarcastic": true,
    "label": "Sarcastic",
    "confidence": 56.68,
    "probabilities": {
      "sarcastic": 56.68,
      "not_sarcastic": 43.32
    }
  }
  ```

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/charitha27-05/Sarcasm-Detection.git
   cd Sarcasm-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application:**
   ```bash
   python app.py
   ```

4. **Open in browser:**
   Navigate to [http://localhost:5050](http://localhost:5050)

---

## 🚀 Deploying to Render (24/7 Free Cloud)

1. Go to [Render.com](https://render.com/) and sign in.
2. Click **New +** ➔ **Web Service**.
3. Select your **`Sarcasm-Detection`** GitHub repository.
4. Settings:
   - **Environment / Language:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Plan:** `Free`
5. Click **Deploy Web Service**!

---

## 👩‍💻 Author
- **Charitha** ([@charitha27-05](https://github.com/charitha27-05))
