import os
import re
import joblib
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Stopwords for cleaning
STOPWORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o',
    're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn',
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)

# Load trained model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "sarcasm_model.joblib")
VEC_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.joblib")

print("Loading model and vectorizer...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)
print("Model loaded successfully!")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sarcasm Detection AI</title>
  <style>
    :root {
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --bg: #0f172a;
      --card-bg: rgba(30, 41, 59, 0.85);
      --border: rgba(255, 255, 255, 0.1);
      --text: #f8fafc;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at 50% 20%, #1e1b4b 0%, #0f172a 100%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 30px 20px;
    }
    .container {
      width: 100%;
      max-width: 680px;
      margin-top: 20px;
    }
    .header {
      text-align: center;
      margin-bottom: 30px;
    }
    .header h1 {
      font-size: 2.4rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      background: linear-gradient(135deg, #a5b4fc 0%, #c084fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 10px;
    }
    .header p {
      color: var(--text-muted);
      font-size: 1.05rem;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 30px;
      box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
    }
    .input-group label {
      display: block;
      font-weight: 600;
      font-size: 0.95rem;
      margin-bottom: 10px;
      color: #e2e8f0;
    }
    textarea {
      width: 100%;
      min-height: 110px;
      padding: 14px 16px;
      font-size: 1rem;
      font-family: inherit;
      border-radius: 12px;
      background: rgba(15, 23, 42, 0.7);
      border: 1px solid var(--border);
      color: white;
      resize: vertical;
      outline: none;
      transition: all 0.2s ease;
    }
    textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
    }
    .samples-container {
      margin-top: 15px;
    }
    .samples-title {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin-bottom: 8px;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 0.82rem;
      color: #cbd5e1;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .chip:hover {
      background: rgba(99, 102, 241, 0.2);
      border-color: var(--primary);
      color: #fff;
    }
    .btn-submit {
      width: 100%;
      margin-top: 22px;
      padding: 14px;
      font-size: 1.05rem;
      font-weight: 700;
      color: white;
      background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
      border: none;
      border-radius: 12px;
      cursor: pointer;
      box-shadow: 0 8px 20px -4px rgba(99, 102, 241, 0.5);
      transition: all 0.2s ease;
    }
    .btn-submit:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px -4px rgba(99, 102, 241, 0.6);
    }
    .btn-submit:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    #resultCard {
      display: none;
      margin-top: 25px;
      padding: 24px;
      border-radius: 16px;
      text-align: center;
      animation: fadeIn 0.3s ease;
      border: 1px solid;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .result-sarcastic {
      background: rgba(239, 68, 68, 0.15);
      border-color: rgba(239, 68, 68, 0.4);
    }
    .result-not-sarcastic {
      background: rgba(34, 197, 94, 0.15);
      border-color: rgba(34, 197, 94, 0.4);
    }
    .result-emoji {
      font-size: 3rem;
      margin-bottom: 8px;
    }
    .result-label {
      font-size: 1.5rem;
      font-weight: 800;
      margin-bottom: 8px;
    }
    .result-confidence {
      font-size: 0.95rem;
      color: #cbd5e1;
      margin-bottom: 15px;
    }
    .progress-bar-bg {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      height: 8px;
      overflow: hidden;
      width: 80%;
      margin: 0 auto;
    }
    .progress-bar-fill {
      height: 100%;
      border-radius: 10px;
      transition: width 0.5s ease;
    }
    .footer {
      margin-top: 40px;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.85rem;
    }
    .footer a {
      color: #a5b4fc;
      text-decoration: none;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🧠 Sarcasm Detection AI</h1>
      <p>Analyze text and detect sarcastic tones using Natural Language Processing</p>
    </div>

    <div class="card">
      <form id="detectForm">
        <div class="input-group">
          <label for="inputText">Enter a sentence, tweet, or headline:</label>
          <textarea id="inputText" placeholder="e.g. Oh fantastic, another Monday morning meeting!" required></textarea>
        </div>

        <div class="samples-container">
          <div class="samples-title">Try sample examples:</div>
          <div class="chips">
            <span class="chip" onclick="setSample('Oh great, another rainy day when I forgot my umbrella!')">🌧️ Rainy Day</span>
            <span class="chip" onclick="setSample('Inclement weather prevents liar from getting to work')">💼 Late to Work</span>
            <span class="chip" onclick="setSample('Eat your veggies: 9 deliciously different recipes')">🥗 Veggie Recipes</span>
            <span class="chip" onclick="setSample('5 ways to file your taxes with less stress')">📊 Tax Tips</span>
          </div>
        </div>

        <button type="submit" class="btn-submit" id="submitBtn">Detect Sarcasm</button>
      </form>

      <div id="resultCard">
        <div class="result-emoji" id="resEmoji"></div>
        <div class="result-label" id="resLabel"></div>
        <div class="result-confidence" id="resConfidence"></div>
        <div class="progress-bar-bg">
          <div class="progress-bar-fill" id="resBar"></div>
        </div>
      </div>
    </div>

    <div class="footer">
      <p>Built with Python, Scikit-Learn & Flask • Project by <a href="https://github.com/charitha27-05/Sarcasm-Detection" target="_blank">charitha27-05</a></p>
    </div>
  </div>

  <script>
    function setSample(text) {
      document.getElementById('inputText').value = text;
      detectSarcasm();
    }

    async function detectSarcasm(e) {
      if (e) e.preventDefault();
      const input = document.getElementById('inputText').value.trim();
      if (!input) return;

      const btn = document.getElementById('submitBtn');
      const card = document.getElementById('resultCard');
      const emoji = document.getElementById('resEmoji');
      const label = document.getElementById('resLabel');
      const confidence = document.getElementById('resConfidence');
      const bar = document.getElementById('resBar');

      btn.disabled = true;
      btn.textContent = 'Analyzing text...';

      try {
        const res = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: input })
        });
        const data = await res.json();

        if (data.success) {
          card.style.display = 'block';
          if (data.is_sarcastic) {
            card.className = 'result-sarcastic';
            emoji.textContent = '😏';
            label.textContent = 'Sarcastic Tone Detected';
            label.style.color = '#f87171';
            bar.style.backgroundColor = '#ef4444';
          } else {
            card.className = 'result-not-sarcastic';
            emoji.textContent = '🙂';
            label.textContent = 'Not Sarcastic (Genuine)';
            label.style.color = '#4ade80';
            bar.style.backgroundColor = '#22c55e';
          }
          confidence.textContent = `Confidence: ${data.confidence}% (Sarcasm probability: ${data.probabilities.sarcastic}%)`;
          bar.style.width = data.confidence + '%';
        }
      } catch (err) {
        alert('Error connecting to backend API');
      } finally {
        btn.disabled = false;
        btn.textContent = 'Detect Sarcasm';
      }
    }

    document.getElementById('detectForm').addEventListener('submit', detectSarcasm);
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Sarcasm Detection API",
        "algorithm": "TF-IDF + Logistic Regression",
        "accuracy": "80.26%"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"success": False, "message": "Field 'text' is required"}), 400

    raw_text = data["text"]
    cleaned = clean_text(raw_text)

    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    pred = int(model.predict(vec)[0])

    sarcastic_prob = round(float(probs[1]) * 100, 2)
    not_sarcastic_prob = round(float(probs[0]) * 100, 2)
    confidence = sarcastic_prob if pred == 1 else not_sarcastic_prob

    return jsonify({
        "success": True,
        "text": raw_text,
        "is_sarcastic": bool(pred == 1),
        "label": "Sarcastic" if pred == 1 else "Not Sarcastic",
        "confidence": confidence,
        "probabilities": {
            "sarcastic": sarcastic_prob,
            "not_sarcastic": not_sarcastic_prob
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
