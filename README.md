# IELTS Writing Task 2 Scoring API

This project is a **Flask-based API** for evaluating IELTS Writing Task 2 essays.
It automatically scores essays across IELTS bands, corrects grammar, and generates model answers using **NLP (Transformers, LanguageTool, and Scikit-learn)**.

---

## ✨ Features

* **Automatic Scoring**: Provides IELTS band scores (Task Response, Coherence & Cohesion, Lexical Resource, Grammar).
* **Grammar Correction**: Uses `language_tool_python` for grammar feedback.
* **Model Answer Generation**: Suggests a sample answer based on the essay prompt.
* **API Endpoint**: Simple JSON REST API with `/evaluate`.
* **Docker Support**: Easy to run in any environment.

---

## 📂 Project Structure

```
.
├── ai.py                # Core scoring logic (NLP, scoring, correction, answer generation)
├── app.py               # Flask API endpoints
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker build file
├── docker-compose.yml   # Docker Compose setup
└── README.md            # Project documentation
```

---

## 🚀 Installation & Usage

### 1. Clone Repository

```bash
git clone https://github.com/<username>/ielts-writing-task2-scoring.git
cd ielts-writing-task2-scoring
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
python app.py
```

API will be available at:
👉 `http://127.0.0.1:5000/evaluate`

### 4. Example API Request

```bash
curl -X POST http://127.0.0.1:5000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Some people think children should do housework. Discuss both views and give your opinion.",
    "essay": "Many people argue that children should do housework because..."
  }'
```

Example JSON Response:

```json
{
  "task_response": 6.5,
  "coherence": 6.0,
  "lexical_resource": 6.5,
  "grammar_score": 6.0,
  "overall_band": 6.5,
  "corrected_essay": "...",
  "model_answer": "...",
  "grammar_errors": [...]
}
```

---

## 🐳 Run with Docker

### Build Docker Image

```bash
docker build -t ielts-scoring .
```

### Run Container

```bash
docker run -p 5000:5000 ielts-scoring
```

### Using Docker Compose

```bash
docker-compose up
```

---

## 🛠 Tech Stack

* **Python 3.10+**
* **Flask** (API framework)
* **Transformers** (DistilBERT embeddings for semantic similarity)
* **Torch** (PyTorch backend)
* **LanguageTool** (grammar checking)
* **Scikit-learn, Pandas, NumPy** (NLP utilities, metrics)
* **Docker** (deployment ready)

---

## 📜 License

MIT License. Free to use, modify, and share.

---

## 🙌 Author

Developed by [Hoàng](https://github.com/<your-username>) as a project to explore **NLP + Automated Essay Scoring**.
