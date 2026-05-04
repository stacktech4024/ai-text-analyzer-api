# AI Text Analyzer API 🤖📝

A FastAPI + LangChain backend that analyses any text and returns:

| Feature | Endpoint |
|---|---|
| Summary · Key Insights · Sentiment | `POST /analyze` |
| Auto-generated Caption + Sentiment | `POST /analyze-caption` |
| Relevant Tags / Hashtags | `POST /generate-tags` |

> **Use case:** auto-generate captions and tags for Top Recruit Tapes or any video/sports content.

---

## Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** – modern async web framework
- **[LangChain](https://python.langchain.com/)** – LLM orchestration (LCEL chains)
- **[OpenAI](https://platform.openai.com/)** – GPT model backend (gpt-3.5-turbo by default)
- **Python 3.12**

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/stacktech4024/ai-text-analyzer-api.git
cd ai-text-analyzer-api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Reference

### `GET /`

Health check.

```json
{ "status": "ok", "message": "AI Text Analyzer API is running." }
```

---

### `POST /analyze`

Analyze text and return summary, insights, and sentiment.

**Request:**
```json
{
  "text": "LeBron James delivered another MVP-caliber performance tonight..."
}
```

**Response:**
```json
{
  "summary": "LeBron scored 40 points leading the Lakers to a playoff win.",
  "insights": [
    "MVP-level performance",
    "40-point scoring output",
    "Crucial playoff victory secured"
  ],
  "sentiment": "positive"
}
```

---

### `POST /analyze-caption`

Generate an engaging caption for the text, plus summary and sentiment.

**Request:**
```json
{ "text": "..." }
```

**Response:**
```json
{
  "caption": "King James takes over! 👑",
  "summary": "LeBron James scored 40 points in a critical Lakers playoff win.",
  "sentiment": "positive"
}
```

---

### `POST /generate-tags`

Generate relevant hashtags / tags for the text.

**Request:**
```json
{ "text": "..." }
```

**Response:**
```json
{
  "tags": ["lebron", "lakers", "nba", "playoffs", "mvp", "basketball"]
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
ai-text-analyzer-api/
├── app/
│   ├── main.py          # FastAPI app + routes
│   ├── models.py        # Pydantic request/response models
│   └── services/
│       └── analyzer.py  # LangChain analysis service
├── tests/
│   └── test_main.py     # pytest test suite (mocked LLM)
├── .env.example
├── pytest.ini
└── requirements.txt
```

---

## Concepts Covered

- What an API is (request → processing → response)
- FastAPI routing, Pydantic validation, dependency injection
- LangChain LCEL chains and prompt templates
- How AI (LLMs) fits into real backend applications
