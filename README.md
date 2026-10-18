# gemini-rag-knowledge-assistant

# 🧠 AI Knowledge Assistant — Gemini RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Gemini_API-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini">
  <img src="https://img.shields.io/badge/RAG-0A9396?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
</p>

## 🚀 About

An AI-powered document assistant that uses Retrieval-Augmented
Generation (RAG) to answer questions about PDF documents.

The application extracts PDF text, splits it into passages,
creates Gemini embeddings, retrieves relevant passages using
cosine similarity, and generates grounded answers with Gemini.

## ✨ Features

- PDF text extraction
- Text chunking with overlap
- Gemini text embeddings
- Semantic retrieval using cosine similarity
- Context-grounded AI responses
- Page references in generated answers
- Interactive Streamlit chatbot
- Retrieved passage inspection

## 🛠️ Tech Stack

Python, Google Gemini API, Google Gen AI SDK, Streamlit,
PyPDF, NumPy, python-dotenv, RAG, Embeddings.

## 📁 Project Structure

```text
gemini-rag-knowledge-assistant/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/gemini-rag-knowledge-assistant.git
cd gemini-rag-knowledge-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a local `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

Get an API key at https://aistudio.google.com/

Never commit your real API key.

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501.

## 🔄 How It Works

1. Upload a PDF.
2. Extract text and split it into passages.
3. Generate an embedding for each passage.
4. Embed the user's question.
5. Retrieve the most similar passages.
6. Send the retrieved context to Gemini.
7. Display the answer and relevant page references.

## 🔮 Future Improvements

- Persistent vector database
- Multi-document search
- Conversation memory
- OCR for scanned documents
- Source-quality evaluation
- User authentication
- Docker deployment

## ⚠️ Disclaimer

AI responses may contain errors. Verify important information
against the original document. Do not upload confidential
documents without authorization.

## 👨‍💻 Author

Mahesh Kolamkar

GitHub: https://github.com/maheshkgenaidevloper

---

⭐ Star the repository if you find it useful!
