# RAG System Tutorial using LangChain

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline using Python OOP concepts and LangChain. It walks you through setting up a document-based QA system using FAISS for vector storage and OpenAI LLM for answering questions.

## 📁 Project Structure

```
.
├── start_rag_here.py       # Main RAG pipeline implementation
├── sample_data.txt         # Input text file used for ingestion
└── README.md               # Documentation and usage guide
```

## 🧠 Components

### 1. **DocumentChunker**

Splits large documents into manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.

### 2. **Embedder**

Uses `HuggingFaceEmbeddings` (e.g., `all-MiniLM-L6-v2`) to convert text chunks into dense vectors.

### 3. **VectorDB**

Uses FAISS to index embeddings and store them efficiently.

### 4. **Retriever**

Fetches relevant document chunks using vector similarity search.

### 5. **RAGPipeline**

Combines a retriever with OpenAI's GPT model to provide answers and source references.

### 6. **RAGSystem**

Orchestrates the end-to-end workflow: loading, chunking, embedding, storing, retrieving, and querying.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
# If stored in Git repo
git clone <repo_url>
cd <repo_name>
```

### 2. Install Dependencies

```bash
pip install langchain faiss-cpu openai sentence-transformers
```

### 3. Prepare Environment Variables

```bash
export OPENAI_API_KEY=your-openai-api-key
```

Alternatively, you can use `.env` file and `dotenv` to manage secrets.

### 4. Add Sample Data

Place your raw document in `sample_data.txt` in the root directory. Example:

```
LangChain is a framework for developing applications powered by language models.
```

### 5. Run the Application

```bash
python start_rag_here.py
```

You should see an answer printed along with source documents.

---

## 📌 Notes

- This is a basic RAG setup; you can extend it using LangChain’s advanced retrievers, rerankers, or LangGraph.
- You can save/load FAISS index using `VectorDB.save_local()` and `load_local()`.

---

## 📈 Future Enhancements

- Add PDF/CSV/URL loaders.
- Metadata-based filtering.
- Use LangGraph for stateful RAG workflows.
- Integrate caching and rate limiters.

---

## 🧠 Credits

Built with 💡 using:

- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [OpenAI](https://openai.com)
- [Sentence Transformers](https://www.sbert.net/)

---

## 📬 Feedback

Feel free to reach out or fork the project for improvements!

