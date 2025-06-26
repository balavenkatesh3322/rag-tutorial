from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from typing import List

# Set environment variables (replace with your own or use dotenv in practice)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"


class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, documents: List[Document]):
        return self.embedding_model.embed_documents([doc.page_content for doc in documents])


class VectorDB:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.db = None

    def build(self, documents: List[Document]):
        self.db = FAISS.from_documents(documents, self.embedder.embedding_model)

    def save_local(self, path: str):
        if self.db:
            self.db.save_local(path)

    def load_local(self, path: str):
        self.db = FAISS.load_local(path, self.embedder.embedding_model)

    def get_retriever(self):
        return self.db.as_retriever()


class RAGPipeline:
    def __init__(self, retriever):
        self.llm = OpenAI(temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True
        )

    def query(self, question: str):
        return self.qa_chain({"query": question})


class RAGSystem:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = TextLoader(file_path)
        self.chunker = DocumentChunker()
        self.embedder = Embedder()
        self.vector_db = VectorDB(self.embedder)

    def setup(self):
        documents = self.loader.load()
        chunks = self.chunker.chunk(documents)
        self.vector_db.build(chunks)

    def run_query(self, question: str):
        retriever = self.vector_db.get_retriever()
        pipeline = RAGPipeline(retriever)
        return pipeline.query(question)


if __name__ == '__main__':
    # Example usage
    rag = RAGSystem(file_path="sample_data.txt")  # You need to provide sample_data.txt
    rag.setup()
    result = rag.run_query("What is the content about?")
    print("Answer:", result['result'])
    print("Sources:", result['source_documents'])
