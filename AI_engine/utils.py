from pydantic import BaseModel
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from functools import lru_cache
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
import PyPDF2
import os 
from fastapi import FastAPI, HTTPException , UploadFile, File, Form

load_dotenv()


# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize Chroma Cloud client
chroma_client = chromadb.Client()


# RAG class definition
class RAG:
    def __init__(self, namespace, model="gpt-4o-mini"):
        # Initialize the language model
        self.llm = ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
        # Load the vector store for the specific website (namespace)
        self.vectorstore = chroma_client.get_or_create_collection(name=namespace)
        

        self.retriever = self.vectorstore.as_retriever()
        self.prompt = hub.pull("rlm/rag-prompt")

        # RAG chain definition
        self.rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(self, question):
        return self.rag_chain.invoke(question)

class QueryRequest(BaseModel):
    question: str

app = FastAPI()
@app.post('/createEmbeddings')
async def create_embeddings(namespace: str = Form(...), file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(file.file)
    
    # print(os.getenv("OPENAI_API_KEY"))
    # for key, value in os.environ.items():
    #     print(f"{key}: {value}")
    file_text = ""
    for page in pdf_reader.pages:
        file_text += page.extract_text()

    # Create embeddings using OpenAI
    print(file_text)
    try:
        
        embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = embeddings_model.embed_documents([file_text])

        # Create or load a Chroma vector store for the given namespace
        vector_store = Chroma(
            persist_directory=f"./data/{namespace}",  # Namespace-based storage
            embedding_function=embeddings_model
        )

        # Add embeddings to the vector store
        vector_store.add_texts([file_text], embeddings)

        # Persist the vector store for later use
        vector_store.persist()

        return {"message": "Embeddings created and stored successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/test_query")
async def ask_question(namespace : str  , query: QueryRequest):
    try:
        rag_instance = RAG(namespace)

        # Generate the answer using the RAG engine
        answer = rag_instance.generate_answer(query.question)
        return {"question": query.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
