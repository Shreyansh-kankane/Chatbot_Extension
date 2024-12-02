import os
from fastapi import FastAPI, HTTPException , UploadFile, File, Form
import numpy as np
from pydantic import BaseModel
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer
from functools import lru_cache
# from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import chromadb
import PyPDF2
# from langchain.prompts import PromptTemplate
# from langchain_cohere.llms import Cohere
# from openai import OpenAI

import uvicorn

from fastapi.middleware.cors import CORSMiddleware

from pinecone import Pinecone, ServerlessSpec


import google.generativeai as genai

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])



# Initialize Chroma Cloud client
chroma_client = chromadb.Client()


# RAG class definition
class RAG:
    def _init_(self, namespace="testing", model="gpt-4o-mini"):
        # Initialize the language model
        self.llm = ChatOpenAI(model=model, api_key="OPENAI_APIKEY")
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

# Lazy loading and caching of RAG instances
@lru_cache(maxsize=100)  # Cache up to 100 RAG instances
def get_rag_instance(namespace: str):
    """Lazy-load the RAG instance, caching the most recently used instances."""
    return RAG(namespace=namespace)

# FastAPI app initialization
app = FastAPI()

origins = [
    "http://localhost:8000",  # React development server
    "https://pec.ac.in",      # PEC website or other allowed domains
    "*"                       # Allow all origins (use with caution in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,              # List of allowed origins
    allow_credentials=True,             # Allow cookies or authentication headers
    allow_methods=["*"],                # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                # Allow all headers
)


# Model for the request to ask questions
class QueryRequest(BaseModel):
    question: str

# Model for the initialization request
class InitRequest(BaseModel):
    domain: str

# Mapping of website codes to namespaces (to be populated during setup)

# Mapping of domain names to namespaces (e.g., example.com -> namespace1)
# We store them on registering as a Website owner
domain_namespace = {} 

class CustomEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, input):
        # Assuming input is a list of strings to be embedded
        return self.model.encode(input, convert_to_tensor=True)
    
    def embed_query(self, input):
        # Assuming input is a list of strings to be embedded
        return self.model.encode(input, convert_to_tensor=True)


# Endpoint to initialize RAG based on a code (namespace)
@app.post("/init-rag")
async def init_rag(request: InitRequest):
    try:
        namespace = domain_namespace.get(request.domain)  # Retrieve the namespace using the provided code
        if not namespace:
            return {"error": "Invalid Code"}

        # Lazily load or get the cached RAG instance
        get_rag_instance(namespace)

        return {"message": f"RAG initialized for namespace: {namespace}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Endpoint to ask questions and get answers from RAG
@app.post("/query")
async def ask_question(domain: str, query: QueryRequest):
    try:
        namespace , api_key = domain_namespace.get(domain)
        if not namespace:
            raise HTTPException(status_code=404, detail="Namespace not found")

       
        # Initialize the RAG instance for the specified namespace
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        custom_embeddings = CustomEmbeddingFunction(embeddings_model)
        
        db3 = Chroma(persist_directory=f"./data/{namespace}" , embedding_function=custom_embeddings)
        
        context = db3.similarity_search(query.question)
        
        if not context:
            return {"question": query.question, "answer": "No relevant information found."}
        


        # Use OpenAI or another model to generate a structured answer based on the context
        answer = generate_response(context, query.question , api_key)
        
        return {"question": query.question, "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
def health_check():
    return {"health": "ok"}  


@app.post('/createEmbeddings')
async def create_embeddings(namespace: str = Form(...), file: UploadFile = File(...)):
    pdf_reader = PyPDF2.PdfReader(file.file)
    print(os.environ.get("PINECONE_API_KEY"))
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    # print(os.getenv("OPENAI_API_KEY"))
    for key, value in os.environ.items():
        print(f"{key}: {value}")
    file_text = ""
    for page in pdf_reader.pages:
        file_text += page.extract_text()

    # Create embeddings using OpenAI
    print("Text Data-----------------" , file_text)
    try:
        
        # embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        # embeddings = embeddings_model.embed_documents([file_text])
        
        
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("embed",embeddings_model)
        # custom_embeddings = CustomEmbeddingFunction(embeddings_model)
        
        # persist_directory =  f'./data/{namespace}'
        
        # vector_store = Chroma(persist_directory=persist_directory, embedding_function=custom_embeddings)

        # # Add the document to the vector store
        # vector_store.add_texts(texts=[file_text], ids=[namespace])
        
        # print("Created embeddings at /data/", namespace)
        # return {"message": "Embeddings created and stored successfully "}
        
        chunk_size = 500  # Adjust as needed for optimal input
        chunks = [file_text[i:i + chunk_size] for i in range(0, len(file_text), chunk_size)]

        # Generate embeddings for each chunk
        chunk_embeddings = [embeddings_model.encode(chunk) for chunk in chunks]
    
        index_name = namespace
        embedding_dimension = 384  # Fixed size for 'all-MiniLM-L6-v2'

        if index_name not in pc.list_indexes():
            pc.create_index(name=index_name, dimension=embedding_dimension , metric='euclidean',
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ))
        index = pc.Index(index_name)
        # Add chunks to Pinecone index with metadata
        vectors = [
            (f"{namespace}_chunk_{i}", embedding, {"text": chunk_text})  # Include metadata as a dictionary
            for i, (embedding, chunk_text) in enumerate(zip(chunk_embeddings, chunks))
        ]

        # Perform the upsert operation to insert vectors along with their metadata
        index.upsert(vectors)

        print(f"Created embeddings in Pinecone under index: {namespace}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

def generate_response(context: str, query: str , api_key : str) -> str:
    print("c:",context,"q:",query)
    # response = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a knowledgeable programming assistant."},
    #         {
    #             "role": "user",
    #             "content": (
    #                 f"Context: {context}\n"
    #                 f"Question: {query}"
    #             ),
    #         },
    #     ],
    # )

    system_prompt = """
    You will be given a piece of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro",system_instruction=system_prompt)

    user_prompt = f"""
    {context}
    Question: {query}
    Helpful Answer:"""

    response = model.generate_content(
        user_prompt,
        generation_config = genai.GenerationConfig(
            max_output_tokens=1000,
            temperature=0.1,
        )
    )

    # Access the generated answer
    print(response.text)
    return response.text


    
@app.post("/test_query")
async def ask_question(namespace: str, query: QueryRequest):
    
    try:
        # Initialize the RAG instance for the specified namespace
        embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        custom_embeddings = CustomEmbeddingFunction(embeddings_model)
        
        
        # db3 = Chroma(persist_directory=f"./data/{namespace}" , embedding_function=custom_embeddings)
        
        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        
        index_name = namespace
        index = pc.Index(index_name)
        
        print("index_name------" , index_name)
        
        query_embedding = custom_embeddings.embed_query([query.question])[0]
        
        # print("query_embedding-----------------------" , query_embedding)
        
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()  # Convert numpy array to list of floats
        elif hasattr(query_embedding, 'detach'):  # Check if it's a tensor (PyTorch)
            query_embedding = query_embedding.detach().numpy().tolist()
        
        
        query_result = index.query(
            vector=query_embedding,
            top_k=5,  # Number of similar results you want
            include_metadata=True,
        )
        
        print("query_result-----------"  , query_result)
        
        
        # context = db3.similarity_search(query.question)
        # context = "E-dukkan is a E-commerce Platform"
        # print("context---------------" , context)
        
        matches = query_result['matches']
        
        print(matches)
        
        if not matches:
            return {"question": query.question, "answer": "No relevant information found."}
        
        
        # if not context:
        #     return {"question": query.question, "answer": "No relevant information found."}
        

        # Combine the results into one string for clarity
        # context = " ".join(result.page_content for result in context)

        # Use OpenAI or another model to generate a structured answer based on the context
        
        context = " ".join([match['metadata']['text'] for match in matches])  # Assuming 'text' is stored in metadata
        
        answer = generate_response(context, query.question , os.environ.get("GOOGLE_API_KEY"))

        # Return the question and generated answer
        return {"question": query.question, "answer": answer}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
   
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)

    
# Run the application with: uvicorn filename:app --reload