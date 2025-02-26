from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embeddings_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.load_local("/workspaces/ARAG-with-CrewaiWebScrape/faiss_index", embeddings_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_genie_business_context",
    "Search and return context information about, 'genie business', payment solutions, fintech, tap to pay, qr online transactions and business related information"
)

tools = [retriever_tool]