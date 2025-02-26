from functools import lru_cache
from utils.tools import tools
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def extract_latest_messages(messages):
    question = None
    docs = None

    for msg in reversed(messages):  # Start from the most recent message
        if hasattr(msg, "type") and hasattr(msg, "content"):  # Ensure valid object
            if msg.type == "human" and question is None:
                question = msg.content
            elif msg.type == "tool" and docs is None:
                docs = msg.content
            
            if question and docs:
                break  # Stop once both are found

    return question, docs


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "secondary":
        model = ChatGroq(model=os.getenv("LLM_MODEL", "llama3-70b-8192"), api_key=groq_api_key, streaming=True)
    elif model_name == "primary":
        model =  ChatGroq(model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"), api_key=groq_api_key)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    return model


def grade_documents(state, config) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """
    print("---CHECK RELEVANCE---")
    # print("STATE:", state)
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: Literal["yes","no"] = Field(description="Relevance score 'yes' or 'no'")

    model = ChatGroq(model=os.getenv("LLM_MODEL", "llama3-70b-8192"), api_key=groq_api_key, streaming=True)

    llm_with_tools = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    #chain
    chain = prompt|llm_with_tools

    messages = state["messages"]
    # docs = messages[-1].content
    # question = messages[0].content
    question, docs = extract_latest_messages(messages)

    score_results = chain.invoke({"question": question,"context": docs})

    score = score_results.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    

def agent(state, config):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    # print("STATE:", state)
    messages = state["messages"]
    # question = messages[0].content
    # question, docs = extract_latest_messages(messages)

    model = ChatGroq(model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"), api_key=groq_api_key)

    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages":messages + [response]}

def rewrite(state, config):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    # print("STATE:", state)
    messages = state["messages"]
    # question = messages[0].content
    question, docs = extract_latest_messages(messages)

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    model = ChatGroq(model=os.getenv("LLM_MODEL", "llama3-70b-8192"), api_key=groq_api_key, streaming=True)
    
    response = model.invoke(msg)
    return {"messages":messages + [HumanMessage(content=response)]}


def generate(state, config):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    # print("STATE:", state)
    messages = state["messages"]
    # question = messages[0].content
    # docs = messages[-1].content
    question, docs = extract_latest_messages(messages)

    model = ChatGroq(model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"), api_key=groq_api_key)

    generate_prompt = PromptTemplate(
        template="""You are a customer service assistant for 'Genie Business' called 'Ashen' for question-answering tasks. \n
        You are always giving answers to a merchant who is looking to get information. \n
        Use the following pieces of retrieved context to answer the question. \n
        If you don't know the answer, just say that you don't know. \n
        Use simpler and keep the answer short but detailed and answer concise. \n
        Here is the context: \n\n {context} \n\n
        Here is the user question: {question} \n""",
        input_variables=["context", "question"],
    )
    # print("---post processing docs---")
    # Post-processing
    def format_docs(docs):
        if isinstance(docs, str):  
            return docs  # Return directly if it's already a string
        return "\n\n".join(doc.page_content for doc in docs)
    
    docs = format_docs(docs)
    # print("---generating output---")
        
    rag_chain = generate_prompt | model | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})

    return {"messages": messages + [AIMessage(content=response)]}

retrieve = ToolNode(tools)


