from typing import Literal, TypedDict
from utils.state import AgentState
from langgraph.graph import END, StateGraph, START
from utils.nodes import grade_documents,agent,rewrite,generate,retrieve
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import uuid

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["primary", "secondary"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)

workflow.add_node("agent", agent)
workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

#Start node
workflow.add_edge(START,"agent")
#Edges
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("rewrite","agent")
#End node
workflow.add_edge("generate",END)
#Compile
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

if __name__=="__main__":
    thread_id = str(uuid.uuid4())
    config = {
    "configurable": {
        "thread_id": thread_id
        }
    }
    inputs = {
    "messages": [HumanMessage(role="user", content="Can you tell me about Genie Business?")]
    }
    # for output in graph.stream(inputs,config=config,stream_mode="values"):
    #     output["messages"][-1].pretty_print()
    outputs = list(graph.stream(inputs, config=config, stream_mode="values"))  # Collect all outputs
    if outputs:
        last_output = outputs[-1]  # Get the last output
        print(last_output["messages"][-1].content)