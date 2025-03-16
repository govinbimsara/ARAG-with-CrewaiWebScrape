import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from agent import graph

# Set up the Streamlit app UI
st.title("🔎 Genie Business Assistant")
st.write("Ask any question, and I'll try to answer it!")

st.sidebar.title("Settings")
session_id = st.sidebar.text_input("Session ID", value="default_session")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store previous messages

# Display chat history
for message in st.session_state.messages:
    role = "🤖 AI" if isinstance(message, AIMessage) else "🧑 You"
    st.markdown(f"**{role}:** {message.content}")

# Input text box for the question
question = st.text_input("Enter your question here:")

# Button to get the answer
if st.button("Get Answer"):
    if question:
        # Append user input to chat history
        st.session_state.messages.append(HumanMessage(content=question))

        # Prepare inputs
        inputs = {"messages": st.session_state.messages}
        config = {"configurable": {"thread_id": session_id}}

        # Process the response
        outputs = list(graph.stream(inputs, config=config, stream_mode="values"))
        if outputs:
            last_output = outputs[-1]
            response = last_output["messages"][-1].content
            st.session_state.messages.append(AIMessage(content=response))  # Store AI response

        # Refresh the page to display the chat history
        st.rerun()
    else:
        st.warning("Please enter a question to get an answer.")

# Additional styling (optional)
st.markdown("---")
st.caption("Powered by Streamlit and LangChain")
