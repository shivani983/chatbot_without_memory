import os
import warnings
import streamlit as st
from langchain._api import LangChainDeprecationWarning
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

def main():
    # Suppress warnings related to LangChain deprecation
    warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

    # Load environment variables
    _ = load_dotenv(find_dotenv())
    groq_api_key = os.environ.get("GROQ_API_KEY")

    # If the API key is not found, raise an error
    if not groq_api_key:
        st.error("GROQ_API_KEY not set in .env")
        st.stop()

    # Initialize the Groq chatbot
    chatbot = ChatGroq(model="llama3-8b-8192")

    # Streamlit UI
    st.set_page_config(page_title="Groq Chatbot", page_icon="🤖", layout="wide")

    # Custom Styling
    st.markdown("""
        <style>
        .header {
            font-size: 2em;
            color: #4CAF50;
            text-align: center;
            font-weight: bold;
        }
        .description {
            font-size: 1.2em;
            color: #555555;
            text-align: center;
        }
        .chatbox {
            background-color: #f5f5f5;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            color: #1d72b8;
            font-weight: bold;
        }
        .bot-message {
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    # UI Section - Title & Description
    st.markdown("<div class='header'>Chatbot with Groq Model without memory</div>", unsafe_allow_html=True)
    st.markdown("<div class='description'>Enter a message below to interact with the chatbot. It will respond based on the model's capabilities.</div>", unsafe_allow_html=True)

    # Add a chatbot avatar (Optional)
    st.sidebar.image("chatbot_picwebp.webp", width=100, use_container_width=True)

    # Input message box for the user
    st.markdown("<div class='chatbox'>", unsafe_allow_html=True)
    user_message = st.text_area("Your message:", "", key="user_message", height=100)

    # When the button is pressed
    if st.button("Send", key="send_button"):
        if user_message.strip() != "":
            try:
                # Prepare the input message for the chatbot
                messages_to_the_chatbot = [HumanMessage(content=user_message)]

                # Get the response from the chatbot
                response = chatbot.invoke(messages_to_the_chatbot)

                # Display the response with custom formatting
                st.markdown(f"<div class='user-message'><h1>Let me help you with this.</h1></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bot-message'> {response.content}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a message.")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
