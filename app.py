import os
from groq import Groq
import streamlit as st

# Initialize Groq client with your API key
api_key = "YOUR_GROQ_API_KEY"  # Replace with your actual Groq API key
client = Groq(api_key=api_key)

# Function to generate a response using the Groq API
def generate_response(user_input):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant knowledgeable about Arabic literature."
            },
            {
                "role": "user",
                "content": user_input,
            }
        ],
        model="llama3-8b-8192",  # You can change this to another model if needed
    )
    
    return chat_completion.choices[0].message.content

# Streamlit interface setup
def main():
    st.title("Arabic Book Chatbot")
    st.write("Ask questions about Arabic books and literature!")

    user_input = st.text_input("Your question:")
    
    if st.button("Submit"):
        if user_input:
            response = generate_response(user_input)
            st.write("Bot:", response)
        else:
            st.write("Please enter a question.")

if __name__ == "__main__":
    main()
