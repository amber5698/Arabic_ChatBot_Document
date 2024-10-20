import os
from groq import Groq
import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np
#GROQ_API_KEY='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'

# Initialize Groq client
client = Groq(
    api_key='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',  # Replace with your actual key
)

# Load an Arabic sentence transformer model for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample Arabic book data (replace this with your actual data)
arabic_books = [
    {"title": "كتاب 1", "content": "محتوى الكتاب الأول."},
    {"title": "كتاب 2", "content": "محتوى الكتاب الثاني."},
    # Add more books as needed
]

# Create embeddings for the book contents
book_embeddings = []
for book in arabic_books:
    embedding = embedding_model.encode(book['content'])
    book_embeddings.append(embedding)

# Function to find the most relevant book based on user query
def retrieve_relevant_book(user_query):
    query_embedding = embedding_model.encode(user_query)
    
    # Calculate cosine similarities
    similarities = np.dot(book_embeddings, query_embedding)
    
    # Get the index of the most similar book
    best_match_index = np.argmax(similarities)
    return arabic_books[best_match_index]

# Function to generate a response using Groq API
def generate_response(user_query):
    relevant_book = retrieve_relevant_book(user_query)
    
    # Create a message for the LLM
    message = f"Based on the book titled '{relevant_book['title']}', here is some information: {relevant_book['content']}. Now, can you answer this question: {user_query}?"
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="llama3-8b-8192",
    )
    
    return chat_completion.choices[0].message.content

# Gradio interface function
def chatbot_interface(user_input):
    response = generate_response(user_input)
    return response

# Set up Gradio interface
iface = gr.Interface(
    fn=chatbot_interface,
    inputs="text",
    outputs="text",
    title="Arabic Book Chatbot",
    description="Ask questions about Arabic books and get responses."
)

# Launch the Gradio app
iface.launch()
