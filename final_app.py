import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # For saving and loading the model

# Set up SSL for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("intents_dataset_aicte.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Paths for saving the model and vectorizer
model_path = "chatbot_model.joblib"
vectorizer_path = "vectorizer.joblib"

# Function to train and save the model
def train_and_save_model():
    tags = []
    patterns = []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)
    
    vectorizer = TfidfVectorizer()
    clf = LogisticRegression(random_state=0, max_iter=10000)
    
    x = vectorizer.fit_transform(patterns)
    y = tags
    clf.fit(x, y)
    
    # Save the model and vectorizer
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    return clf, vectorizer

# Load the model and vectorizer, or train and save if they don't exist
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    clf, vectorizer = train_and_save_model()

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

def main():
    st.title("Chatbot")

    # Sidebar menu
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Initialize chat log file if not present
    if not os.path.exists('chat_log.csv'):
        with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

    if choice == "Home":
        st.write("Welcome to the chatbot. Start chatting below. Type 'bye' to end the conversation.")

        # Conversation history to be displayed dynamically
        if "conversation" not in st.session_state:
            st.session_state.conversation = []

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input("You:", key="user_input")
            submitted = st.form_submit_button("Send")

            if submitted and user_input:
                # Process chatbot response
                response = chatbot(user_input)

                # Log the conversation
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow([user_input, response, timestamp])

                # Update session state with the conversation
                st.session_state.conversation.append(("You", user_input))
                st.session_state.conversation.append(("Chatbot", response))

                # End chat if "bye" or "goodbye" is detected
                if user_input.lower() in ['bye', 'goodbye']:
                    st.write("Thank you for chatting with me. Have a great day!")
                    st.stop()

        # Display conversation history dynamically
        for sender, message in st.session_state.conversation:
            if sender == "You":
                st.text(f"You: {message}")
            else:
                st.text(f"Chatbot: {message}")

    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history available.")

    elif choice == "About":
        st.subheader("About the Chatbot")
        st.write("""
        This chatbot is built using Natural Language Processing (NLP) and Logistic Regression. 
        It understands user intents and provides appropriate responses. 
        The chatbot is designed for an interactive experience using the Streamlit framework.
        """)

if __name__ == '__main__':
    main()
