import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import customtkinter as ctk

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    intents = json.load(file)

patterns = []
tags = []
corpus = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        patterns.append(' '.join(words))
        tags.append(intent['tag'])
        corpus.append((words, intent['tag']))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

def predict_class(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    sentence_str = ' '.join(sentence_words)
    
    sentence_vec = vectorizer.transform([sentence_str])
    similarities = cosine_similarity(sentence_vec, X)
    index = np.argmax(similarities)
    tag = tags[index]
    
    return tag

def get_response(tag, intents_json):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def chatbot_response(text):
    tag = predict_class(text)
    response = get_response(tag, intents)
    return response

# GUI Code using customtkinter
def send():
    user_input = entry.get()
    entry.delete(0, ctk.END)
    
    if user_input.lower() == "exit":
        root.destroy()
        return
    
    chat_window.configure(state="normal")
    chat_window.insert(ctk.END, "You: " + user_input + '\n')
    
    response = chatbot_response(user_input)
    chat_window.insert(ctk.END, "Bot: " + response + '\n')
    chat_window.configure(state="disabled")

# Setting up the GUI
root = ctk.CTk()
root.title("Chatbot")

ctk.set_appearance_mode("dark")  # Modes: "system" (default), "light", "dark"
ctk.set_default_color_theme("blue")  # Themes: "blue" (default), "green", "dark-blue"

chat_window = ctk.CTkTextbox(root, width=500, height=400, state="disabled", wrap="word")
chat_window.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

entry = ctk.CTkEntry(root, width=400)
entry.grid(row=1, column=0, padx=10, pady=10)

send_button = ctk.CTkButton(root, text="Send", command=send)
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
