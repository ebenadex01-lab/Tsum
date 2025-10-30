

from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
import spacy
import streamlit as st
from transformers import pipeline

model_name = "t5-small"
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def summarize_text(text):
    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=600, truncation=True)
    summary_ids = model.generate(input_ids, max_length=500, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



def answer_question(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=600, truncation=True)
    answer_ids = model.generate(inputs, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)




def nlp_agent():
    print("Hello! I am your NLP Agent.")
    print("You can ask me to summarize text or answer questions.")

    context_memory = []  # Optional: store previous conversation/context

    while True:
        user_input = input("\nWhat can I do for you? (type 'exit' to quit): ").strip()
        task = detect_intent(user_input)

        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            context_memory = []
            print("Memory cleared.")
            continue

        elif task == "summarize":
            text = input("Enter the text to summarize:\n")
            summary = summarize_text(text)
            print("\nSummary:\n", summary)
            context_memory.append({"task": "summarize", "input": text, "output": summary})

        elif task == "question":
            context = input("Enter the context text:\n")
            question = input("Enter your question:\n")
            answer = answer_question(question, context)
            print("\nAnswer:\n", answer)
            context_memory.append({"task": "question", "context": context, "question": question, "answer": answer})

        else:
            print("I'm not sure how to handle that request. Please ask me to 'summarize' or ask a 'question'.")




def detect_intent(user_input):
    if "summarize" in user_input.lower():
        return "summarize"
    elif "?" in user_input:
        return "question"
    else:
        return "unknown"




def answer_question(question):
    global context_memory
    input_text = f"question: {question} context: {context_memory}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    answer_ids = model.generate(inputs["input_ids"], max_length=50)
    return tokenizer.decode(answer_ids[0], skip_special_tokens=True)



@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    qa_model = pipeline("question-answering", model="t5-small", tokenizer="t5-small")
    return summarizer, qa_model

summarizer, qa_model = load_models()

st.title("🤖 Tsum")

task = st.radio("Choose Task:", ["Summarization", "Question Answering"])

if task == "Summarization":
    text_input = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if text_input:
            summary = summarizer(text_input, max_length=500, min_length=100, do_sample=False)
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])
        else:
            st.warning("Please enter some text.")

elif task == "Question Answering":
    context_input = st.text_area("Enter context text:")
    question_input = st.text_input("Enter your question:")
    if st.button("Answer"):
        if context_input and question_input:
            answer = qa_model(question=question_input, context=context_input)
            st.subheader("Answer:")
            st.write(answer['answer'])
        else:
            st.warning("Please provide both context and question.")

