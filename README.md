# 🧠 AI Text Summarizer using T5

A simple and interactive **AI-powered text summarizer** built with **Python**, **Streamlit**, and **Hugging Face Transformers**.  
This project leverages the **T5 (Text-to-Text Transfer Transformer)** model to generate concise and high-quality summaries from long text passages.  
The app can be accessed locally or shared online using **ngrok**.

---

## 🚀 Features
- 🧩 Summarize long articles, essays, or reports using the **T5 Transformer**  
- ⚡ Real-time summarization through an intuitive **Streamlit** interface  
- 🌐 Share your app publicly with a single command using **ngrok**  
- 💡 Easy to deploy, extend, and customize  
- 🪶 Lightweight and beginner-friendly implementation  

---

## 📦 Tech Stack
- **Python 3.9+**
- **Streamlit** – Web app framework  
- **Transformers (Hugging Face)** – NLP model library  
- **T5 Model** – Pre-trained text summarization model (`t5-small`, `t5-base`, etc.)  
- **Ngrok** – Local-to-public tunneling  

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Ebenezer/ai-summarizer.git
### 2. create environment
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Input text
text = "Artificial Intelligence is transforming industries by automating tasks..."

# Preprocess
input_text = "summarize: " + text
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

# Generate summary
summary_ids = model.generate(inputs, max_length=100, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Summary:", summary)

🧠 How It Works

User enters text through the Streamlit interface

The app preprocesses the text and sends it to the T5 Transformer model

The model generates a concise summary in natural language

Streamlit displays the result instantly on the UI


🤝 Contributing

Contributions are welcome!
If you’d like to improve this project:

Fork the repository

Create a new branch (git checkout -b feature-name)

Commit your changes (git commit -m "Add new feature")

Push to your fork and open a Pull Request

💬 Acknowledgments

Streamlit

Hugging Face Transformers

T5 Model Paper (Raffel et al., 2020)

Ngrok


👨‍💻 Author

Developed by ebenadex01-lab

🌟 If you find this project useful, please give it a star!

cd ai-summarizer

