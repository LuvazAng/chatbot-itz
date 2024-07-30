from flask import Flask, request, jsonify, render_template
import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode
import torch
from transformers import AutoTokenizer, pipeline

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Función para normalizar texto
def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = unidecode(text)
    words = nltk.word_tokenize(text, language='spanish')
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Carga de documentos
directory = 'C:/Users/darkd/Desktop/UIChatbot/normalization'
documents = []
filenames = []

# Leer y normalizar documentos
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            normalized_text = normalize_text(text)
            documents.append(normalized_text)
            filenames.append(filename)

# Tokenización de documentos
tokenized_documents = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# Carga de Gemma2
model = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Consulta BM25
    normalized_query = normalize_text(query)
    tokenized_query = normalized_query.split()
    scores = bm25.get_scores(tokenized_query)

    # Combinar nombres de archivos y puntajes
    documents_with_scores = list(zip(filenames, scores))
    documents_with_scores.sort(key=lambda x: x[1], reverse=True)
    n_top_documents = 1
    top_documents = documents_with_scores[:n_top_documents]

    if not top_documents:
        return jsonify({'error': 'No documents found'}), 404

    filename, score = top_documents[0]
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
        doc_content = file.read().strip()

    # Extraer los fragmentos más relevantes
    sentences = nltk.sent_tokenize(doc_content, language='spanish')
    relevant_sentences = [sentence for sentence in sentences if any(word in sentence.lower() for word in normalized_query.split())]
    relevant_text = ' '.join(relevant_sentences) if relevant_sentences else doc_content

    messages = [
        {"role": "user", "content": f"Respuesta a la consulta '{query}': {relevant_text}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = gemma_pipeline(
        prompt, max_new_tokens=256, 
        add_special_tokens=True, 
        do_sample=True, temperature=0.7, 
        top_k=50, top_p=0.95)
    
    summary = outputs[0]["generated_text"][len(prompt):]
    
    return jsonify({
        'query': query,
        'document': filename,
        'score': score,
        'summary': summary
    })

if __name__ == '__main__':
    app.run(debug=True)
