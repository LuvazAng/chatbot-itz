# INTEGRACIÓN BM25 + LLM (Gemma)

Para la integración del algoritmo BM25 y el modelo LLM Gemma de Google, se realizó una integración de códigos y la adaptación de estos dos. Es decir, se ejecutará el algoritmo BM25 para entregarle los datos obtenidos al modelo LLM, de esta manera, podrá generar una respuesta más precisa.

## EXPLICACIÓN DE CÓDIGO.

Este código realiza la normalización de texto, consulta de documentos utilizando BM25 y genera resúmenes de los documentos clave usando el modelo Gemma2.

### Importaciones de bibliotecas.

```python
import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode
import torch
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
```

Importa las bibliotecas necesarias, incluyendo `os` para manipulación de archivos, `rank_bm25` para la búsqueda, `nltk` para procesamiento de lenguaje natural, `string` para manipulación de texto, `unidecode` para normalización de texto, `torch` para PyTorch, y `transformers` para modelos de HuggingFace.

### Descarga de recursos de NLTK.

```python
nltk.download('stopwords')
nltk.download('punkt')
```

Descarga los recursos necesarios de NLTK para trabajar con palabras vacías y tokenización.

### Función para documentar texto.

```python
def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = unidecode(text)
    words = nltk.word_tokenize(text, language='spanish')
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
```

Elimina puntuación, convierte el texto a minúsculas, elimina acentos, tokeniza el texto y elimina las palabras vacías en español.

### Cargar y normalización de documentos.

```python
directory = 'C:/Users/luisv/OneDrive/Documentos/Verano Cientifico/Progrmacion/t8-py-langchain/normalization'
documents = []
filenames = []

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read().strip()
            normalized_text = normalize_text(text)
            documents.append(normalized_text)
            filenames.append(filename)
```

Lee todos los archivos de texto en el directorio especificado, los normaliza y guarda los textos y nombres de archivo.

### Tokenización de documentos y creación del índice BM25.

```python
tokenized_documents = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_documents)
```

### Consulta BM25.

```python
query = "Dime donde se encuentra la universidad"
normalized_query = normalize_text(query)
tokenized_query = normalized_query.split()
scores = bm25.get_scores(tokenized_query)

documents_with_scores = list(zip(filenames, scores))
documents_with_scores.sort(key=lambda x: x[1], reverse=True)
n_top_documents = 3
top_documents = documents_with_scores[:n_top_documents]
```

Normaliza y tokeniza la consulta, calcula los puntajes de relevancia para cada documento y selecciona los tres documentos más relevantes.

### Carga del modelo Gemma y configuración del pipeline.

```python
model = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
```

Carga el modelo Gemma y configura el pipeline para generación de texto.

### Generación de resúmenes para los documentos más relevantes.

```python
for filename, score in top_documents:
    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
        doc_content = file.read().strip()

    messages = [
        {"role": "user", "content": f"Resumen del documento sobre {query}: {doc_content}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = gemma_pipeline(
        prompt, max_new_tokens=256, 
        add_special_tokens=True, 
        do_sample=True, temperature=0.7, 
        top_k=50, top_p=0.95)
    
    summary = outputs[0]["generated_text"][len(prompt):]
    print("--------------------------------------------------------------------------------------------------------------")
    print(f"Resumen del documento {filename}")
    print("--------------------------------------------------------------------------------------------------------------")
    print(summary)
```

Para cada uno de los documentos más relevantes, lee su contenido, crea un mensaje de entrada para el modelo, genera un resumen utilizando el pipeline Gemma2, y muestra el resumen generado.



Este flujo de trabajo permite buscar documentos relevantes utilizando BM25 y generar resúmenes de esos documentos utilizando un modelo de lenguaje avanzado, proporcionando así una herramienta poderosa para la recuperación y resumen de información.
