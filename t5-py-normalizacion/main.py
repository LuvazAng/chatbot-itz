import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Función para normalizar texto
def normalize_text(text):
    # Remover puntuaciones
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convertir a minúsculas
    text = text.lower()
    # Remover tildes
    text = unidecode(text)
    # Tokenizar el texto
    words = nltk.word_tokenize(text, language='spanish')
    # Remover stop words
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Directorio con archivos de texto
#directory = 'C:/Users/luisv/OneDrive/Documentos/Verano Cientifico/Progrmacion/t5-py-normalizacion/normalization'  # Cambia esto al directorio correcto
directory = 'D:/Documentos/Verano Cientifico/Programación/t5-py-normalizacion/normalization'

documents = []
filenames = []
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            documents.append(normalize_text(file.read().strip()))
        filenames.append(filename)
            
print(f"Documentos leídos: {len(documents)}")

# Tokenizar documentos y eliminar stop words
tokenized_documents = [doc.split() for doc in documents]

bm25 = BM25Okapi(tokenized_documents)

query = "sistemas computacionales"
# Normalizar y tokenizar consulta
normalized_query = normalize_text(query)
tokenized_query = normalized_query.split()
print(tokenized_query)

result = bm25.get_top_n(tokenized_query, documents, n=1)
print(result)

scores = bm25.get_scores(tokenized_query)

# Combinar nombres de archivos y puntajes
documents_with_scores = list(zip(filenames, scores))

# Ordenar documentos por puntaje en orden descendente
documents_with_scores.sort(key=lambda x: x[1], reverse=True)

# Número de documentos a devolver
n_top_documents = 5

# Obtener los top n documentos
top_documents = documents_with_scores[:n_top_documents]

# Imprimir los top n documentos
for filename, score in top_documents:
    print(f"{filename}: {score:.3f}")
