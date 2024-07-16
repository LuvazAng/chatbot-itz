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
    return words

# Directorio con archivos de texto
directory = 'C:/Users/luisv/OneDrive/Documentos/Verano Cientifico/Progrmacion/t8-py-langchain/normalization'  # Asegúrate de que el directorio sea correcto

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

print(f"Documentos leídos: {len(documents)}")

# Crear el modelo BM25
bm25 = BM25Okapi(documents)

query = "Dime donde se encuentra la universidad"
# Normalizar y tokenizar consulta
tokenized_query = normalize_text(query)

result = bm25.get_top_n(tokenized_query, documents, n=1)
print("Resultado de la consulta:", result)

scores = bm25.get_scores(tokenized_query)

# Combinar nombres de archivos y puntajes
documents_with_scores = list(zip(filenames, scores))

# Ordenar documentos por puntaje en orden descendente
documents_with_scores.sort(key=lambda x: x[1], reverse=True)

# Número de documentos a devolver
n_top_documents = 3

# Obtener los top n documentos
top_documents = documents_with_scores[:n_top_documents]

# Imprimir los top n documentos
for filename, score in top_documents:
    print(f"{filename}: {score:.3f}")
