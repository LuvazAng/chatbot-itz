import os
from rank_bm25 import BM25Okapi

# Lista de stop words
stop_words = {'y', 'el', 'la', 'en', 'de', 'que', 'a',
              'los', 'del', 'se', 'con', 'las', 'por', 
              'un', 'para', 'una', 'es', 'al', 'lo', 
              'como', 'más', 'o', 'sus', 'le', 'ya', 
              'me'}

# Directorio que contiene los archivos de texto
directory = '' 

# Leer los documentos desde archivos .txt
documents = []
filenames = []
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read().strip())
            filenames.append(filename)

# Verificar que se hayan leído los documentos
print(f"Documentos leídos: {len(documents)}")

# Tokenizar los documentos y eliminar stop words
tokenized_documents = [[word for word in doc.split() if word.lower() not in stop_words] for doc in documents]

# Crear un objeto BM25
bm25 = BM25Okapi(tokenized_documents)

# Definir la consulta
# Ingresaras el texto que desees (de preferencia relacionado con el/los textos a analizar)
query = "" 
# Tokenizar la consulta y eliminar stop words
tokenized_query = [word for word in query.split() if word.lower() not in stop_words]

# Obtener los puntajes para la consulta
scores = bm25.get_scores(tokenized_query)

# Imprimir los resultados con el nombre del archivo
for i, score in enumerate(scores):
    print(f"Documento: {filenames[i]} - Puntaje: {score:.3f}")
