# Normalización.

Para obtener mejores resultados con el uso de BM25, se creó un programa en Python que normaliza el texto, es decir, solo conservará la información importante. Se deberán seguir las siguientes reglas:

- No debe contener signos de puntuación.
- No debe contener tildes.
- Eliminación de stop words (de, la, el, etc.).
- Texto en minúsculas.

Primero para todo este proceso se tendrán que instalar las siguientes librerías.

```powershell
pip install nltk unidecode
```

## Explicación del código.

###### NORMALIZACIÓN DE TEXTO.

Se creó un script en Python llamado `normalization.py`

```python
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')
```

- `os`: Proporciona funciones para interactuar con el sistema operativo.
- `re`: Proporciona funciones para trabajar con expresiones regulares.
- `nltk`: Natural Language Toolkit, utilizado para el procesamiento del lenguaje natural.
- `string`: Proporciona constantes y clases para la manipulación de cadenas.
- `stopwords`: Contiene listas de stop words para diferentes idiomas.
- `unidecode`: Permite la transliteración de caracteres Unicode a ASCII.

#### Función para normalizar texto

```python
def normalizeText(text):
    # Reemplazar caracteres especiales por espacios
    text = text.replace('/', ' ').replace('-', ' ').replace('_', ' ')
    # Remover URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
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
```

- `normalizeText`: Normaliza el texto eliminando caracteres especiales, URLs, puntuaciones, convirtiendo a minúsculas, removiendo tildes, tokenizando y eliminando stop words

#### Función para eliminar líneas específicas y vacías.

```python
def deleteSpacesAndSpecific(text, patron):
    regex_patron = re.compile(patron)
    lineas = text.split('\n')
    lineas_filtradas = [linea for linea in lineas if linea.strip() != '' and not regex_patron.match(linea.strip())]
    return '\n'.join(lineas_filtradas)
```

- `deleteSpacesAndSpecific`: Elimina líneas específicas y líneas vacías según el patrón proporcionado.

#### Directorio de archivos de texto.

```python
input_directory = 'directorio/donde/estan/los/txt'
output_directory = 'directorio/donde/se/guardarán'
```

- `input_directory`: Directorio donde se encuentran los archivos de texto.
- `output_directory`: Directorio donde se guardarán los archivos normalizados.

#### Crear el directorio de salida si no existe.

```python
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
```

- Crea el directorio de salida si no existe.

#### Patrón para identificar líneas a eliminar.

```python
patron = r'Página \| \d+'
```

- Define el patrón de las líneas que deben ser eliminadas.

#### Procesar cada archivo en el directorio.

```python
for filename in os.listdir(input_directory):
    if filename.endswith('.txt'):
        input_filepath = os.path.join(input_directory, filename)
        
        # Leer el archivo con manejo de errores de codificación
        with open(input_filepath, 'rb') as file:
            raw_data = file.read()
        
        # Intentar decodificar el contenido en UTF-8, ISO-8859-1, etc.
        try:
            text = raw_data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = raw_data.decode('iso-8859-1')
            except UnicodeDecodeError:
                print(f"No se pudo decodificar el archivo {filename}")
                continue
        
        # Eliminar líneas específicas y vacías
        text_limpio = deleteSpacesAndSpecific(text, patron)
        
        # Normalizar el texto
        normalized_text = normalizeText(text_limpio)
        
        # Crear el nuevo nombre de archivo
        base_filename = os.path.splitext(filename)[0]
        output_filename = base_filename + '-NOR.txt'
        output_filepath = os.path.join(output_directory, output_filename)
        
        # Guardar el texto normalizado en el nuevo archivo
        with open(output_filepath, 'w', encoding='utf-8') as file:
            file.write(normalized_text)

print("Normalización completada.")
```

- Recorre cada archivo en el directorio de entrada.
- Lee el archivo manejando posibles errores de codificación.
- Elimina líneas específicas y vacías.
- Normaliza el texto.
- Guarda el texto normalizado en un nuevo archivo en el directorio de salida.

##### NORMALIZACIÓN DE QUERY.

Para el proceso de normalización, también se aplicará al query donde se utiliza el algoritmo `BM25`. Este script permitirá tener el texto y el query normalizados y obtener mejores resultados.

```powershell
pip install nltk unidecode rank_bm25
```

Se creó un script en Python llamado `main.py`

```python
import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')
```

- `os`: Proporciona funciones para interactuar con el sistema operativo.
- `BM25Okapi`: Clase de la librería `rank_bm25` para implementar el algoritmo BM25.
- `nltk`: Natural Language Toolkit, utilizado para el procesamiento del lenguaje natural.
- `string`: Proporciona constantes y clases para la manipulación de cadenas.
- `stopwords`: Contiene listas de stop words para diferentes idiomas.
- `unidecode`: Permite la transliteración de caracteres Unicode a ASCII.

#### Función para normalizar texto.

```python
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
```

- `normalize_text`: Normaliza el texto eliminando puntuaciones, convirtiendo a minúsculas, removiendo tildes, tokenizando y eliminando stop words.

#### Directorio de archivos de texto.

```python
directory = 'directorio/donde/estan/los/txt/normalizados'
```

- `directory`: Especifica la ruta al directorio que contiene los archivos de texto que quieres procesar.

#### Leer documentos.

```python
documents = []
filenames = []
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            documents.append(normalize_text(file.read().strip()))
        filenames.append(filename)
            
print(f"Documentos leídos: {len(documents)}")
```

- Lee y normaliza los documentos en el directorio especificado, almacenando el contenido normalizado en `documents` y los nombres de los archivos en `filenames`.

#### Tokenizar documentos.

```python
tokenized_documents = [doc.split() for doc in documents]
```

- Tokeniza los documentos, dividiéndolos en palabras.

#### Crear objeto BM25.

```python
bm25 = BM25Okapi(tokenized_documents)
```

- Crea un objeto BM25 utilizando los documentos tokenizados.

#### Definir y normalizar la consulta.

```python
query = "sistemas computacionales"
# Normalizar y tokenizar consulta
normalized_query = normalize_text(query)
tokenized_query = normalized_query.split()
print(tokenized_query)
```

- Define la consulta, la normaliza y la tokeniza.

#### Obtener resultados.

```python
result = bm25.get_top_n(tokenized_query, documents, n=1)
print(result)

scores = bm25.get_scores(tokenized_query)
```

- `get_top_n`: Obtiene el documento más relevante para la consulta.
- `get_scores`: Obtiene los puntajes de relevancia para la consulta en cada documento.

#### Combinar nombres de archivos y puntajes.

```python
# Combinar nombres de archivos y puntajes
documents_with_scores = list(zip(filenames, scores))

# Ordenar documentos por puntaje en orden descendente
documents_with_scores.sort(key=lambda x: x[1], reverse=True)
```

- Combina los nombres de los archivos con sus puntajes y los ordena en orden descendente de relevancia.

#### Imprimir los top N documentos.

```python
# Número de documentos a devolver
n_top_documents = 5

# Obtener los top n documentos
top_documents = documents_with_scores[:n_top_documents]

# Imprimir los top n documentos
for filename, score in top_documents:
    print(f"{filename}: {score:.3f}")
```

- Imprime los `n` documentos más relevantes junto con sus puntajes.

Este código procesa archivos de texto, normaliza el contenido, evalúa la relevancia de cada documento respecto a una consulta dada utilizando el algoritmo BM25, y muestra los documentos más relevantes.


