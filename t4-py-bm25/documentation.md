# BM25.

Para el uso de este código y probar el algoritmo de BM25, se utilizará la librería de Python llamada **'rank_bm25'**, utilizando BM25Okapi.

```python
pip install rank_bm25
```

## Formula de BM25

$$
\text{BM25}(q, D) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

donde: - \( q \) es la consulta. - \( D \) es el documento. - \( qi \) es el término de la consulta.

- \( f(qi, D) \) es la frecuencia del término \( qi \) en el documento \( D \).
- \( |D| \) es la longitud del documento \( D \).
- \( \text{avgdl} \) es la longitud promedio de los documentos en la colección.
- \( k_1 \) y \( b \) son parámetros de ajuste. - \( IDF(q_i) \) es la frecuencia inversa de documentos del término \( q_i \), definida como:
  
  $$
  IDF(q_i) = \log \left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)
  $$
  
  donde:
- \( N \) es el número total de documentos.
- \( n(q_i) \) es el número de documentos que contienen el término \( q_i \).



## Explicación de código.

#### Importación de librerías.

```python
import os
from rank_bm25 import BM25Okapi
```

- `os`: Proporciona funciones para interactuar con el sistema operativo.
- `BM25Okapi`: Clase de la librería `rank_bm25` para implementar el algoritmo BM25.

#### Definir stop words.

```python
stop_words = {'y', 'el', 'la', 'en', 'de', 'que', 'a', 'los', 
                'del', 'se', 'con', 'las', 'por', 'un', 'para', 
                'una', 'es', 'al', 'lo', 'como', 'más', 'o', 'sus', 
                'le', 'ya', 'me'}
```

- Las stop words son palabras comunes que generalmente se eliminan en el procesamiento de texto porque no aportan mucho valor semántico.

#### **Definir el directorio de los archivos de texto.**

```python
directory = ''
```

- Aquí deberías especificar la ruta al directorio que contiene los archivos de texto que quieres procesar.

#### Leer los documentos desde archivos .txt

```python
documents = []
filenames = []
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            documents.append(file.read().strip())
            filenames.append(filename)
```

- Se recorre el directorio especificado y se leen todos los archivos `.txt`, almacenando su contenido en `documents` y sus nombres en `filenames`.

#### Verificar que se hayan leído los documentos.

```python
print(f"Documentos leídos: {len(documents)}")
```

- Se imprime la cantidad de documentos leídos para confirmar que se han cargado correctamente.

#### Tokenizar los documentos y eliminar stop words.

```python
tokenized_documents = [[word for word in doc.split() 
                        if word.lower() not in stop_words] 
                        for doc in documents]
```

- Se tokenizan los documentos, es decir, se dividen en palabras, y se eliminan las stop words.

#### Crear un objeto BM25.

```python
bm25 = BM25Okapi(tokenized_documents)
```

- Se crea un objeto BM25 utilizando los documentos tokenizados.

#### Definir la consulta.

```python
query = ""
```

Aquí puedes ingresar la consulta que deseas realizar. Debe estar relacionada con el contenido de los documentos.

#### **Tokenizar la consulta y eliminar stop words.**

```python
tokenized_query = [word for word in query.split() if word.lower()
                   not in stop_words]
```

#### Obtener los puntajes para la consulta.

```python
scores = bm25.get_scores(tokenized_query)
```

#### Imprimir los resultados con el nombre del archivo.

```python
for i, score in enumerate(scores):
    print(f"Documento: {filenames[i]} - Puntaje: {score:.3f}")
```

Se imprimen los puntajes de cada documento junto con el nombre del archivo correspondiente, mostrando qué tan relevante es cada documento para la consulta ingresada.


