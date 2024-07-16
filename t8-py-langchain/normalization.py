import os
import re
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Función para normalizar texto
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

# Función para eliminar líneas específicas y vacías
def deleteSpacesAndSpecific(text, patron):
    regex_patron = re.compile(patron)
    lineas = text.split('\n')
    lineas_filtradas = [linea for linea in lineas if linea.strip() != '' and not regex_patron.match(linea.strip())]
    return '\n'.join(lineas_filtradas)

# Directorio con archivos de texto
input_directory = 'C:/Users/luisv/OneDrive/Documentos/Verano Cientifico/Progrmacion/t8-py-langchain/txt'
output_directory = 'C:/Users/luisv/OneDrive/Documentos/Verano Cientifico/Progrmacion/t8-py-langchain/normalization'


# Crear el directorio de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Patrón para identificar las líneas que deben ser eliminadas
patron = r'Página \| \d+'

# Procesar cada archivo en el directorio
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
