# PRUEBAS DE MODELOS LLM.

Para las pruebas de IA, se realizaron instalaciones de los modelos desde la plataforma de Hugging Face. Esta plataforma nos permitió realizar la descarga de cada uno de los modelos, como Llama2 7B, Gemma2 9B, Mistral 7B y Phi3 13B.

Para el uso de estos modelos, se siguieron los siguientes pasos para la descarga y uso de cada uno de ellos.

## Pasos para descargar y ejecutar los modelos LLM.

1. Crear una cuenta en Hugging Face.

2. Una vez creada la cuenta, ir a: Settings > Access Tokens > New Token. Al crear un nuevo token, se deberá elegir el tipo de token como "Write". Una vez ingresado el nombre y seleccionado el tipo, se hará clic en "Generate a token".

3. Una vez obtenido el token, se deberán instalar las siguientes librerías para poder utilizar los modelos.

```python
pip install torch
pip install transformers
pip install huggingface_hub
```

- `torch.` Esta es la biblioteca principal para trabajar con modelos de PyTorch. Necesaria para cargar, entrenar e inferir modelos de aprendizaje profundo. 

- `transformers.` Proporciona acceso a una amplia variedad de modelos preentrenados para tareas de NLP, como GPT, BERT, y más. También incluye utilidades para la tokenización y la creación de pipelines.

- `huggingface_hub.`Permite interactuar con el Hugging Face Hub, donde puedes descargar modelos preentrenados y subir tus propios modelos.

#### NOTA.

Si cuentas con tarjeta gráfica, corrobora que tengas el controlador CUDA. Esta herramienta nos permitirá hacer uso de la GPU.

Para corroborar si tienes CUDA en tu tarjeta gráfica, corre este script en Python.

```python
import torch

if torch.cuda.is_available():
    print("CUDA está disponible y PyTorch puede usar la GPU.")
else:
    print("CUDA no está disponible o PyTorch no puede usar la GPU.")
```

#### NOTA 2.

Es posible que alguno de los modelos te pida instalar otras librerías. Estas son las posibles librerías que pueden llegar a pedir.

```python
pip install sentencepiece
pip install tokenizers
pip install accelerate
```

- `sentencepiece.` Un modelo de tokenización que es utilizado por algunos modelos preentrenados (especialmente los de Google, como BERT). Necesario si el modelo que estás utilizando requiere tokenización basada en SentencePiece.

- `tokenizers.` Proporciona una implementación rápida de varios algoritmos de tokenización. Es una alternativa a los tokenizadores de `transformers` para mejorar el rendimiento.

- `accelerate.` Para acelerar el entrenamiento y la inferencia en múltiples GPUs o en entornos distribuidos.

#### NOTA 3.

Es importate que si al instalar pytorch no sale con `torch    2.3.1+cu118` o variantes de `+cuXXX`, este no podrá funcionar con tu GPU, es importante saber si la GPU que se esta utilizando, podrá ejecutar la biblioteca CUDA.

Para conocer que versión de Pytorch tienes instalada, con ejecutar el comando `pip list` podrás corroborarlo así como todas las librerías que hayas instalado.

# 4. EXPLICACIÓN DE CÓDIGO.

Cómo se te menciono previamente, antes de ejecutar el código, debes instalar las siguientes librerías.

```powershell
pip install torch transformers huggingface_hub
```

Se importarán las librerías que se utilizarán en el código.

```python
import torch
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
```

- `torch`: Librería para el cálculo numérico y aprendizaje profundo.
- `AutoTokenizer` y `pipeline` de `transformers`: Utilizados para el procesamiento de texto y generación de texto respectivamente.
- `login` de `huggingface_hub`: Utilizado para autenticarse en Hugging Face.

#### Inicio de sesión de Hugging Face.

```python
# Inicia sesión en Hugging Face
login(token="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")
```

- Autentica el usuario en Hugging Face utilizando un token de acceso.

#### Cargar el modelo y el tokenizador.

```python
model = "google/gemma-2b-it" #Aquí tu podrás elegir el de preferencia.

# Carga el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model)
```

- Define el modelo a utilizar (`google/gemma-2b-it`).
- Carga el tokenizador del modelo especificado.

#### Configurar el Pipeline de generación de texto.

```python
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
```

- Configura el pipeline de generación de texto utilizando el modelo especificado.
- `torch_dtype` se establece en `torch.bfloat16` para usar el tipo de dato bfloat16, lo que puede mejorar el rendimiento en ciertos GPU.
- `device` se establece en `cuda` para usar la GPU si está disponible.

#### Definir mensajes y preparar el prompt

```python
messages = [
    {"role": "user", "content": "Make me a simple code in Java"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
```

- Define una lista de mensajes, en este caso, un único mensaje donde el usuario pide una explicación del código.
- `apply_chat_template` del tokenizador aplica una plantilla de chat a los mensajes, creando un prompt adecuado para la generación de texto.

#### Generar texto.

```python
outputs = pipeline(
    prompt,
    max_new_tokens=350,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
```

- Utiliza el pipeline de generación de texto para generar una respuesta basada en el prompt.
- `max_new_tokens=350`: Especifica el número máximo de nuevos tokens a generar.
- `add_special_tokens=True`: Añade tokens especiales si es necesario.
- `do_sample=True`: Habilita la generación de texto mediante muestreo.
- `temperature=0.7`: Controla la aleatoriedad del muestreo (valores más bajos hacen que la salida sea menos aleatoria).
- `top_k=50` y `top_p=0.95`: Configuran técnicas de muestreo (top-k y top-p) para controlar la calidad y diversidad de la generación de texto.

#### Imprimir texto generado.

```python
print(outputs[0]["generated_text"][len(prompt):])
```

Claro, aquí tienes una explicación detallada del código proporcionado, incluyendo las librerías necesarias y su instalación.

### Instalación de Librerías

Antes de ejecutar el código, debes instalar las siguientes librerías:

```bash
pip install torch transformers huggingface_hub
```

### Explicación del Código

```python
import torch
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login
```

- `torch`: Librería para el cálculo numérico y aprendizaje profundo.
- `AutoTokenizer` y `pipeline` de `transformers`: Utilizados para el procesamiento de texto y generación de texto respectivamente.
- `login` de `huggingface_hub`: Utilizado para autenticarse en Hugging Face.

### Inicio de sesión en Hugging Face

```python
# Inicia sesión en Hugging Face
login(token="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")
```

- Autentica el usuario en Hugging Face utilizando un token de acceso.

### Cargar el modelo y el tokenizador

```python
model = "google/gemma-2b-it" # Elegirás tu modelo de preferencia

# Carga el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model)
```

- Define el modelo a utilizar (`google/gemma-2b-it`).
- Carga el tokenizador del modelo especificado.

### Configurar el Pipeline de generación de texto

```python
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)
```

- Configura el pipeline de generación de texto utilizando el modelo especificado.
- `torch_dtype` se establece en `torch.bfloat16` para usar el tipo de dato bfloat16, lo que puede mejorar el rendimiento en ciertos GPU.
- `device` se establece en `cuda` para usar la GPU si está disponible.

### Definir mensajes y preparar el prompt

```python
messages = [
    {"role": "user", "content": "Aquí ingresarás el prompt"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)
```

- Define una lista de mensajes, en este caso, un único mensaje donde el usuario pide una explicación del código.
- `apply_chat_template` del tokenizador aplica una plantilla de chat a los mensajes, creando un prompt adecuado para la generación de texto.

### Generar texto

```python
outputs = pipeline(
    prompt,
    max_new_tokens=350,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)
```

Utiliza el pipeline de generación de texto para generar una respuesta basada en el prompt.

- `max_new_tokens=350`: Especifica el número máximo de nuevos tokens a generar.
- `add_special_tokens=True`: Añade tokens especiales si es necesario.
- `do_sample=True`: Habilita la generación de texto mediante muestreo.
- `temperature=0.7`: Controla la aleatoriedad del muestreo (valores más bajos hacen que la salida sea menos aleatoria).
- `top_k=50` y `top_p=0.95`: Configuran técnicas de muestreo (top-k y top-p) para controlar la calidad y diversidad de la generación de texto.

### Imprimir el texto generado

```python
print(outputs[0]["generated_text"][len(prompt):])
```

- Imprime el texto generado, omitiendo la parte del prompt.

Este código se conecta a Hugging Face, carga un modelo y un tokenizador, y utiliza un pipeline de generación de texto para generar una respuesta a un mensaje dado. La respuesta se genera utilizando técnicas de muestreo para asegurar calidad y diversidad en el texto generado.
