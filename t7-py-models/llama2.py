import torch
from transformers import AutoTokenizer, pipeline
from huggingface_hub import login

# Inicia sesión en Hugging Face
#login(token="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")

model = "meta-llama/Llama-2-7b-hf"

# Carga el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

messages = [
    {"role": "user", "content": "Create a story that describes the meaning of love"},
]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

outputs = pipeline(
    prompt,
    max_new_tokens=256,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

print(outputs[0]["generated_text"][len(prompt):])