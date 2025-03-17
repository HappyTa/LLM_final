import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
)


def load_model(model_name):
    """
    Loads the specified model and its tokenizer.

    Args:
        model_name (str): The model identifier from Hugging Face (e.g., "google/flan-t5-large").

    Returns:
        model: The loaded model.
        tokenizer: The associated tokenizer.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle different model types
    if model_name == "google/flan-t5-large":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_name == "meta-llama/llama-2-7b-hf":
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_name == "FacebookAI/roberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_name == "openai-community/gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        raise ValueError(f"Model {model_name} is not supported in this function.")

    print(f"✅ Loaded {model_name} on {device}")
    return model, tokenizer


def fact_checking_pipeline(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
