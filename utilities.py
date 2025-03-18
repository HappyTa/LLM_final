import torch
import torch.nn.functional as F
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from tqdm import tqdm


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name):
    """
    Loads the specified model and its tokenizer.

    Args:
        model_name (str): The model identifier from Hugging Face (e.g., "google/flan-t5-large").

    Returns:
        model: The loaded model.
        tokenizer: The associated tokenizer.
    """
    device = get_device()

    # Handle different model types
    if model_name == "google/flan-t5-large":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_name == "meta-llama/llama-2-7b-hf":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
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

    print(f"Loaded {model_name} on {device}")
    return model, tokenizer


def generate_response(model, tokenizer, claim):
    device = get_device()

    text = f"Is this claim true? {claim}"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def fact_checker(roberta_model, roberta_tn, claim, response):
    device = get_device()

    input = roberta_tn(claim, response, return_tensors="pt", truncation=True).to(device)
    output = roberta_model(**input)
    probs = F.softmax(output.logits, dim=-1)

    labels = ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]
    return labels[torch.argmax(probs)]


def evaluation_pipeline(model, tokenizer, dataset):
    print("Loading roberta-large-mnli to use a judge")

    roberta_name = "FacebookAI/roberta-large-mnli"
    roberta_model, roberta_tn = load_model(roberta_name)

    results = []
    for data in tqdm(dataset, desc="Fact-Checking"):
        claim = data["claim"]
        true_label = data["label"]

        model_response = generate_response(model, tokenizer, claim)

        verdict = fact_checker(roberta_model, roberta_tn, claim, model_response)

        results.append(
            {
                "Claim": claim,
                "LLM_Response": model_response,
                "Fact-Check Verdict": verdict,
                "FEVER Label": true_label,
            }
        )

    model_name = model.config.architectures
    df = pd.DataFrame(results)
    df.to_csv(f"{model_name}_fact_check_results.csv", index=False)
    print("Fact-checking complete! Results saved to fact_check_results.csv")
