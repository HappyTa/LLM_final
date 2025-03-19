from ctypes import ArgumentError
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.models.aria.modeling_aria import AriaGroupedExpertsGemm
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import os


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedding_model(selected_model=None):
    available_models = {
        "1": "sentence-transformers/all-MiniLM-L6-v2",
        "2": "sentence-transformers/all-mpnet-base-v2",
        "3": "sentence-transformers/sentence-t5-large",
    }

    if not selected_model:
        """Ask the user to select an embedding model and load it."""

        print("Available Models:")
        for key, model in available_models.items():
            print(f"{key}: {model}")

        # Ask user for selection
        choice = input("\nEnter the number of the model you want to use: ").strip()
    else:
        choice = selected_model

    if choice in available_models:
        return (SentenceTransformer(available_models[choice]), None)
    else:
        print("Unkown model selected, defaulting to all-MiniLM-L6-v2")
        return (SentenceTransformer(available_models["1"]), None)


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
    outputs = model.generate(**inputs, max_new_tokens=60)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def fact_checker(roberta_model, roberta_tn, claim, response):
    device = get_device()

    # Tokenize
    input = roberta_tn(claim, response, return_tensors="pt", truncation=True).to(device)
    output = roberta_model(**input)
    probs = F.softmax(output.logits, dim=-1)

    labels = ["REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]
    return labels[torch.argmax(probs)]


def compute_similarity(embed_model, reference_answer, generated_answer):
    device = get_device()
    ref_embedding = embed_model.encode(reference_answer, convert_to_tensor=True).to(
        device
    )
    gen_embedding = embed_model.encode(generated_answer, convert_to_tensor=True).to(
        device
    )
    similarity = util.pytorch_cos_sim(ref_embedding, gen_embedding).item()
    return similarity  # Between -1 and 1, where 1 means identical


def evaluation_pipeline(models_tn, dataset, d_type=0, embed_model=None):
    if not models_tn or not dataset:
        raise ArgumentError("Missing value for models_tn and dataset")

    if d_type == 0:
        print("Loading roberta-large-mnli to use a judge")
        eval_model, eval_tn = load_model("FacebookAI/roberta-large-mnli")
        dataset_name = "FEVER"
    elif d_type == 1:
        eval_model, eval_tn = load_embedding_model()
        dataset_name = "TruthfulQA"
    else:
        raise ArgumentError(
            "Unkown dataset selected, please only pass in valid datasets"
        )

    for model, tokenizer in models_tn:
        model_name = model.config._name_or_path.split("/")[-1]
        print(f"\nEvaluating {model_name}")

        if d_type == 0:
            results = fever_evaluator(model, tokenizer, eval_model, eval_tn, dataset)
        elif d_type == 1:
            results = truthful_evaluator(model, tokenizer, eval_model, dataset)
        else:
            raise ArgumentError(
                "Unkown dataset selected, please only pass in valid datasets"
            )

        # Create output folders if it does not exist
        if not os.path.exists("./evaluation_output"):
            os.makedirs("./evaluation_output")

        file_name = f"./evaluation_output/{model_name}_{dataset_name}_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(file_name, index=False)
        print(f"Evaluation complete for {model_name}! Results saved to {file_name}")


def truthful_evaluator(model, tokenizer, eval_model, dataset, similarity_threshold=0.7):
    model_name = model.config._name_or_path.split("/")[-1]
    results = []

    for data in tqdm(dataset, desc=f"Evaluating {model_name} on truthful_qa"):
        question = data["question"]
        true_answer = data["best_answer"]

        # Ask model for a response
        llm_response = generate_response(model, tokenizer, question)

        # Similarity-based evaluation for TruthfulQA
        similarity = compute_similarity(eval_model, true_answer, llm_response)
        is_correct = 1 if similarity >= similarity_threshold else 0

        # Store results
        results.append(
            {
                "Question": question,
                "LLM Answer": llm_response,
                "Actual Answer": true_answer,
                "Predicted Correct": is_correct,
            }
        )

    return results


def fever_evaluator(model, tokenizer, eval_model, eval_tn, dataset):
    results = []
    for data in tqdm(dataset, desc="Fact-Checking"):
        claim = data["claim"]
        true_label = data["label"]

        # Feed claim to model
        model_response = generate_response(model, tokenizer, claim)

        # Check to see if its true
        verdict = fact_checker(eval_model, eval_tn, claim, model_response)

        results.append(
            {
                "Claim": claim,
                "LLM_Response": model_response,
                "Fact-Check Verdict": verdict,
                "FEVER Label": true_label,
            }
        )

    return results
