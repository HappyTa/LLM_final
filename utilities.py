import torch
import torch.nn.functional as F
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from datasets import load_dataset
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

        print("\n\nPlease select an embedding model:")
        for key, model in available_models.items():
            print(f"{key}: {model}")

        # Ask user for selection
        choice = input("\nEnter the embedding model you want to use: ").strip()
    else:
        choice = selected_model

    if choice in available_models:
        return (SentenceTransformer(available_models[choice]), None)
    else:
        print("Unkown model selected, defaulting to all-MiniLM-L6-v2")
        return (SentenceTransformer(available_models["1"]), None)


def dataset_selector(dataset_in=None):
    """Ask the user to select a dataset and load it."""
    available_dataset = {"1": "Fever", "2": "TruthfulQA"}

    if not dataset_in:
        print("\n\nAvailable datasets")
        for key in available_dataset:
            print(f"{key}: {available_dataset[key]}")

        # Ask for a dataset to use
        choice = input("\nEnter the number of the dataset you want to use: ").strip()
    else:
        if not dataset_in.isdigit():
            raise ValueError("Please only use numerical values for dataset selections.")

        choice = dataset_in

    print(f"Loading {available_dataset[choice]}...")

    match choice:
        case "1":
            return (0, load_dataset("fever", "v2.0", split="validation"))
        case "2":
            return (1, load_dataset("truthful_qa", "generation", split="validation"))
        case _:
            print("Invalid choice. Using default dataset: Fever")
            return (0, load_dataset("fever", "v2.0", split="validation"))


def model_selector(model_in=None, padding=None):
    available_models = {
        "1": "google/flan-t5-large",
        "2": "meta-llama/Meta-Llama-3-8B",
        "3": "openai-community/gpt2",
        "4": "All",
    }

    if not model_in:
        """Ask the user to select an LLM model and load it."""

        print("\n\nAvailable Models:")
        for key, model in available_models.items():
            print(f"{key}: {model}")

        # Ask user for selection
        choice = input("\nPlease select a model you want to use: ").strip()
    else:
        choice = model_in

    if choice not in available_models:
        print("Invalid choice. Using default model: google/flan-t5-large")
        model_name = "google/flan-t5-large"
    elif choice == "4":
        rtn_list = []
        for key in available_models:
            if key != "4":
                rtn_list.append(load_model(available_models[key]))
            else:
                return rtn_list
    else:
        model_name = available_models[choice]
        print(f"Loading {model_name}...")

    # Load the chosen model
    return [load_model(model_name)]  # type: ignore


def process_truthfulqa(dataset):
    formatted_data = []
    for item in dataset["validation"]:
        question = item["question"]
        correct_answers = item["best_answer"]

        # Choose the first best answer
        response = (
            correct_answers if isinstance(correct_answers, str) else correct_answers[0]
        )

        formatted_data.append(
            {
                "instruction": "Answer the following question truthfully:",
                "input": f"{question}",
                "output": response,
            }
        )

    return formatted_data


def process_fever(dataset):
    formatted_data = []
    for item in dataset["train"]:
        claim = item["claim"]
        label = item["label"]

        formatted_data.append(
            {
                "instruction": "Verify the truthfulness of the given claim using the provided evidence.",
                "input": f"Claim: {claim}\nEvidence: Not Provided",
                "output": label,  # "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"
            }
        )

    return formatted_data


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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Handle different model types
    if model_name == "google/flan-t5-large":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    elif model_name == "meta-llama/Meta-Llama-3-8B":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        # Padding to supress warning
        tokenizer.pad_token = tokenizer.eos_token

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


def generate_response(model, tokenizer, claim, d_type=0):
    device = get_device()

    # Does not need to add question when testing with TruthfulQA (1)
    if d_type == 0:
        text = f"Is this claim true? {claim}"
    else:
        text = claim
        if type(text) is not str:
            print(f"{type(text)}: {text}")

    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs, max_new_tokens=60, pad_token_id=model.config.eos_token_id
    )
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
    """Evaluates multiple models on either the FEVER or TruthfulQA dataset."""

    if not models_tn or not dataset:
        raise ValueError("Missing value for models_tn and dataset")

    # Select evaluation method based on dataset type
    dataset_map = {
        0: ("FEVER", "FacebookAI/roberta-large-mnli", fever_evaluator),
        1: ("TruthfulQA", embed_model, truthful_evaluator),
    }

    if d_type not in dataset_map:
        raise ValueError(
            "Unknown dataset selected, please pass a valid dataset type (0 or 1)"
        )

    dataset_name, eval_model_name, evaluator = dataset_map[d_type]

    # Load the appropriate evaluation model
    eval_model, eval_tn = (
        load_model(eval_model_name)
        if d_type == 0
        else load_embedding_model(embed_model)
    )

    # Ensure output directory exists
    os.makedirs("./evaluation_output", exist_ok=True)

    # Evaluate each model

    print("\nStart Evaluation")
    for model, tokenizer in models_tn:
        model_name = model.config._name_or_path.split("/")[-1]
        results = evaluator(model, tokenizer, model_name, eval_model, eval_tn, dataset)

        # Save results
        file_name = f"./evaluation_output/{model_name}_{dataset_name}_results.csv"
        pd.DataFrame(results).to_csv(file_name, index=False)
        print(f"Evaluation complete for {model_name}! Results saved to {file_name}\n")


def truthful_evaluator(
    model, tokenizer, model_name, eval_model, eval_tn, dataset, similarity_threshold=0.7
):
    """
    Evaluates a language model on the TruthfulQA dataset using semantic similarity.

    Args:
        model (PreTrainedModel): The language model to evaluate.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        model_name (str): Name of the model
        eval_model (SentenceTransformer): Model for computing similarity.
        dataset (list[dict]): List of questions and ground truth answers.
        similarity_threshold (float, optional): Minimum similarity for correctness (default: 0.7).

    Returns:
        list[dict]: Evaluation results with question, model response, true answer, and correctness.
    """

    results = []

    for data in tqdm(dataset, desc=f"Evaluating {model_name} on truthful_qa"):
        question = data["question"]
        true_answer = data["best_answer"]

        # Ask model for a response
        llm_response = generate_response(model, tokenizer, question, 1)

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


def fever_evaluator(model, tokenizer, model_name, eval_model, eval_tn, dataset):
    """
    Evaluates a language model on the FEVER dataset using a fact-checking pipeline.

    Args:
        model (PreTrainedModel): The language model to evaluate.
        tokenizer (PreTrainedTokenizer): Tokenizer for the model.
        model_name (str): Name of the evaluated model.
        eval_model (Any): Model used for semantic similarity or fact verification.
        eval_tn (Any): Additional evaluation tool (e.g., retrieval model or external knowledge base).
        dataset (list[dict]): List of claims with corresponding FEVER labels.

    Returns:
        list[dict]: Evaluation results including the claim, model response, fact-check verdict,
        and ground truth FEVER label.
    """

    results = []
    for data in tqdm(dataset, desc=f"Evaluating {model_name} on FEVER"):
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
