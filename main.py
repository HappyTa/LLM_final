from utilities import load_model, evaluation_pipeline
from datasets import load_dataset
import sys


def model_selector():
    """Ask the user to select an LLM model and load it."""
    available_models = {
        "1": "google/flan-t5-large",
        "2": "meta-llama/Llama-2-7b-hf",
        "3": "facebook/roberta-large-mnli",
        "4": "openai-community/gpt2",
    }

    print("Available Models:")
    for key, model in available_models.items():
        print(f"{key}: {model}")

    # Ask user for selection
    choice = input("\nEnter the number of the model you want to use: ").strip()

    if choice not in available_models:
        print("Invalid choice. Using default model: google/flan-t5-large")
        model_name = "google/flan-t5-large"
    else:
        model_name = available_models[choice]
        print(f"Loading {model_name}...")

    # Load the chosen model
    return load_model(model_name)


def dataset_selector():
    """Ask the user to select a dataset and load it."""
    available_dataset = {
        "1": "Fever",
        "2": "TruthfulQA",
    }

    print("Available datasets")
    for key, dataset in available_dataset:
        print(f"{key}. {dataset}")

    choice = input("\nEnter the number of the dataset you want to use: ").strip()

    print(f"Loading {available_dataset[choice]}...")

    match choice:
        case "1":
            return load_dataset("fever", "v2.0", split="validation")
        case "2":
            return load_dataset("truthful_qa", split="validation")
        case _:
            print("Invalid choice. Using default dataset: Fever")
            return load_dataset("fever", "v2.0", split="validation")


def main():
    # Select model
    model, tokenizer = model_selector()

    # grab datasets
    dataset = dataset_selector()

    run_fever = input("Would you like to validates the models on Fever dataset? (y/N)")
    if run_fever.lower() == "y":
        # run validation
        evaluation_pipeline(model, tokenizer, dataset)
        pass

    # close program
    sys.exit(0)


if __name__ == "__main__":
    main()
