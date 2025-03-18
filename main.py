from utilities import load_model, evaluation_pipeline
from datasets import load_dataset
import sys


def model_selector():
    """Ask the user to select an LLM model and load it."""
    available_models = {
        "1": "google/flan-t5-large",
        "2": "meta-llama/llama-2-7b-hf",
        "3": "openai-community/gpt2",
        "4": "All",
    }

    print("Available Models:")
    for key, model in available_models.items():
        print(f"{key}: {model}")

    # Ask user for selection
    choice = input("\nEnter the number of the model you want to use: ").strip()

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


def dataset_selector():
    """Ask the user to select a dataset and load it."""
    available_dataset = {"1": "Fever", "2": "TruthfulQA", "3": "placeholder"}

    print("Available datasets")
    for key in available_dataset:
        print(f"{key}: {available_dataset[key]}")

    # Ask for a dataset to use
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
    # model, tokenizer = model_selector()
    models_tns = model_selector()
    # grab datasets
    dataset = dataset_selector()

    run_fever = input("Would you like to validates the model(s)? (y/N)")
    if run_fever.lower() == "y":
        # run validation
        if len(models_tns) > 1:
            if __debug__:
                print("mutli-model mode")

            for model, tokenizer in models_tns:
                print(f"\nEvaluating {model.config.architectures}")
                evaluation_pipeline(model, tokenizer, dataset)
        else:
            if __debug__:
                print("single-model mode")

            model = models_tns[0][0]
            tokenizer = models_tns[0][1]

            evaluation_pipeline(model, tokenizer, dataset)

    # close program
    sys.exit(0)


if __name__ == "__main__":
    main()
