from utilities import load_model, evaluation_pipeline
from datasets import load_dataset
import sys


def model_selector(std_in=None):
    available_models = {
        "1": "google/flan-t5-large",
        "2": "meta-llama/llama-2-7b-hf",
        "3": "openai-community/gpt2",
        "4": "All",
    }

    if not std_in:
        """Ask the user to select an LLM model and load it."""

        print("Available Models:")
        for key, model in available_models.items():
            print(f"{key}: {model}")

        # Ask user for selection
        choice = input("\nEnter the number of the model you want to use: ").strip()
    else:
        choice = std_in

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


def dataset_selector(dataset=None):
    """Ask the user to select a dataset and load it."""
    available_dataset = {"1": "Fever", "2": "TruthfulQA"}

    if not dataset:
        print("Available datasets")
        for key in available_dataset:
            print(f"{key}: {available_dataset[key]}")

        # Ask for a dataset to use
        choice = input("\nEnter the number of the dataset you want to use: ").strip()
    else:
        if not dataset.isdigit():
            raise ValueError("Please only use numerical values for dataset selections.")

        choice = dataset

    print(f"Loading {available_dataset[choice]}...")

    match choice:
        case "1":
            return (0, load_dataset("fever", "v2.0", split="validation"))
        case "2":
            return (1, load_dataset("truthful_qa", split="validation"))
        case _:
            print("Invalid choice. Using default dataset: Fever")
            return (0, load_dataset("fever", "v2.0", split="validation"))


def main():
    if len(sys.argv) > 1:
        if not sys.argv[1].isdigit():
            raise EnvironmentError(
                "Please only pass in numerical values for model selection choice"
            )
    elif len(sys.argv) == 3:
        if not sys.argv[1].isdigit() or sys.argv[2].isdigit():
            raise EnvironmentError(
                "Please only pass in numerical values for model selection/datset selection choice"
            )
    # Select model
    models_tns = model_selector(sys.argv[1] if len(sys.argv) > 1 else None)

    # grab datasets
    dataset_type, dataset = dataset_selector(
        sys.argv[2] if len(sys.argv) == 3 else None
    )

    # If Truthful is being use, check if user passed in a embeded model choice
    if dataset_type == 1 and len(sys.argv) == 4:
        emb_model = sys.argv[4]
    else:
        emb_model = None

    # run validation
    print("\nStarting Validation")
    if len(models_tns) > 1 and __debug__:
        print("mutli-model mode")
    elif __debug__:
        print("single-model mode")

    evaluation_pipeline(models_tns, dataset, dataset_type, emb_model)

    # close program
    sys.exit(0)


if __name__ == "__main__":
    main()
