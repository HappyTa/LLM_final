from utilities import load_model, fact_checking_pipeline
from datasets import load_dataset
import sys


def main():
    # Declare Models Names
    flan_name = "google/flan-t5-large"
    llama_2_name = "meta-llama/Llama-2-7b-hf"
    roberta_name = "FacebookAI/roberta-large-mnli"
    gpt_2_name = "openai-community/gpt2"

    # Load Models
    flan_model, flan_tn = load_model(flan_name)
    llama_model, llama_tn = load_model(llama_2_name)
    roberta_model, roberta_tn = load_model(roberta_name)
    gpt_model, gpt_tn = load_model(gpt_2_name)

    # grab datasets
    fever = load_dataset("fever", "v2.0", split="validation")
    # truthfulqa = load_dataset(
    #     "truthful_qa",
    #     split="validation",
    # )
    print("load good")
    # close program
    sys.exit(0)


if __name__ == "__main__":
    pass
