from utilities import (
    model_selector,
    process_fever,
    process_truthfulqa,
)
from transformers import TrainingArguments
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, TaskType
from trl import SFTTrainer
import sys
import os


def prompt_instruction_format(sample):
    return f"""### Instruction:
        Use the Task below and the Input given to write the Response:

        ### Task:
        {sample["instruction"]}

        ### Input:
        {sample["input"]}

        ### Response:
        {sample["output"]}
        """


def fine_tune():
    # load dataset
    print("\nLoading datasets...")
    tf_dataset = load_dataset("truthful_qa", "generation")
    fe_dataset = load_dataset("fever", "v2.0", trust_remote_code=True)

    # Preprocessing
    print("\nPreprocessing...")
    tf_dataset = tf_dataset["validation"].map(process_truthfulqa)  # type: ignore
    fe_dataset = fe_dataset["train"].map(process_fever)  # type: ignore

    # Trying with just fever
    dataset = concatenate_datasets([fe_dataset, tf_dataset], axis=0)
    dataset = dataset.remove_columns(
        [
            col
            for col in dataset.column_names
            if col not in ["instruction", "input", "output"]
        ]
    )
    # load lambda 3
    model_tns = model_selector("2")
    model = model_tns[0][0]
    tokenizer = model_tns[0][1]

    # load LoRA
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Training Configuration
    training_args = TrainingArguments(
        output_dir="./llama3_lora_finetuned",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        save_strategy="epoch",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=prompt_instruction_format,
        args=training_args,
    )

    trainer.train()

    print("Model fine-tuning complete!")

    # Save fine-tuned model
    model.save_pretrained("./lora_finetuned_model")
    tokenizer.save_pretrained("./lora_finetuned_model")

    model.save_pretrained("./lora_finetuned_model_adapter", save_adapter=True)
    tokenizer.save_pretrained("./lora_finetuned_model_adapter")
    print("Model saved successfully!")

    sys.exit(0)


if __name__ == "__main__":
    fine_tune()
