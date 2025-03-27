from utilities import dataset_selector, model_selector
from transformers import TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import sys


def fine_tune():
    # load lambda 3.1
    model_tns = model_selector("2")
    model = prepare_model_for_kbit_training(model_tns[0][0])  # Prep model for LoRA
    tokenizer = model_tns[0][1]

    # load LoRA
    lora_config = LoraConfig(
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model=model, peft_config=lora_config)

    # load truthful
    dataset = load_dataset("truthful_qa", "generation")
    dataset = dataset["validation"].train_test_split(test_size=0.2, seed=69)

    def tokenize_data(examples):
        return tokenizer(
            examples["question"],
            examples["best_answer"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    dataset = dataset.map(tokenize_data, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./lora_finetuned_model",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_dir="./logs",
        num_train_epochs=3,
        save_steps=500,
        logging_steps=100,
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,  # Enables mixed precision for faster training
        push_to_hub=False,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("./lora_finetuned_model")
    tokenizer.save_pretrained("./lora_finetuned_model")

    sys.exit(0)


if __name__ == "__main__":
    fine_tune()
