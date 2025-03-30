from utilities import (
    dataset_selector,
    model_selector,
    process_fever,
    process_truthfulqa,
)
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, TaskType
import sys


# Tokenization function with tokenizer as a parameter
def tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["instruction"] + " " + examples["input"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    labels = tokenizer(
        examples["output"], truncation=True, padding="max_length", max_length=128
    )
    inputs["labels"] = labels["input_ids"]
    return inputs


def fine_tune():
    # load lambda 3.1
    model, tokenizer = model_selector("2")

    # load dataset
    tf_dataset = load_dataset("truthful_qa", "multiple_choice")
    fe_dataset = load_dataset("fever")

    # Preprocessing
    tf_dataset = process_truthfulqa(tf_dataset)
    fe_dataset = process_fever(fe_dataset)

    dataset = Dataset.from_list(tf_dataset + fe_dataset)

    # load LoRA
    lora_config = LoraConfig(
        r=32,  # LoRA rank (can be adjusted between 16-64)
        lora_alpha=64,  # Scaling factor (typically 2x r)
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
        lora_dropout=0.05,  # Regularization
        bias="none",
        task_type=TaskType.CAUSAL_LM,  # Suitable for text generation
    )
    model = get_peft_model(model=model, peft_config=lora_config)
    model.print_trainable_parameters()  # Verify trainable parameters

    # tokenize dataset
    dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Training Configuration
    training_args = TrainingArguments(
        output_dir="./llama3-lora-finetuned",
        per_device_train_batch_size=2,  # Adjust based on memory (A100 can handle 2-4)
        gradient_accumulation_steps=16,  # Accumulate gradients to simulate larger batch size
        learning_rate=2e-4,  # LoRA allows higher LR
        num_train_epochs=3,  # Adjust as needed
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,  # Mixed precision training for efficiency
        report_to="none",
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer
    )

    # Start Training
    trainer.train()

    # Save fine-tuned model
    model.save_pretrained("./lora_finetuned_model")
    tokenizer.save_pretrained("./lora_finetuned_model")
    #
    # # load truthful
    # dataset = dataset["validation"].train_test_split(test_size=0.2, seed=69)
    #
    # def tokenize_data(examples):
    #     return tokenizer(
    #         examples["question"],
    #         examples["best_answer"],
    #         truncation=True,
    #         padding="max_length",
    #         max_length=512,
    #     )
    #
    # dataset = dataset.map(tokenize_data, batched=True)
    #
    # # Training arguments
    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     per_device_train_batch_size=2,  # Keep small to avoid OOM
    #     gradient_accumulation_steps=4,  # Simulate larger batch size
    #     fp16=True,  # Enables mixed precision training
    #     optim="adamw_bnb_8bit",  # Memory-efficient optimizer
    #     logging_steps=10,
    #     evaluation_strategy="steps",
    #     save_steps=500,
    #     report_to="none",
    # )
    #
    # # Define Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    # )
    #
    # trainer.train()
    #
    #
    sys.exit(0)


if __name__ == "__main__":
    fine_tune()
