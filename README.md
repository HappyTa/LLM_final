# Fine-Tuning LLaMA 3.3 for Fact-Checking and Hallucination Detection in LLMs

You can read about the research by downloading the *Fine-Tuning LLaMA.pdf* file and checkout the presentation using the *LLM_Final_Presentation.pptx*.

### Requirements

- `transformers`
- `datasets`
- `peft`
- `accelerate`
- `bitsandbytes`
- `torch`
- `trl`

To instal use:

```cli
pip install transformers datasets peft accelerate bitsandbytes torch tr
```

Permission to `meta-llama/Meta-Llama-3-8B`, once you have acquire the permission, please create an access token and use:

```cli
huggingface-cli login
```

In order to add the token to your machine. You might night to do `pip install -U "huggingface_hub[cli]"` to install it.


### How to use

```cli
python3 run.py
```

### Datasets and Models used

#### Datasets:

- FEVER: https://huggingface.co/datasets/fever/fever
- TruthfulQA: https://huggingface.co/datasets/truthfulqa/truthful_qa

#### Models

- meta-llama/Meta-Llama-3-8B
- openai-community/gpt2
- google/flan-t5-large







