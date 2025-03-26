from utilities import model_selector, load_lora


def fine_tune():
    # load lambda 3.1
    model_tns = model_selector("2", padding=True)

    # load LoRA
    lora_config = {}
    lora = load_lora(model=model_tns[0], config=lora_config)

    pass


if __name__ == "__main__":
    pass
