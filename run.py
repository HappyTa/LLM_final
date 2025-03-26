import sys
from evaluator import evaluate


def main():
    # Print out avaialble mode
    available_mode = {"1": "Evaluation", "2": "Fine-tuning"}
    keys_l = list(available_mode.keys())
    print("Avaiable modes:")
    for key in available_mode:
        print(f"{key}. {available_mode[key]}")
    mode = input(f"Please select a mode ({keys_l[0]}-{keys_l[-1]}): ")

    # Match mode with correct operation.
    match mode:
        case "1":
            evaluate()
        case "2":
            pass
        case _:
            raise ValueError(
                f"Invalid input for mode selection, please pick a number from {keys_l[0]} to {keys_l[-1]}"
            )

    sys.exit(0)


if __name__ == "__main__":
    main()
