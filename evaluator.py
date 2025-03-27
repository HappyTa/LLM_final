from utilities import dataset_selector, model_selector, evaluation_pipeline
import sys


def evaluate(models_tns=None):
    # Check for stdin values
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
    if not models_tns:
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
    if len(models_tns) > 1 and __debug__:
        print("mutli-model mode")
    elif __debug__:
        print("single-model mode")

    evaluation_pipeline(models_tns, dataset, dataset_type, emb_model)

    # close program
    sys.exit(0)


if __name__ == "__main__":
    evaluate()
