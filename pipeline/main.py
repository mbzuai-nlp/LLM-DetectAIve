import argparse
from model_pipeline import run_training_pipeline 


def main():
    parser = argparse.ArgumentParser(description="Run the training pipeline for the model.")

    parser.add_argument('--file_path', type=str, required=True, help='Path to the input dataset file.')
    parser.add_argument('--out_path', type=str,  default=".", help='Path to the saving model, tokenizer and dataset.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to be trained.')
    parser.add_argument('--num_labels', type=int, default=4, help='Number of labels for the classification task.')
    parser.add_argument('--sample_frac', type=float, default=1.0, help='Fraction of the dataset to sample for training.')
    parser.add_argument('--num_trials', type=int, default=5, help='Number of trials for hyperparameter search.')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs for training.')

    args = parser.parse_args()

    labels_dict = {
        "human_text": 0,
        "human-written | machine-polished": 1,
        "machine_text": 2,
        "machine-generated | machine-humanized": 3
    }

    run_training_pipeline(
        file_path=args.file_path,
        labels_dict=labels_dict,
        model_name=args.model_name,
        num_labels=args.num_labels,
        sample_frac=args.sample_frac,
        num_trials=args.num_trials,
        num_epochs=args.num_epochs,
        save_dir=args.out_path
    )

if __name__ == "__main__":
    main()
