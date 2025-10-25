import os
import json
import subprocess
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from train import ModelArguments, DataTrainingArguments, OurTrainingArguments

@dataclass
class NoiseTestArguments:
    noise_ratios: str = field(
        default="0.1,0.2,0.3,0.4",
        metadata={"help": "Comma-separated list of noise ratios to test."}
    )
    data_path: str = field(
        default="data/trec/train.csv",
        metadata={"help": "Path to the training data."}
    )
    noise_type: str = field(
        default="asym",
        metadata={"help": "Type of noise to introduce (e.g., 'asym', 'sym')."}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments, NoiseTestArguments))
    model_args, data_args, training_args, noise_args = parser.parse_args_into_dataclasses()

    os.makedirs("models", exist_ok=True)

    noise_ratios = [float(r) for r in noise_args.noise_ratios.split(',')]

    for ratio in noise_ratios:
        print(f"Testing with noise ratio: {ratio}")

        # Generate noisy data
        corrupted_data_path = f"data/{data_args.dataset_name}/train_corrupted_{ratio}.csv"
        subprocess.run([
            "python", "data/corrupt.py",
            "--src_data_path", noise_args.data_path,
            "--save_path", corrupted_data_path,
            "--noise_type", noise_args.noise_type,
            "--noise_ratio", str(ratio)
        ])

        # Train the model
        subprocess.run([
            "python", "train.py",
            "--pretrained_model_name_or_path", model_args.pretrained_model_name_or_path,
            "--dropout_rate", str(model_args.dropout_rate),
            "--p_threshold", str(model_args.p_threshold),
            "--temp", str(model_args.temp),
            "--alpha", str(model_args.alpha),
            "--lambda_p", str(model_args.lambda_p),
            "--lambda_r", str(model_args.lambda_r),
            "--gmm_max_iter", str(model_args.gmm_max_iter),
            "--gmm_tol", str(model_args.gmm_tol),
            "--gmm_reg_covar", str(model_args.gmm_reg_covar),
            "--dataset_name", data_args.dataset_name,
            "--train_file_path", corrupted_data_path,
            "--eval_file_path", data_args.eval_file_path,
            "--batch_size", str(data_args.batch_size),
            "--batch_size_mix", str(data_args.batch_size_mix),
            "--max_sentence_len", str(data_args.max_sentence_len),
            "--seed", str(training_args.seed),
            "--warmup_strategy", training_args.warmup_strategy,
            "--warmup_epochs", str(training_args.warmup_epochs),
            "--train_epochs", str(training_args.train_epochs),
            "--lr", str(training_args.lr),
            "--grad_acc_steps", str(training_args.grad_acc_steps),
            "--patience", str(training_args.patience),
            "--model_save_path", f"{training_args.model_save_path}_{ratio}",
            "--device", training_args.device,
        ])

if __name__ == "__main__":
    main()
