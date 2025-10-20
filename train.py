import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import HfArgumentParser, set_seed

from lightning_datamodule import SelfMixDataModule
from lightning_module import SelfMixLightningModule
from gpu_monitoring_callback import GpuMonitoringCallback


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """

    # Huggingface's original arguments
    pretrained_model_name_or_path: Optional[str] = field(
        default="bert-base-uncased",
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        },
    )
    dropout_rate: float = field(default=0.1, metadata={"help": "Dropout rate"})

    # SelfMix's arguments
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The pretrained model checkpoint for weights initialization."
        },
    )
    p_threshold: float = field(
        default=0.5, metadata={"help": "Clean probability threshold"}
    )
    temp: float = field(default=0.5, metadata={"help": "Temperature for sharpen function"})
    alpha: float = field(default=0.75, metadata={"help": "Alpha for beta distribution"})
    lambda_p: float = field(default=0.2, metadata={"help": "Weight for Pseudo Loss"})
    lambda_r: float = field(default=0.3, metadata={"help": "Weight for R-Drop loss"})
    class_reg: bool = field(
        default=False,
        metadata={"help": "Whether to apply class regularization to loss"},
    )
    ## gmm arguments
    gmm_max_iter: int = field(
        default=10, metadata={"help": "The number of EM iterations to perform"}
    )
    gmm_tol: float = field(default=1e-2, metadata={"help": "The convergence threshold"})
    gmm_reg_covar: float = field(
        default=5e-4,
        metadata={"help": "Non-negative regularization added to the diagonal of covariance."},
    )
    num_classes: int = field(default=2, metadata={"help": "Number of classes"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Name of dataset"}
    )
    train_file_path: Optional[str] = field(
        default=None, metadata={"help": "The train data file (.csv)"}
    )
    eval_file_path: Optional[str] = field(
        default=None, metadata={"help": "The eval data file (.csv)"}
    )
    batch_size: int = field(default=32, metadata={"help": "Batch size"})
    batch_size_mix: int = field(default=16, metadata={"help": "Batch size for mix train"})

    max_sentence_len: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sentence length after tokenization. Sequences longer."
        },
    )


@dataclass
class OurTrainingArguments:
    seed: Optional[int] = field(default=1, metadata={"help": "Seed"})
    warmup_epochs: Optional[int] = field(
        default=2,
        metadata={
            "help": "Number of epochs to warmup the model"
        },
    )
    train_epochs: int = field(default=4, metadata={"help": "Mix-up training epochs"})
    lr: float = field(default=1e-5, metadata={"help": "Learning rate"})
    gpus: int = field(default=1, metadata={"help": "Number of gpus"})
    patience: int = field(default=3, metadata={"help": "Patience for early stop"})
    model_save_path: Optional[str] = field(
        default="./model_checkpoints", metadata={"help": "The path to save model"}
    )


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(training_args.seed)

    datamodule = SelfMixDataModule(data_args, model_args)
    model = SelfMixLightningModule(model_args, training_args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=training_args.model_save_path,
        filename="{epoch}-{val_f1:.2f}",
        save_top_k=1,
        monitor="val_f1",
        mode="max",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_f1",
        patience=training_args.patience,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=training_args.warmup_epochs + training_args.train_epochs,
        gpus=training_args.gpus,
        callbacks=[checkpoint_callback, early_stop_callback, GpuMonitoringCallback()],
        # Add other trainer arguments as needed
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
