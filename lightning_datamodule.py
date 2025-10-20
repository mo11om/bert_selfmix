import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from datasets import SelfMixDataset, load_dataset


class SelfMixDataModule(pl.LightningDataModule):
    def __init__(self, data_args, model_args):
        super().__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.pretrained_model_name_or_path
        )
        self.tokenizer.add_tokens(["<span>", "</span>", "N/A"])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset, self.num_classes = load_dataset(
                self.data_args.train_file_path, self.data_args.dataset_name
            )
            self.model_args.num_classes = self.num_classes
        if stage == "validate" or stage is None:
            self.eval_dataset, _ = load_dataset(
                self.data_args.eval_file_path, self.data_args.dataset_name
            )

    def train_dataloader(self):
        train_dataset_all = SelfMixDataset(
            data_args=self.data_args,
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            mode="all",
        )
        return DataLoader(
            dataset=train_dataset_all,
            batch_size=self.data_args.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def val_dataloader(self):
        eval_dataset_all = SelfMixDataset(
            data_args=self.data_args,
            dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            mode="all",
        )
        return DataLoader(
            dataset=eval_dataset_all,
            batch_size=self.data_args.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def get_train_dataloaders(self, pred, prob):
        labeled_dataset = SelfMixDataset(
            data_args=self.data_args,
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            mode="labeled",
            pred=pred,
            probability=prob,
        )
        labeled_loader = DataLoader(
            dataset=labeled_dataset,
            batch_size=self.data_args.batch_size_mix,
            shuffle=True,
            num_workers=8,
        )

        unlabeled_dataset = SelfMixDataset(
            data_args=self.data_args,
            dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            mode="unlabeled",
            pred=pred,
        )
        unlabeled_loader = DataLoader(
            dataset=unlabeled_dataset,
            batch_size=self.data_args.batch_size_mix,
            shuffle=True,
            num_workers=8,
        )

        return labeled_loader, unlabeled_loader
