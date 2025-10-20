import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.optim import Adam
from transformers import AutoModel


class Bert4Classify(nn.Module):
    def __init__(self, pretrained_model_name_or_path, dropout_rate, num_classes):
        super(Bert4Classify, self).__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.encoder.resize_token_embeddings(
            len(self.encoder.embeddings.word_embeddings.weight) + 3
        )

        d_model = (
            768
            if ("bert" in pretrained_model_name_or_path or "BERT" in pretrained_model_name_or_path)
            else 1024
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, input_ids, att_mask):
        sentence_emb = self.get_sentence_embedding(input_ids, att_mask)
        output = self.classify(sentence_emb)
        return output

    def get_sentence_embedding(self, input_ids, att_mask):
        max_len = att_mask.sum(1).max()
        input_ids = input_ids[:, :max_len]
        att_mask = att_mask[:, :max_len]
        all_hidden = self.encoder(input_ids, att_mask)
        sentence_emb = all_hidden[0][:, 0]
        return sentence_emb

    def classify(self, x):
        output = self.mlp(x)
        return output


class CB_CE_Loss(nn.Module):
    def __init__(self, num_samples, beta=0.99):
        super(CB_CE_Loss, self).__init__()
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(num_samples))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        self.weights = weights

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        class_weights = self.weights.to(logits.device)
        weighted_loss = ce_loss * class_weights[targets]
        return torch.mean(weighted_loss)


def get_class_counts(train_dataset, num_classes):
    from collections import Counter

    all_labels = train_dataset.labels
    label_counts = Counter(all_labels)
    class_counts = [label_counts[i] for i in range(num_classes)]
    return class_counts


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(
        F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
    )
    q_loss = F.kl_div(
        F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
    )

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.0)
        q_loss.masked_fill_(pad_mask, 0.0)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class SelfMixLightningModule(pl.LightningModule):
    def __init__(self, model_args, training_args):
        super().__init__()
        self.save_hyperparameters()
        self.model_args = model_args
        self.training_args = training_args
        self.model = Bert4Classify(
            model_args.pretrained_model_name_or_path,
            model_args.dropout_rate,
            model_args.num_classes,
        )
        self.state = "warmup"

    def setup(self, stage):
        if stage == "fit":
            class_counts = get_class_counts(
                self.trainer.datamodule.train_dataset,
                self.trainer.datamodule.num_classes,
            )
            self.loss_fn = CB_CE_Loss(class_counts)

    def forward(self, input_ids, att_mask):
        return self.model(input_ids, att_mask)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.training_args.lr)

    def on_train_epoch_start(self):
        if self.current_epoch == self.training_args.warmup_epochs:
            self.state = "train"
            full_train_loader = self.trainer.datamodule.train_dataloader()
            self.prob = self._eval_samples(full_train_loader)
            self.pred = self.prob > self.model_args.p_threshold
            self.trainer.reset_train_dataloader(self)

    def training_step(self, batch, batch_idx):
        if self.state == "warmup":
            return self._warmup_step(batch)
        else:
            return self._mixup_step(batch)

    def _warmup_step(self, batch):
        input_ids, att_mask, labels, _ = batch
        logits = self(input_ids, att_mask)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def _mixup_step(self, batch):
        labeled_batch = batch["labeled"]
        unlabeled_batch = batch["unlabeled"]
        inputs_x, inputs_x_att, targets_x, _, _ = labeled_batch
        inputs_u, att_u, _ = unlabeled_batch

        targets_x = F.one_hot(targets_x, num_classes=self.model_args.num_classes)
        targets_x = targets_x.cuda(non_blocking=True)

        with torch.no_grad():
            out_u = self.model(inputs_u, att_u)
            p = torch.softmax(out_u, dim=1)
            pt = p ** (1 / self.model_args.temp)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        sents_x = self.model.get_sentence_embedding(inputs_x, inputs_x_att)
        sents_u1 = self.model.get_sentence_embedding(inputs_u, att_u)
        sents_u2 = self.model.get_sentence_embedding(inputs_u, att_u)

        all_sents = torch.cat([sents_x, sents_u1], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        rand_idx = torch.randperm(all_sents.size(0))
        l = np.random.beta(self.model_args.alpha, self.model_args.alpha)
        l = max(l, 1 - l)
        mixed_sents = l * all_sents + (1 - l) * all_sents[rand_idx]
        mixed_targets = l * all_targets + (1 - l) * all_targets[rand_idx]

        logits = self.model.classify(mixed_sents)
        logits_u1 = self.model.classify(sents_u1)
        logits_u2 = self.model.classify(sents_u2)

        loss_mix = -torch.mean(
            torch.sum(F.log_softmax(logits, dim=-1) * mixed_targets, dim=-1)
        )
        pse_loss = -torch.mean(F.log_softmax(logits_u1, dim=1).min(dim=1)[0]) * 0.5 - torch.mean(
            F.log_softmax(logits_u2, dim=1).min(dim=1)[0]
        ) * 0.5
        kl_loss = compute_kl_loss(logits_u1, logits_u2)
        loss = (
            loss_mix
            + pse_loss * self.model_args.lambda_p
            + kl_loss * self.model_args.lambda_r
        )

        self.log_dict(
            {"loss_mix": loss_mix, "pse_loss": pse_loss, "kl_loss": kl_loss}
        )
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, att_mask, labels, index = batch
        logits = self(input_ids, att_mask)
        preds = logits.argmax(dim=-1)
        return {"preds": preds, "labels": labels, "index": index}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).cpu().numpy()
        # The following logic is to handle distributed validation
        # Reorder predictions to match the original dataset order
        indices = torch.cat([x["index"] for x in outputs]).cpu().numpy()
        y_pred = np.zeros_like(labels)
        y_pred[indices] = preds
        y_true = np.zeros_like(labels)
        y_true[indices] = labels

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')

        self.log_dict({
            "val_acc": accuracy,
            "val_precision": macro_precision,
            "val_recall": macro_recall,
            "val_f1": macro_f1
        })


    def _eval_samples(self, eval_loader):
        self.model.eval()
        loss_func = nn.CrossEntropyLoss(reduction="none")
        losses = np.zeros(len(eval_loader.dataset))
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                input_ids, att_mask, labels, index = [elem.cuda() for elem in data]
                outputs = self.model(input_ids, att_mask)
                pred = torch.softmax(outputs, dim=-1)
                loss = loss_func(pred, labels).cpu().detach().numpy()
                index = index.long().cpu().detach().numpy()
                losses[index] = loss

        if self.model_args.class_reg:
            labels = np.array(eval_loader.dataset.labels, dtype=int)
            for now_class in range(self.model_args.num_classes):
                indices = np.where(labels == now_class)[0]
                losses[indices] = (losses[indices] - losses[indices].mean()) / losses[
                    indices
                ].var()
        else:
            losses = (losses - losses.min()) / (losses.max() - losses.min())

        gmm = GaussianMixture(
            n_components=2,
            max_iter=self.model_args.gmm_max_iter,
            tol=self.model_args.gmm_tol,
            reg_covar=self.model_args.gmm_reg_covar,
        )
        losses = losses.reshape(-1, 1)
        gmm.fit(losses)
        prob = gmm.predict_proba(losses)
        prob = prob[:, gmm.means_.argmin()]
        self.model.train()
        return prob

    def train_dataloader(self):
        if self.state == "warmup":
            return self.trainer.datamodule.train_dataloader()
        else:
            labeled_loader, unlabeled_loader = self.trainer.datamodule.get_train_dataloaders(self.pred, self.prob)
            return {"labeled": labeled_loader, "unlabeled": unlabeled_loader}
