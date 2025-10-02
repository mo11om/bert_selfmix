from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.mixture import GaussianMixture
import os, logging, warnings
import torch
import math
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
 
logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings('ignore')


def metric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }
    
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class SelfMixTrainer:
    def __init__(self, model, train_data=None, eval_data=None, model_args=None, training_args=None):
        self.model = model.cuda()
        self.train_data = train_data
        self.eval_data = eval_data
        self.model_args = model_args
        self.training_args = training_args
        if self.training_args is not None:
            self.optimizer = Adam(self.model.parameters(), lr=training_args.lr)
    
    def warmup(self):
        logger.info("***** Warmup stage *****")
        
        train_loader = self.train_data.run("all")
        
        if self.training_args.warmup_strategy == "epoch":
            warmup_samples = self.training_args.warmup_epochs * len(train_loader.dataset)
            warmup_epochs = self.training_args.warmup_epochs
        elif self.training_args.warmup_strategy == "samples":
            warmup_samples = self.training_args.warmup_samples
            warmup_epochs = self.training_args.warmup_samples // len(train_loader.dataset) + \
                            int(self.training_args.warmup_samples % len(train_loader.dataset) > 0)
        else:
            warmup_samples, warmup_epochs = 0, 0
            
        loss_func = nn.CrossEntropyLoss()
        now_samples = 0
        
        patience_counter = 0
        best_eval_metric = float('-inf')
        
        for epoch_id in range(1, warmup_epochs + 1):
            logger.info("***** Warmup epoch %d *****", epoch_id)
            
            self.model.train()
            train_loss, train_acc = 0., 0.
            for i, data in enumerate(train_loader): 
                # print (data) 
                # input_ids, att_mask, labels, _  = [Variable(elem.cuda()) for elem in data]
                input_ids, att_mask, labels, _ = [elem.cuda() for elem in data]
                logits = self.model(input_ids, att_mask)
                loss = loss_func(logits, labels)
                train_loss += loss.item()
                
                pred = logits.argmax(dim=-1).cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                train_acc += (pred == labels).sum()

                loss = loss / self.training_args.grad_acc_steps
                loss.backward()

                if (i + 1) % self.training_args.grad_acc_steps == 0 or i + 1 == len(train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                now_samples += input_ids.size(0)
                if now_samples >= warmup_samples:
                    logger.info("Warmup Stage ends in %d samples", now_samples)
                    break
            if now_samples >= warmup_samples:
                break        
                        
            logger.info("Warmup train samples [{:6d}/{:6d}], Loss: {:4f}, Accuracy: {:.2%}"
                        .format(now_samples, warmup_samples, train_loss / len(train_loader), train_acc / len(train_loader.dataset)))
                    
            if self.eval_data is not None:
                eval_loader = self.eval_data.run("all")
                # self.evaluate(eval_loader)
                eval_metrics = self.evaluate(eval_loader)

                # Assuming a single evaluation metric is being monitored for patience, e.g., 'eval_accuracy'
                current_eval_metric = eval_metrics.get('f1', -1)

                if current_eval_metric > best_eval_metric:
                    best_eval_metric = current_eval_metric
                    self.save_model()
                    logger.info("Model saved to %s", self.training_args.model_save_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.training_args.patience:
                    logger.info(f"Early stopping triggered. No improvement in evaluation metric for {self.training_args.patience} consecutive epochs.")
                     
                    break


                
    def evaluate(self, eval_loader=None):
        if eval_loader is None:
            eval_loader = self.eval_data.run("all")        
        self.model.eval()
        y_true, y_pred = np.zeros(len(eval_loader.dataset), dtype=int), np.zeros(len(eval_loader.dataset), dtype=int)
        for j, data in enumerate(eval_loader):
            val_input_ids, val_att, val_labels, index = [Variable(elem.cuda()) for elem in data]
            with torch.no_grad():
                index = index.long().cpu().detach().numpy()
                pred = self.model(val_input_ids, val_att).argmax(dim=-1).cpu().detach().numpy()
                val_labels = val_labels.cpu().detach().numpy()
            y_true[index] = val_labels
            y_pred[index] = pred
            
        eval_res = metric(y_true, y_pred)
        logger.info("Eval Results: Accuracy: {:.2%}, Precision: {:.2%}, Recall: {:.2%}, F1: {:.2%}"
            .format(eval_res['accuracy'], eval_res['precision'], eval_res['recall'], eval_res['f1']))
        return eval_res
    def train(self):
        logger.info("***** Mixup Train *****")       

        train_loader = self.train_data.run("all")
        eval_loader = self.eval_data.run("all")
        best_eval_metric = float('-inf')
        patience_counter = 0
        for epoch_id in range(1, self.training_args.train_epochs + 1):
            prob = self._eval_samples(train_loader)
            pred = (prob > self.model_args.p_threshold)
            labeled_train_loader, unlabeled_train_loader = self.train_data.run("train", pred, prob)
            
            logger.info("***   Train epoch  %d ***", epoch_id)
            self._train_epoch(labeled_train_loader, unlabeled_train_loader)
            
            logger.info("*** Evaluate epoch %d ***", epoch_id)
            eval_res=self.evaluate(eval_loader)
            if eval_res['f1']>=best_eval_metric:
                best_eval_metric=eval_res['f1']
                patience_counter=0  
                self.save_model()
                logger.info("Model saved to %s", self.training_args.model_save_path)
               
    
    def _train_epoch(self, labeled_train_loader, unlabeled_train_loader):
        labeled_train_iter = iter(labeled_train_loader)
        unlabeled_train_iter = iter(unlabeled_train_loader)
        val_iteration = len(labeled_train_loader)
        self.model.train()
        
        for batch_idx in range(val_iteration):
            try:
                inputs_x, inputs_x_att, targets_x, _, _ = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(labeled_train_loader)
                inputs_x, inputs_x_att, targets_x, _  = next(labeled_train_iter)

            try:
                inputs_u, att_u, _ = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_train_loader)
                inputs_u, att_u, _ = next(unlabeled_train_iter)

            
            targets_x = F.one_hot(targets_x, num_classes=self.model_args.num_classes)
            inputs_x, inputs_x_att, targets_x = inputs_x.cuda(), inputs_x_att.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, att_u = inputs_u.cuda(), att_u.cuda()
                        
            self.model.eval()
            with torch.no_grad():
                # Predict labels for unlabeled data.
                out_u = self.model(inputs_u, att_u)
                p = torch.softmax(out_u, dim=1)
                pt = p ** (1 / self.model_args.temp)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()
                
            self.model.train()
            sents_x = self.model.get_sentence_embedding(inputs_x, inputs_x_att)
            sents_u1 = self.model.get_sentence_embedding(inputs_u, att_u)
            sents_u2 = self.model.get_sentence_embedding(inputs_u, att_u)
            
            all_sents = torch.cat(
                [sents_x, sents_u1], dim=0)
            all_targets = torch.cat(
                [targets_x, targets_u], dim=0)
            
            # mixup
            rand_idx = torch.randperm(all_sents.size(0))
            l = np.random.beta(self.model_args.alpha, self.model_args.alpha)
            l = max(l, 1 - l)
            mixed_sents = l * all_sents + (1 - l) * all_sents[rand_idx]
            mixed_targets = l * all_targets + (1 - l) * all_targets[rand_idx]
            
            logits = self.model.classify(mixed_sents)
            logits_u1 = self.model.classify(sents_u1)
            logits_u2 = self.model.classify(sents_u2)
            
            # compute loss
            loss_mix = -torch.mean(torch.sum(F.log_softmax(logits, dim=-1) * mixed_targets, dim=-1))            
            pse_loss = -torch.mean(F.log_softmax(logits_u1, dim=1).min(dim=1)[0]) * 0.5 \
                        - torch.mean(F.log_softmax(logits_u2, dim=1).min(dim=1)[0]) * 0.5
            kl_loss = compute_kl_loss(logits_u1, logits_u2)
            loss = loss_mix + pse_loss * self.model_args.lambda_p + kl_loss * self.model_args.lambda_r
            loss = loss / self.training_args.grad_acc_steps
            loss.backward()
            if (batch_idx + 1) % self.training_args.grad_acc_steps == 0 or batch_idx + 1 == val_iteration:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
        logger.info("Loss_Mix: {:.4f}, Loss_P: {:.4f}, Loss_R: {:.4f}, Loss: {:.4f} "
                    .format(loss_mix, pse_loss, kl_loss, loss))

    # def _eval_samples(self, eval_loader):
    #     """
    #     Sample selection
    #     """
    #     self.model.eval()
    #     loss_func = nn.CrossEntropyLoss(reduction='none')
    #     losses = np.zeros(len(eval_loader.dataset))
    #     with torch.no_grad():
    #         for i, data in enumerate(eval_loader):
    #             input_ids, att_mask, labels, index = [Variable(elem.cuda()) for elem in data] 
    #             outputs = self.model(input_ids, att_mask) 
    #             pred = torch.softmax(outputs, dim=-1)
    #             loss = loss_func(pred, labels).cpu().detach().numpy()
    #             index = index.long().cpu().detach().numpy()
    #             losses[index] = loss
                
    #     if self.model_args.class_reg:
    #         labels = np.array(eval_loader.dataset.labels, dtype=int)
    #         for now_class in range(self.model_args.num_classes):
    #             indices = np.where(labels == now_class)[0]
    #             losses[indices] = (losses[indices] - losses[indices].mean()) / losses[indices].var()
    #     else:
    #         losses = (losses - losses.min()) / (losses.max() - losses.min())
        
    #     gmm = GaussianMixture(
    #         n_components=2, 
    #         max_iter=self.model_args.gmm_max_iter, 
    #         tol=self.model_args.gmm_tol, 
    #         reg_covar=self.model_args.gmm_reg_covar
    #     )
    #     losses = losses.reshape(-1, 1)
    #     gmm.fit(losses)
    #     prob = gmm.predict_proba(losses) 
    #     prob = prob[:,gmm.means_.argmin()]
    #     return prob
    
    
    def _eval_samples(self, eval_loader):
        """
        Calculates per-sample loss, fits GMM, and returns clean probabilities.
        Includes robust filtering for NaN/Inf values before GMM fitting.
        """
        self.model.eval()
        loss_func = nn.CrossEntropyLoss(reduction='none')
        losses = np.zeros(len(eval_loader.dataset))
        
        with torch.no_grad():
            for i, data in enumerate(eval_loader):
                # Ensure data is handled correctly (using Variable only where necessary)
                input_ids, att_mask, labels, index = [Variable(elem.cuda()) for elem in data] 
                outputs = self.model(input_ids, att_mask) 
                
                # Use softmax outputs for loss calculation to match original code structure
                pred = torch.softmax(outputs, dim=-1) 
                
                # Compute loss and store in the losses array at the correct index
                loss = loss_func(pred, labels).cpu().detach().numpy()
                index_np = index.long().cpu().detach().numpy()
                losses[index_np] = loss
                
        # --- NaN/Inf Handling and Normalization ---
        
        # 1. Identify which losses are finite (not NaN or Inf)
        finite_mask = np.isfinite(losses)
        finite_indices = np.where(finite_mask)[0]
        finite_losses = losses[finite_indices]
        
        # Initialize final probability array (defaulting to 0 for non-finite samples)
        original_prob = np.zeros(len(eval_loader.dataset))
        
        if finite_losses.size < 2: # GMM requires at least 2 samples
            logger.error(f"Only {finite_losses.size} finite loss values remain. Cannot reliably fit GMM.")
            # Returns 0 probability for all samples if GMM fitting is impossible
            return original_prob 
            
        
        # 2. Perform Normalization on only the FINITE losses
        normalized_finite_losses = np.copy(finite_losses) # Operate on a copy
        
        if self.model_args.class_reg:
            labels_array = np.array(eval_loader.dataset.labels, dtype=int)
            finite_labels = labels_array[finite_indices]
            
            for now_class in range(self.model_args.num_classes):
                class_mask = (finite_labels == now_class)
                class_losses = finite_losses[class_mask]
                
                if class_losses.size > 0:
                    mean = class_losses.mean()
                    var = class_losses.var()
                    
                    # Handle near-zero variance
                    if var < 1e-6:
                        normalized_finite_losses[class_mask] = (class_losses - mean)
                    else:
                        normalized_finite_losses[class_mask] = (class_losses - mean) / var
        else:
            # Global normalization
            losses_min, losses_max = finite_losses.min(), finite_losses.max()
            if losses_max - losses_min < 1e-6:
                normalized_finite_losses = (finite_losses - losses_min)
            else:
                normalized_finite_losses = (finite_losses - losses_min) / (losses_max - losses_min)
                
        
        # 3. Fit GMM on FINITE and NORMALIZED losses
        gmm = GaussianMixture(
            n_components=2, 
            max_iter=self.model_args.gmm_max_iter, 
            tol=self.model_args.gmm_tol, 
            reg_covar=self.model_args.gmm_reg_covar
        )
        
        # Reshape for GMM fitting
        normalized_finite_losses = normalized_finite_losses.reshape(-1, 1)
        gmm.fit(normalized_finite_losses)
        
        # 4. Get clean probabilities for FINITE SAMPLES ONLY
        probabilities_matrix = gmm.predict_proba(normalized_finite_losses) 
        index_of_clean_component = gmm.means_.argmin()
        finite_prob = probabilities_matrix[:, index_of_clean_component]
        
        # 5. Map probabilities back to the original dataset size
        original_prob[finite_indices] = finite_prob
        
        if not np.all(finite_mask):
             non_finite_count = np.sum(~finite_mask)
             logger.warning(f"Note: {non_finite_count} samples had non-finite loss and were assigned probability 0 for splitting.")
        
        return original_prob

    def get_labeled_unlabeled_loaders(self):
        """
        Implements the dynamic data splitting process (SelfMix) to separate the 
        training data into labeled (clean) and unlabeled (noisy) subsets 
        based on the current model's loss and GMM fitting.

        Returns:
            labeled_trainloader (DataLoader): Loader for the clean/labeled subset.
            unlabeled_trainloader (DataLoader): Loader for the noisy/unlabeled subset.
        """
        logger.info("***** Starting Dynamic Data Split for Mixup *****")
        
        # 1. INITIAL SETUP and DATA LOADING
        self.model.eval()
        train_loader = self.train_data.run("all")
        loss_func = nn.CrossEntropyLoss(reduction='none')
        
        total_samples = len(train_loader.dataset)
        losses = np.zeros(total_samples)
        
        # 2. CALCULATE PER-SAMPLE LOSS
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                # Ensure data is handled correctly (using Variable only where necessary,
                # but matching the provided trainer.py style for consistency)
                # The _eval_samples uses Variable, so we match it here.
                input_ids, att_mask, labels, index = [Variable(elem.cuda()) for elem in data] 
                
                # Note: The original _eval_samples passes outputs (logits) to loss_func, 
                # but then uses softmax(outputs) inside the loss_func call (loss_func(pred, labels)).
                # To be precise to the original code's logic which is loss_func(pred, labels), 
                # where pred=softmax(outputs), we will stick to the logic for now.
                # However, typically CrossEntropyLoss expects logits. 
                # Sticking to the logic of the provided _eval_samples for minimal change.
                outputs = self.model(input_ids, att_mask) 
                pred = torch.softmax(outputs, dim=-1)
                
                loss = loss_func(pred, labels).cpu().detach().numpy()
                index = index.long().cpu().detach().numpy()
                losses[index] = loss

        # 3. OPTIONAL: CLASS-WISE REGULARIZATION AND NORMALIZATION
        if self.model_args.class_reg:
            labels_array = np.array(train_loader.dataset.labels, dtype=int)
            for now_class in range(self.model_args.num_classes):
                indices = np.where(labels_array == now_class)[0]
                # Avoid division by zero/near-zero variance
                var = losses[indices].var()
                if var < 1e-6:
                    losses[indices] = (losses[indices] - losses[indices].mean())
                else:
                    losses[indices] = (losses[indices] - losses[indices].mean()) / var
        else:
            # Global min-max scaling
            losses_min, losses_max = losses.min(), losses.max()
            if losses_max - losses_min < 1e-6:
                # Avoid division by zero
                losses = losses - losses_min
            else:
                losses = (losses - losses_min) / (losses_max - losses_min)
        
        # 4. FIT GAUSSIAN MIXTURE MODEL (GMM)
        losses = losses.reshape(-1, 1)
        gmm = GaussianMixture(
            n_components=2, 
            max_iter=self.model_args.gmm_max_iter, 
            tol=self.model_args.gmm_tol, 
            reg_covar=self.model_args.gmm_reg_covar
        )
        gmm.fit(losses)
        
        # 5. DETERMINE CLEAN PROBABILITY ('prob')
        probabilities_matrix = gmm.predict_proba(losses) 
        # Select the probability of belonging to the clean (low-loss) component
        index_of_clean_component = gmm.means_.argmin()
        prob = probabilities_matrix[:, index_of_clean_component]

        # 6. THRESHOLD AND CREATE SPLIT MASK ('pred')
        pred = (prob > self.model_args.p_threshold)

        # 7. CREATE LABELED AND UNLABELED DATALOADERS
        labeled_trainloader, unlabeled_trainloader = self.train_data.run("train", pred=pred, prob=prob)
        
        logger.info(f"Split results: Labeled (Clean) samples: {pred.sum()}, Unlabeled (Noisy) samples: {total_samples - pred.sum()}")
        return labeled_trainloader, unlabeled_trainloader
    def save_model(self):
        self.model.save_model(self.training_args.model_save_path)
