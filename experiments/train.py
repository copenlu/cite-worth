import torch
import gc
import random
import numpy as np
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset
from typing import AnyStr, Union, List
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForMaskedLM
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.optim import Adam
import ipdb

from datareader import collate_batch_transformer
from datareader import collate_batch_transformer_with_weight
from datareader import collate_batch_language_modeling
from datareader import collate_sequence_batch_transformer
from datareader import DataParallelV2
from metrics import ClassificationEvaluator
from model import AutoTransformerMultiTask


class AbstractTransformerTrainer:
    """
    An abstract class which other trainers should implement
    """
    def __init__(
            self,
            model = None,
            device = None,
            tokenizer = None
    ):

        self.model = model
        self.device = device
        self.tokenizer = tokenizer

    def create_optimizer(self, lr: float, weight_decay: float=0.0):
        """
        Create a weighted adam optimizer with the given learning rate
        :param lr:
        :return:
        """

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return AdamW(optimizer_grouped_parameters, lr=lr)

    def save(self, model_file: AnyStr):
        """
        Saves the current model
        :return:
        """
        if not Path(model_file).parent.exists():
            Path(model_file).parent.mkdir(parents=True, exist_ok=True)
        if self.multi_gpu:
            save_model = self.model.module
        else:
            save_model = self.model
        torch.save(save_model.state_dict(), model_file)

    def load(self, model_file: AnyStr, new_classifier: bool = False):
        """
        Loads the model given by model_file
        :param model_file:
        :return:
        """
        if self.multi_gpu:
            model_dict = self.model.module.state_dict()
            load_model = self.model.module
        else:
            model_dict = self.model.state_dict()
            load_model = self.model
        if new_classifier:
            weights = {k: v for k, v in torch.load(model_file, map_location=lambda storage, loc: storage).items() if "classifier" not in k and "pooler" not in k}
            model_dict.update(weights)
            load_model.load_state_dict(model_dict)
        else:
            weights = torch.load(model_file, map_location=lambda storage, loc: storage)
            model_dict.update(weights)
            load_model.load_state_dict(model_dict)

    def freeze(self, exclude_params: List=[]):
        """
        Freeze the model weights
        :return:
        """
        for n,p in self.model.named_parameters():
            if n not in exclude_params:
                p.requires_grad = False


class TransformerClassificationTrainer(AbstractTransformerTrainer):
    """
    A class to encapsulate all of the training and evaluation of a
    transformer model for classification
    """
    def __init__(
            self,
            transformer_model: Union[AnyStr, torch.nn.Module],
            device: torch.device,
            num_labels: Union[int, List],
            multi_task: bool = False,
            tokenizer=None,
            multi_gpu: bool = False,
            ensemble_edu: bool = False,
            ensemble_sent: bool = False
    ):
        if type(num_labels) != list:
            num_labels = [num_labels]

        if type(transformer_model) == str:
            self.model_name = transformer_model
            # Create the model
            if multi_task:
                self.model = AutoTransformerMultiTask(transformer_model, num_labels).to(device)
            else:
                config = AutoConfig.from_pretrained(transformer_model, num_labels=num_labels[0])
                self.model = AutoModelForSequenceClassification.from_pretrained(transformer_model, config=config).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        else:
            self.model_name = 'custom'
            self.model = transformer_model
            self.tokenizer = tokenizer
            if tokenizer == None:
                print("WARNING: No tokenizer passed to trainer, incorrect padding token may be used.")

        if multi_gpu:
            self.model = DataParallelV2(self.model)
        self.device = device
        self.num_labels = num_labels
        self.multi_task = multi_task
        self.multi_gpu = multi_gpu
        self.ensemble_edu = ensemble_edu
        self.ensemble_sent = ensemble_sent

    def evaluate(
            self,
            validation_dset: Dataset,
            eval_averaging: AnyStr = 'binary',
            return_labels_logits: bool = False,
            sequence_modeling: bool = False
    ):
        """
        Runs a round of evaluation on the given dataset
        :param validation_dset:
        :return:
        """
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        # Create the validation evaluator
        validation_evaluator = ClassificationEvaluator(
            validation_dset,
            self.device,
            num_labels=self.num_labels[0],
            averaging=eval_averaging,
            pad_token_id=pad_token_id,
            sequence_modeling=sequence_modeling,
            multi_gpu=self.multi_gpu,
            ensemble_edu=self.ensemble_edu,
            ensemble_sent=self.ensemble_sent
        )
        return validation_evaluator.evaluate(self.model, return_labels_logits=return_labels_logits)

    def train(
            self,
            train_dset: Union[List, Dataset],
            validation_dset: Dataset,
            logger = None,
            lr: float = 3e-5,
            n_epochs: int = 2,
            batch_size: int = 8,
            weight_decay: float = 0.0,
            warmup_steps: int = 200,
            log_interval: int = 1,
            metric_name: AnyStr = 'accuracy',
            patience: int = 10,
            model_file: AnyStr = "model.pth",
            class_weights=None,
            use_scheduler: bool = True,
            eval_averaging: AnyStr = 'binary',
            lams: List = None,
            clip_grad: float = None,
            num_dataset_workers: int = 10
    ):
        if type(train_dset) != list:
            train_dset = [train_dset]
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        if lams is None:
            lams = [1.0] * len(train_dset)

        if (isinstance(class_weights, str) and class_weights == 'CRF'):
            collate_fn = partial(collate_sequence_batch_transformer, pad_token_id)
        else:
            collate_fn = partial(collate_batch_transformer, pad_token_id)

        # Create the loss function
        if class_weights is None:
            loss_fn = torch.nn.CrossEntropyLoss()
        elif (isinstance(class_weights, str) and class_weights == 'sample_based_weight'):
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            collate_fn = partial(collate_batch_transformer_with_weight, pad_token_id)
        elif isinstance(class_weights, str) and class_weights == 'balanced':
            # Calculate the weights
            # n_pos = sum(train_dset.dataset['label'] == 1)
            # neg_weight = n_pos / len(train_dset)
            # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, 1. - neg_weight]).to(self.device))
            if isinstance(train_dset[0], Subset):
                labels = train_dset[0].dataset.getLabels().astype(np.int64)
            else:
                labels = train_dset[0].getLabels().astype(np.int64)
            weight = torch.tensor(len(labels) / (self.num_labels[0] * np.bincount(labels)))
            weight = weight.type(torch.FloatTensor).to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        elif isinstance(class_weights, str) and class_weights == 'CRF':
            loss_fn = self.model.loss_fn
        else:
            loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).type(torch.FloatTensor).to(self.device))

        # Create the training dataloader(s)
        train_dls = [DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_dataset_workers
        ) for ds in train_dset]

        # Create the optimizer
        optimizer = self.create_optimizer(lr, weight_decay)

        total = sum(len(dl) for dl in train_dls)

        if use_scheduler:
            # Create the scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                warmup_steps,
                n_epochs * len(train_dls[0])#total
            )

        # Set up metric tracking
        best_metric = 0.0
        patience_counter = 0
        # Main training loop
        for ep in range(n_epochs):
            # Training loop
            dl_iters = [iter(dl) for dl in train_dls]
            dl_idx = list(range(len(dl_iters)))

            finished = [0] * len(dl_iters)
            i = 0
            with tqdm(total=total, desc="Training") as pbar:
                while sum(finished) < len(dl_iters):
                    random.shuffle(dl_idx)
                    for d in dl_idx:
                        task_dl = dl_iters[d]
                        try:
                            batch = next(task_dl)
                        except StopIteration:
                            finished[d] = 1
                            continue

                        self.model.train()
                        optimizer.zero_grad()

                        batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
                        input_ids = batch[0]
                        masks = batch[1]
                        labels = batch[2]
                        original_labels = labels

                        if not self.multi_task:

                            if self.ensemble_sent:
                                (logits, logits_sent) = self.model(input_ids, attention_mask=masks)
                            else:
                                outputs = self.model(input_ids, attention_mask=masks)
                                (logits,) = (outputs['logits'],)
                        else:
                            if self.ensemble_sent:
                                (logits, logits_sent) = self.model(input_ids, attention_mask=masks, task_num=d)
                            else:
                                outputs = self.model(input_ids, attention_mask=masks, task_num=d)
                                (logits,) = (outputs['logits'],)

                        # Calculate what the weight of the loss should be
                        loss_weight = lams[d]
                        if (isinstance(class_weights, str) and class_weights == 'sample_based_weight'):
                            sample_weight = batch[-1]
                            loss_weight *= sample_weight
                            loss = (loss_weight * loss_fn(logits.view(-1, self.num_labels[d]), labels.view(-1))).mean()
                        elif (isinstance(class_weights, str) and class_weights == 'CRF'):
                            seq_mask = batch[3]
                            loss = loss_fn(logits, labels, seq_mask)
                        else:
                            loss = loss_weight * loss_fn(logits.view(-1, self.num_labels[d]), labels.view(-1))
                            if self.ensemble_sent:
                                loss += loss_weight * loss_fn(logits_sent.view(-1, self.num_labels[d]), original_labels.view(-1))
                                loss /= 2

                        if self.multi_gpu:
                            loss = loss.mean()

                        if i % log_interval == 0 and logger is not None:
                            logger.log({"Loss": loss.item()})
                        loss.backward()
                        # Clip gradients
                        if clip_grad:
                          torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)

                        optimizer.step()
                        i += 1
                        pbar.update(1)
                    if use_scheduler:
                        scheduler.step()

            gc.collect()

            # Inline evaluation
            (val_loss, acc, P, R, F1) = self.evaluate(validation_dset, eval_averaging, sequence_modeling=(class_weights == 'CRF'))
            if metric_name == 'accuracy':
                metric = acc
            else:
                metric = F1
                if eval_averaging is None:
                    # Macro average if averaging is None
                    metric = sum(F1) / len(F1)

            print(f"{metric_name}: {metric}")

            if logger is not None:
                # Log
                logger.log({
                    'Validation accuracy': acc,
                    'Validation Precision': P,
                    'Validation Recall': R,
                    'Validation F1': F1,
                    'Validation loss': val_loss}
                )
            else:
                print({
                    'Validation accuracy': acc,
                    'Validation Precision': P,
                    'Validation Recall': R,
                    'Validation F1': F1,
                    'Validation loss': val_loss}
                )

            # Saving the best model and early stopping
            # if val_loss < best_loss:
            if metric > best_metric:
                best_metric = metric
                if logger != None:
                    logger.log({f"best_{metric_name}": metric})
                else:
                    print(f"Best {metric_name}: {metric}")
                self.save(model_file)
                patience_counter = 0
            else:
                patience_counter += 1
                # Stop training once we have lost patience
                if patience_counter == patience:
                    break

            gc.collect()

        self.load(model_file)


class TransformerMLMTrainer(AbstractTransformerTrainer):
    """
    A class to encapsulate all of the training and evaluation of a
    transformer model for masked language modeling + aux objectives
    """
    def __init__(
            self,
            transformer_model: Union[AnyStr, torch.nn.Module],
            device: torch.device,
            multi_gpu: bool = False,
            from_scratch: bool = False,
            sequence_model: bool = False
    ):
        if isinstance(transformer_model, str):
            self.model_name = transformer_model
            if from_scratch:
                self.model = AutoModelForMaskedLM.from_config(transformer_model).to(device)
            else:
                self.model = AutoModelForMaskedLM.from_pretrained(transformer_model).to(device)
        else:
            self.model = transformer_model.to(device)
            self.model_name = transformer_model.transformer_model

        if multi_gpu:
            self.model = DataParallelV2(self.model)
        self.multi_gpu = multi_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = device
        self.sequence_model = sequence_model

    def evaluate(self, validation_dset: Dataset, eval_averaging: AnyStr = 'binary'):
        """
        Runs a round of evaluation on the given dataset
        :param validation_dset:
        :return:
        """
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        else:
            pad_token_id = 0
        # Create the validation evaluator
        validation_evaluator = ClassificationEvaluator(
            validation_dset,
            self.device,
            pad_token_id=pad_token_id,
            mlm=True,
            multi_gpu=self.multi_gpu
        )
        return validation_evaluator.evaluate(self.model)

    def train(
            self,
            train_dset: Dataset,
            validation_dset: Dataset,
            logger = None,
            lr: float = 3e-5,
            n_epochs: int = 2,
            batch_size: int = 8,
            weight_decay: float = 0.0,
            warmup_steps: int = 200,
            log_interval: int = 1,
            metric_name: AnyStr = 'accuracy',
            patience: int = 10,
            model_file: AnyStr = "model.pth",
            class_weights=None,
            use_scheduler: bool = True,
            eval_averaging: AnyStr = 'binary',
            clip_grad: float = None,
            inline_evaluate: bool = True,
            save_steps: int = 0,
            with_cls_objective: bool = False,
            unmask_cls_objective: bool = False,
            cls_objective_only: bool = False,
            num_dataset_workers: int = 10,
            gradient_accumulation: int = 1
    ):

        collate_fn = partial(collate_batch_language_modeling, self.tokenizer)

        # Create the training dataloader(s)
        train_dl = DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_dataset_workers
        )

        # Create the optimizer
        optimizer = self.create_optimizer(lr, weight_decay)

        total = len(train_dl)

        if use_scheduler:
            # Create the scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                warmup_steps,
                n_epochs * len(train_dl)
            )

        # Set up metric tracking
        best_metric = float('-inf')
        patience_counter = 0
        # Main training loop
        for ep in range(n_epochs):
            # Training loop
            dl_iters = [iter(train_dl)]#[iter(dl) for dl in train_dls]
            dl_idx = list(range(len(dl_iters)))
            finished = [0] * len(dl_iters)
            i = 0
            loss = 0
            with tqdm(total=total, desc="Training") as pbar:
                while sum(finished) < len(dl_iters):
                    random.shuffle(dl_idx)
                    for d in dl_idx:
                        task_dl = dl_iters[d]
                        try:
                            batch = next(task_dl)
                        except StopIteration:
                            finished[d] = 1
                            continue

                        self.model.train()
                        #ipdb.set_trace()
                        batch = (batch['input_ids'], batch['labels'], batch['cls_labels'], batch['unmasked_ids'])
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids = batch[0]
                        #masks = batch[1]
                        labels = batch[1]
                        cls_labels = batch[2]
                        attention_mask = torch.tensor(input_ids != self.tokenizer.pad_token_id).to(self.device)
                        if cls_objective_only:
                            unmasked_ids = batch[3] if unmask_cls_objective or self.sequence_model else None
                            (cls_loss, cls_logits, mlm_logits) = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                cls_labels=cls_labels,
                                unmasked_ids=unmasked_ids
                            )
                            loss_curr = cls_loss
                        elif with_cls_objective:
                            unmasked_ids = batch[3] if unmask_cls_objective or self.sequence_model else None
                            (cls_loss, mlm_loss, cls_logits, mlm_logits) = self.model(
                                input_ids,
                                attention_mask=attention_mask,
                                labels=labels,
                                cls_labels=cls_labels,
                                unmasked_ids=unmasked_ids
                            )
                            loss_curr = cls_loss + mlm_loss
                        else:
                            unmasked_ids = batch[3] if unmask_cls_objective or self.sequence_model else None
                            (loss_curr, cls_logits, mlm_logits) = self.model(input_ids, attention_mask=attention_mask, labels=labels, unmasked_ids=unmasked_ids)
                        if self.multi_gpu:
                            loss += (1. / gradient_accumulation) * loss_curr.mean()
                        else:
                            loss += (1. / gradient_accumulation) * loss_curr

                        if i % log_interval == 0 and logger is not None:
                            logger.log({"Loss": loss.item()})


                        if i % gradient_accumulation == 0:
                            loss.backward()
                            # Clip gradients
                            if clip_grad:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                            optimizer.step()
                            optimizer.zero_grad()
                            loss = 0

                        i += 1
                        pbar.update(1)
                        if save_steps > 0 and i % save_steps == 0:
                            self.save_pretrained(model_file)
                    if use_scheduler:
                        scheduler.step()

            gc.collect()

            if inline_evaluate:
                # Inline evaluation
                val_loss = self.evaluate(validation_dset, eval_averaging)
                # Make negative since we're maximizing
                metric = -val_loss

                print(f"{metric_name}: {metric}")

                if logger is not None:
                    # Log
                    logger.log({
                        'Validation loss': val_loss}
                    )
                else:
                    print({
                        'Validation loss': val_loss}
                    )

                # Saving the best model and early stopping
                # if val_loss < best_loss:
                if metric > best_metric:
                    best_metric = metric
                    if logger != None:
                        logger.log({f"best_{metric_name}": metric})
                    else:
                        print(f"Best {metric_name}: {metric}")
                    self.save_pretrained(model_file)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    # Stop training once we have lost patience
                    if patience_counter == patience:
                        break

            gc.collect()

    def save_pretrained(self, model_dir):
        """
        HuggingFace save pretrained functionality
        :return:
        """
        self.tokenizer.save_pretrained(model_dir)
        if self.multi_gpu:
            self.model.module.save_pretrained(model_dir)
        else:
            self.model.save_pretrained(model_dir)

    def from_pretrained(self, model_dir):
        """
        HuggingFace from_pretrained
        :param model_dir:
        :return:
        """
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir)
        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
