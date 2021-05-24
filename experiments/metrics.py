import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from sklearn.metrics import roc_curve, auc
from functools import partial
from typing import Tuple, List, Callable, AnyStr
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from collections import Counter
from transformers.data.data_collator import DataCollatorForLanguageModeling
import ipdb

from datareader import collate_batch_transformer
from datareader import collate_sequence_batch_transformer


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return np.sum(preds == labels).astype(np.float32) / float(labels.shape[0])


def acc_f1(preds: List, labels: List, averaging: AnyStr = 'binary') -> Tuple[float, float, float, float]:
    acc = accuracy(preds, labels)
    P, R, F1, _ = precision_recall_fscore_support(labels, preds, average=averaging)

    return acc,P,R,F1


def average_precision(labels: np.ndarray, order: np.ndarray) -> float:
    """
    Calculates the average precision of a ranked list
    :param labels: True labels of the items
    :param order: The ranking order
    :return: Average precision
    """
    j = 0
    ap = 0
    for i, v in enumerate(labels[order]):
        if v == 1:
            j += 1
            ap += j / (i + 1)
    return ap / j

def plot_label_distribution(labels: np.ndarray, logits: np.ndarray) -> matplotlib.figure.Figure:
    """ Plots the distribution of labels in the prediction

    :param labels: Gold labels
    :param logits: Logits from the model
    :return: None
    """
    predictions = np.argmax(logits, axis=-1)
    labs, counts = zip(*list(sorted(Counter(predictions).items(), key=lambda x: x[0])))

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.bar(labs, counts, width=0.2)
    ax.set_xticks(labs, [str(l) for l in labs])
    ax.set_ylabel('Count')
    ax.set_xlabel("Label")
    ax.set_title("Prediction distribution")
    return fig


class ClassificationEvaluator:
    """Wrapper to evaluate a model for classification tasks

    """

    def __init__(
            self,
            dataset: Dataset,
            device: torch.device,
            num_labels: int = 2,
            averaging: AnyStr = 'binary',
            pad_token_id: int = None,
            mlm: bool = False,
            multi_gpu: bool = False,
            sequence_modeling: bool = False,
            ensemble_edu: bool = False,
            ensemble_sent: bool = False
    ):
        self.dataset = dataset
        if isinstance(dataset, Subset):
            self.all_labels = list(dataset.dataset.getLabels(dataset.indices))
        else:
            self.all_labels = dataset.getLabels()
        if sequence_modeling:
            collator = collate_sequence_batch_transformer
        else:
            collator = collate_batch_transformer

        if mlm:
            collate_fn = DataCollatorForLanguageModeling(dataset.tokenizer)
        elif pad_token_id is None:
            collate_fn = partial(collator, dataset.tokenizer.pad_token_id)
        else:
            collate_fn = partial(collator, pad_token_id)

        self.dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=collate_fn
        )
        self.device = device
        self.averaging = averaging
        self.num_labels = num_labels
        self.mlm = mlm
        self.pad_token_id = pad_token_id
        self.multi_gpu = multi_gpu
        self.sequence_modeling = sequence_modeling
        self.ensemble_edu = ensemble_edu
        self.ensemble_sent = ensemble_sent

    def predict(
            self,
            model: torch.nn.Module
    ) -> Tuple:
        model.eval()
        with torch.no_grad():
            labels_all = []
            logits_all = []
            losses_all = []
            preds_all = []
            for batch in tqdm(self.dataloader, desc="Evaluation"):
                if isinstance(batch, dict):
                    batch = [batch['input_ids'], batch['labels']]
                batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)
                input_ids = batch[0]
                if self.mlm:
                    labels = batch[1]
                    masks = torch.tensor(input_ids != self.pad_token_id).to(self.device)
                else:
                    masks = batch[1]
                    labels = batch[2]

                if self.sequence_modeling:
                    # For CRF
                    seq_mask = batch[3]
                    outputs = model(input_ids, attention_mask=masks, seq_mask=seq_mask, labels=labels)
                    labels_all.extend([l for seq in list(labels.detach().cpu().numpy()) for l in seq if l != -1])
                else:
                    input_dict = {'input_ids': input_ids, 'attention_mask': masks}
                    if self.multi_gpu and not self.mlm:
                        outputs = model(**input_dict)
                        (logits,) = (outputs['logits'],)
                        outputs = (torch.nn.CrossEntropyLoss()(outputs[0].reshape(-1, self.num_labels), labels.reshape(-1)),) + outputs
                    else:
                        input_dict['labels'] = labels
                        outputs = model(**input_dict)
                        (loss,logits) = (outputs['loss'],outputs['logits'])

                    if self.ensemble_edu:
                        edu_sizes = input_dict['edu_sizes']
                        # Get final label and ensemble the edu logits
                        sentence_sizes = [len(sent) for seq in edu_sizes for sent in seq]
                        logits_split = torch.split(outputs[1], sentence_sizes)
                        labels_split = torch.split(labels, sentence_sizes)
                        labels = torch.tensor([l[0] for l in labels_split])
                        final_logits = []
                        for j,logits in enumerate(logits_split):
                            probs = torch.nn.Softmax(-1)(logits)
                            # Take the average probability
                            if len(probs.shape) > 1:
                                probs = probs.mean(0)
                            if self.ensemble_sent:
                                probs_sent = torch.nn.Softmax(-1)(outputs[2][j])
                                probs = (probs + probs_sent) / 2
                            final_logits.append(probs.unsqueeze(0))
                        outputs = (outputs[0], torch.cat(final_logits, 0))
                    elif self.ensemble_sent:
                        final_logits = []
                        for ledu, lsent in zip(outputs[1], outputs[2]):
                            probs = torch.nn.Softmax(-1)(torch.cat([ledu.unsqueeze(0), lsent.unsqueeze(0)], 0))
                            final_logits.append(probs.mean(0).unsqueeze(0))
                        outputs = (outputs[0], torch.cat(final_logits, 0))

                    labels_all.extend(list(labels.detach().cpu().numpy()))
                if not self.mlm:
                    # For memory issues
                    logits_all.extend(list(outputs[1].detach().cpu().numpy()))

                if self.multi_gpu and self.mlm:
                    losses_all.append(outputs[0].mean().item())
                else:
                    losses_all.append(outputs[0].item())
                if self.sequence_modeling:
                    masks = (labels != -1)
                    preds = model.decode(outputs[1], masks)
                    preds_all.extend([t for seq in preds for t in seq])
                elif not self.mlm:
                    preds = np.argmax(outputs[1].detach().cpu().numpy().reshape(-1, self.num_labels), axis=-1)
                    preds_all.extend([p for p in preds])
        if not self.mlm:
            assert len(labels_all) == len(self.all_labels)
            assert len(logits_all) == len(self.all_labels)
            assert len(preds_all) == len(self.all_labels)
        return labels_all, logits_all, losses_all, preds_all

    def roc_auc(self, model: torch.nn.Module):
        labels_all, logits_all, losses_all = self.predict(model)
        logits = np.asarray(logits_all).reshape(-1, self.num_labels)
        labels = np.asarray(labels_all).reshape(-1)
        fpr, tpr, _ = roc_curve(labels, logits[:, 1])
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def evaluate(
            self,
            model: torch.nn.Module,
            plot_callbacks: List[Callable] = [],
            return_labels_logits: bool = False
    ) -> Tuple:
        """Collect evaluation metrics on this dataset

        :param model: The pytorch model to evaluate
        :param plot_callbacks: Optional function callbacks for plotting various things
        :return: (Loss, Accuracy, Precision, Recall, F1)
        """
        labels_all, logits_all, losses_all, preds_all = self.predict(model)
        loss = sum(losses_all) / len(losses_all)
        if self.mlm:
            ret_vals = loss
        else:
            if self.averaging == 'binary' and self.num_labels > 2:
                preds_all = [p if p == 0 else 1 for p in preds_all]
            acc,P,R,F1 = acc_f1(np.asarray(preds_all), np.asarray(labels_all).reshape(-1), averaging=self.averaging)
            ret_vals = (loss, acc, P, R, F1)

            # Plotting
            plots = []
            for f in plot_callbacks:
                plots.append(f(labels_all, logits_all))

            if len(plots) > 0:
                ret_vals = (loss, acc, P, R, F1), plots

        # Labels and logits
        if return_labels_logits:
            ret_vals = ret_vals + (labels_all, logits_all, preds_all,)

        return ret_vals
