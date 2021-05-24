from typing import AnyStr, List, Tuple, Callable
from torch.utils.data import Dataset
from functools import partial
from rouge_score import rouge_scorer
from tqdm import tqdm
from tokenizations import get_alignments
from nltk import Tree
from collections import defaultdict
import itertools as it
import pandas as pd
import numpy as np
import json
import torch
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorForLanguageModeling
from torch.nn.parallel._functions import Scatter
from torch.nn.parallel import DataParallel
from pu_learning import estimate_class_prior_probability
from pu_learning import get_negative_sample_weights
import ipdb


LABELS = {
    'non-check-worthy': 0,
    'check-worthy': 1
}

FINE_GRAINED_LABELS = {
    "introduction": 1, # Intro
    "abstract": 2, # Abstract
    "method": 3, # Methods
    "methods": 3,# Methods
    "results": 4, # Results
    "discussion": 5, # Discussion
    "discussions": 5, # Discussion
    "conclusion": 6, # Conclusion
    "conclusions": 6, # Conclusion
    "results and discussion": 5, # Discussion
    "related work": 7, # Background
    "experimental results": 4, # Results
    "literature review": 7, # Background
    "experiments": 4, # Results
    "background": 7, # Background
    "methodology": 3, # Methods
    "conclusions and future work": 6, # Conclusion
    "related works": 7, # Background
    "limitations": 5, # Discussion
    "procedure": 3, # Methods
    "material and methods": 3, # Methods
    "discussion and conclusion": 5, # Discussion
    "implementation": 3, # Methods
    "evaluation": 4, # Results
    "performance evaluation": 4, # Results
    "experiments and results": 4, # Results
    "overview": 1, # Introduction
    "experimental design": 3, # Methods
    "discussion and conclusions": 5, # Discussion
    "results and discussions": 5, # Discussion
    "motivation": 1, # Introduction
    "proposed method": 3, # Methods
    "analysis": 4, # Results
    "future work": 6, # Conclusion
    "results and analysis": 4, # Results
    "implementation details": 3 # Methods
}

def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            size = int(np.ceil(len(obj) / len(target_gpus)))
            lists = [obj[i * size:(i + 1) * size] for i in range(len(target_gpus))]
            for i in range(len(lists)):
                if isinstance(lists[i][0], torch.Tensor):
                    lists[i][0] = lists[i][0].to(target_gpus[i])
            return lists
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


class DataParallelV2(DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


def read_citation_detection_jsonl_single_line(jsonl_file: AnyStr, domain_list: List[AnyStr] = None):
    """
    Reads in the jsonl file for citation detection and returns a dataframe for a single-sentence dataset
    :param jsonl_file: Location of the dataset
    :return:
    """
    with open(jsonl_file) as f:
        data = [json.loads(l.strip()) for l in f]

    # Get sentences and labels
    if domain_list is None:
        dataset = [[s['text'], LABELS[s['label']]] for d in data for s in d['samples']]
    else:
        dataset = [[s['text'], LABELS[s['label']]] for d in data for s in d['samples'] if d['mag_field_of_study'][0] in domain_list]

    return pd.DataFrame(dataset, columns=['text', 'label'])


def read_citation_detection_jsonl(jsonl_file: AnyStr, domain_list: List[AnyStr] = None):
    """
    Reads in the jsonl file for citation detection and returns a list of dicts
    :param jsonl_file: Location of the dataset
    :return:
    """
    with open(jsonl_file) as f:
        dataset = [json.loads(l.strip()) for l in f]

    # Get sentences and labels
    if domain_list is not None:
        dataset = [d for d in dataset if d['mag_field_of_study'][0] in domain_list]

    return dataset


def text_to_batch_transformer(text: List, tokenizer: PreTrainedTokenizer) -> Tuple[List, List]:
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :return: A list of IDs and a mask
    """
    max_length = min(512, tokenizer.model_max_length)
    input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False) for t in text]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def text_to_sequence_batch_transformer(text: List, tokenizer: PreTrainedTokenizer) -> Tuple[List, List]:
    """Turn a list of text into a sequence of sentences separated by SEP token

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :return: A list of IDs and a mask
    """
    max_length = min(512, tokenizer.model_max_length)
    input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False) for t in text]
    input_ids = [[id_ for i,sent in enumerate(input_ids) for j,id_ in enumerate(sent) if (i == 0 or j != 0)][:tokenizer.model_max_length]]
    input_ids[0][-1] = tokenizer.sep_token_id


    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def text_to_batch_transformer_bpemb(text: List, tokenizer, max_seq_len: int = 512) -> Tuple[List, List]:
    """
    Creates a tokenized batch for input to a bilstm model
    :param text: A list of sentences to tokenize
    :param tokenizer: A tokenization function to use (i.e. fasttext)
    :return: Tokenized text as well as the length of the input sequence
    """
    # Some light preprocessing
    input_ids = [tokenizer.encode_ids_with_eos(t)[:max_seq_len] for t in text]
    for ids in input_ids:
      ids[-1] = tokenizer.EOS

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


def collate_batch_transformer(pad_token_id: int, input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    if isinstance(input_data[0][2], List):
        labels = [l for i in input_data for l in i[2]]
    else:
        labels = [i[2] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [pad_token_id] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels)


def collate_sequence_batch_transformer(pad_token_id: int, input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = [i[0][0] for i in input_data]
    masks = [i[1][0] for i in input_data]
    labels = [i[2] for i in input_data]
    seq_lens = [len(i[2]) for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [pad_token_id] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    max_seq_len = max(seq_lens)
    labels = [(l + [-1] * (max_seq_len - len(l))) for l in labels]
    seq_masks = [[1] * s + [0] * (max_seq_len - s) for s in seq_lens]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    assert (all(len(l) == max_seq_len for l in labels))

    return torch.tensor(input_ids), torch.tensor(masks), torch.tensor(labels), torch.tensor(seq_masks)


def collate_batch_transformer_with_weight(pad_token_id: int, input_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return collate_batch_transformer(pad_token_id, input_data) + (torch.tensor([i[3] for i in input_data]),)


def collate_batch_language_modeling(tokenizer: PreTrainedTokenizer, input_data: Tuple):
    lm_collator = DataCollatorForLanguageModeling(tokenizer)
    input_ids = [i[0] for i in input_data]
    lm_batch = lm_collator(input_ids)
    max_length = max([len(i) for i in input_ids])
    input_ids = [(i + [tokenizer.pad_token_id] * (max_length - len(i))) for i in input_ids]
    if isinstance(input_data[0][1], List):
        labels = [l for i in input_data for l in i[1]]
    else:
        labels = [i[1] for i in input_data]
    lm_batch['cls_labels'] = torch.tensor(labels)
    lm_batch['unmasked_ids'] = torch.tensor(input_ids)
    return lm_batch


class TransformerSingleSentenceDataset(Dataset):

    def __init__(self, jsonl_file: AnyStr, tokenizer, tokenizer_fn: Callable = text_to_batch_transformer, use_fine_labels: bool = False):

        self.dataset = read_citation_detection_jsonl_single_line(jsonl_file)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.use_fine_labels = use_fine_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset.iloc[idx].values
        # Calls the text_to_batch function
        input_ids, masks = self.tokenizer_fn([row[0]], self.tokenizer)
        label = row[1]
        return input_ids, masks, label

    def getLabels(self):
        return self.dataset.values[:,1]


class TransformerMultiSentenceDataset(Dataset):

    def __init__(self, jsonl_file: AnyStr, tokenizer, tokenizer_fn: Callable = text_to_sequence_batch_transformer, use_fine_labels: bool = False):

        self.dataset = read_citation_detection_jsonl(jsonl_file)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.use_fine_labels = use_fine_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        sents = [s['text'] for s in row['samples']]
        labels = [LABELS[s['label']] if s['label'] == 'non-check-worthy' or not self.use_fine_labels else FINE_GRAINED_LABELS[row['section_title'].lower()] for s in row['samples']]
        # Calls the text_to_batch function
        input_ids, masks = self.tokenizer_fn(sents, self.tokenizer)
        n_labels = sum(np.array(input_ids)[0] == self.tokenizer.sep_token_id)
        return input_ids, masks, labels[:n_labels]

    def getLabels(self, indices=None):
        if indices is None:
            return np.asarray([LABELS[s['label']] if s['label'] == 'non-check-worthy' or not self.use_fine_labels else
                               FINE_GRAINED_LABELS[d['section_title'].lower()] for d in self.dataset for s in
                               d['samples']])
        else:
            data = [self.dataset[i] for i in indices]
            return np.asarray([LABELS[s['label']] if s['label'] == 'non-check-worthy' or not self.use_fine_labels else
                               FINE_GRAINED_LABELS[d['section_title'].lower()] for d in data for s in d['samples']])


class CitationDetectionSingleDomainDataset(Dataset):

    def __init__(self, jsonl_files: List[AnyStr], tokenizer, domain: AnyStr, tokenizer_fn: Callable = text_to_batch_transformer):
        datasets = []
        for f in jsonl_files:
            datasets.append(read_citation_detection_jsonl_single_line(f, domain_list=[domain]))

        self.dataset = pd.concat(datasets)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.domain = domain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset.iloc[idx].values
        # Calls the text_to_batch function
        input_ids, masks = self.tokenizer_fn([row[0]], self.tokenizer)
        label = row[1]
        return input_ids, masks, label


class CitationDetectionSingleDomainMultiSentenceDataset(Dataset):

    def __init__(self, jsonl_files: List[AnyStr], tokenizer, domain: AnyStr, tokenizer_fn: Callable = text_to_sequence_batch_transformer):
        dataset = []
        for f in jsonl_files:
            dataset.extend(read_citation_detection_jsonl(f, domain_list=[domain]))

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.domain = domain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        sents = [s['text'] for s in row['samples']]
        labels = [LABELS[s['label']] for s in row['samples']]
        # Calls the text_to_batch function
        input_ids, masks = self.tokenizer_fn(sents, self.tokenizer)
        n_labels = sum(np.array(input_ids)[0] == self.tokenizer.sep_token_id)
        return input_ids, masks, labels[:n_labels]

    def getLabels(self, indices=None):
        if indices is None:
            return np.asarray([LABELS[s['label']] for d in self.dataset for s in d['samples']])
        else:
            data = [self.dataset[i] for i in indices]
            return np.asarray([LABELS[s['label']] for d in data for s in d['samples']])


class TransformerClassifierPUCDataset(Dataset):
    """Dataset reader for citation detection with positive unlabelled learning and
    positive-negative removal

    """

    def __init__(
            self,
            base_dataset: Dataset,
            validation_dataset: Dataset,
            base_network: torch.nn.Module,
            device: torch.device,
            scale: int = 1.0,
            puc: bool = False
    ):
        super(TransformerClassifierPUCDataset, self).__init__()

        # Take care of when a subset is used
        if type(base_dataset) == torch.utils.data.Subset:
            self.tokenizer = base_dataset.dataset.tokenizer
            indices = base_dataset.indices
            orig_dataset = base_dataset.dataset.dataset.copy()
            base_dataset = base_dataset.dataset
            base_dataset.dataset = orig_dataset.iloc[indices]
            base_dataset.dataset = base_dataset.dataset.reset_index(drop=True)
        else:
            self.tokenizer = base_dataset.tokenizer

        # Get dataloders for the train and validation datasets
        collate_fn = partial(collate_batch_transformer, self.tokenizer.pad_token_id)
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_fn)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_fn)

        if puc:
            # Estimate the class prior
            prior = estimate_class_prior_probability(base_network, train_dl, val_dl, device)
            print(prior)

        # Only look at negative samples
        original_dataset = base_dataset.dataset.copy()
        base_dataset.dataset = base_dataset.dataset[base_dataset.dataset['label'] == 0]

        # Get negatives weight, combine into one dataset and duplicate the negatives
        train_dl = torch.utils.data.DataLoader(base_dataset, batch_size=8, collate_fn=collate_fn)
        val_dl = torch.utils.data.DataLoader(validation_dataset, batch_size=8, collate_fn=collate_fn)
        neg_weights = get_negative_sample_weights(train_dl, val_dl, base_network, device)
        assert neg_weights.shape == base_dataset.dataset.shape, "Should have double the number of negative sample weights"

        positives = original_dataset[original_dataset['label'] == 1]
        positives['weight'] = [1.] * positives.shape[0]
        keep_examples = np.asarray([True] * neg_weights.shape[0])
        if puc:
            # Keep adding examples until p(y=1) equals our estimate
            ordered_idx = np.argsort(neg_weights[:,1])[::-1]
            n_convert = int(np.ceil(original_dataset.shape[0] * prior)) - positives.shape[0]
            keep_examples[ordered_idx[:n_convert]] = False
            converted_positives = base_dataset.dataset[~keep_examples].copy()
            converted_positives['label'] = [1] * converted_positives.shape[0]
            converted_positives['weight'] = [1.] * converted_positives.shape[0]

        kept_negatives = base_dataset.dataset[keep_examples].copy()
        kept_negatives_plus = kept_negatives.copy()
        kept_negatives_plus['label'] = [1] * kept_negatives_plus.shape[0]
        kept_negatives['weight'] = neg_weights[keep_examples, 0]
        kept_negatives_plus['weight'] = neg_weights[keep_examples, 1]

        if puc:
            self.dataset = pd.concat([positives, kept_negatives, kept_negatives_plus, converted_positives],
                                 ignore_index=True)
        else:
            self.dataset = pd.concat([positives, kept_negatives, kept_negatives_plus],
                                     ignore_index=True)
        self.scale = scale

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item) -> Tuple:
        row = self.dataset.iloc[item].values
        input_ids, mask = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        weight = self.scale * row[2]
        return input_ids, mask, label, weight, item


class TransformerLanguageModelingDataset(Dataset):

    def __init__(self, jsonl_file: AnyStr, tokenizer, tokenizer_fn: Callable = text_to_batch_transformer, ret_label: bool = True):

        if jsonl_file[-4:] == '.csv':
            self.dataset = pd.read_csv(jsonl_file)
        else:
            self.dataset = read_citation_detection_jsonl_single_line(jsonl_file)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.ret_label = ret_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset.iloc[idx].values
        # Calls the text_to_batch function
        input_ids, masks = self.tokenizer_fn([row[0]], self.tokenizer)
        label = row[1]
        if self.ret_label:
            return input_ids[0], label
        else:
            return input_ids[0]

    def getLabels(self):
        return self.dataset.values[:,1]


class TransformerLanguageModelingMultiSentenceDataset(Dataset):

    def __init__(self, jsonl_file: AnyStr, tokenizer, tokenizer_fn: Callable = text_to_sequence_batch_transformer, ret_label: bool = True):

        self.dataset = read_citation_detection_jsonl(jsonl_file)
        self.tokenizer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.ret_label = ret_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        sents = [s['text'] for s in row['samples']]
        labels = [LABELS[s['label']] for s in row['samples']]
        # Calls the text_to_batch function
        input_ids, masks = self.tokenizer_fn(sents, self.tokenizer)
        if self.ret_label:
            n_labels = sum(np.array(input_ids)[0] == self.tokenizer.sep_token_id)
            return input_ids[0], labels[:n_labels]
        else:
            return input_ids[0]

    def getLabels(self):
        return np.asarray([LABELS[s['label']] for d in self.dataset for s in d['samples']])
