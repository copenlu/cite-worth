import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import AnyStr, List
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
import ipdb


class CNN(torch.nn.Module):
    def __init__(
            self,
            embeddings: np.array,
            n_labels: int,
            in_channels: int,
            out_channels: int,
            kernel_heights: List[int],
            stride: int = 1,
            padding: int = 0,
            dropout: float = 0.0
    ):
        super(CNN, self).__init__()

        self.embedding = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight = torch.nn.Parameter(torch.tensor(embeddings, dtype=torch.float), requires_grad=True)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(in_channels, out_channels,
                                                    (kernel_height, embeddings.shape[1]),
                                                    stride, padding)
                            for kernel_height in kernel_heights])

        output_units = n_labels #if n_labels > 2 else 1
        self.final = torch.nn.Linear(len(kernel_heights) * out_channels, output_units)

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)  # conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2)  # maxpool_out.size() = (batch_size, out_channels)

        return max_out

    def forward(self, input, attention_mask):
        input = self.embedding(input) * attention_mask.unsqueeze(-1) # Zero out padding
        # input.size() = (batch_size, num_seq, embedding_length)
        input = input.unsqueeze(1)
        # input.size() = (batch_size, 1, num_seq, embedding_length)
        input = self.dropout(input)

        conv_out = [self.conv_block(input, self.conv_layers[i]) for i in range(len(self.conv_layers))]
        all_out = torch.cat(conv_out, 1)
        # all_out.size() = (batch_size, num_kernels*out_channels)
        fc_in = self.dropout(all_out)
        # fc_in.size()) = (batch_size, num_kernels*out_channels)
        output = self.final(fc_in)

        return (output,)


class TransformerClassifier(nn.Module):
    """
        Multiple transformers for different domains
        """

    def __init__(
            self,
            pretrained_embeddings,
            ff_dim: int = 2048,
            d_model: int = 768,
            n_layers: int = 6,
            n_heads: int = 6,
            n_classes: int = 2,
            dropout_prob: float = 0.1,
            **kwargs
    ):
        super(TransformerClassifier, self).__init__()
        self.ff_dim = ff_dim
        self.d_model = d_model

        self.embeddings = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1)

        self.projection = nn.Linear(pretrained_embeddings.shape[1], d_model, bias=False)

        self.xformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                dim_feedforward=ff_dim,
                dropout=dropout_prob
            ),
            n_layers
        )

        # final classifier layers (d_model x n_classes)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.LongTensor,
            labels: torch.LongTensor = None
    ):
        embs = self.embeddings(input_ids)
        #embs = self.projection(embs)
        # Sequence length first
        inputs = embs.permute(1, 0, 2)
        # Flags the 0s instead of 1s
        masks = attention_mask == 0

        output = self.xformer(inputs, src_key_padding_mask=masks)
        pooled_output = output[0]
        logits = self.classifier(pooled_output)
        outputs = (logits,)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs = (loss,) + outputs

        return outputs


class AutoTransformerMultiTask(nn.Module):
    """
    Implements a transformer with multiple classifier heads for multi-task training
    """
    def __init__(self, transformer_model: AnyStr, task_num_labels: List):
        super(AutoTransformerMultiTask, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in task_num_labels])
        self.act = nn.Tanh()

        # Create the classifier heads
        self.task_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, n_labels) for n_labels in task_num_labels])
        self.task_num_labels = task_num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            task_num=0,
            lam=1.0
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]
        if len(sequence_output.shape) == 3:
            sequence_output = sequence_output[:,0]
        assert sequence_output.shape[0] == input_ids.shape[0]
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling[task_num](sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.task_classifiers[task_num](pooled_output)

        outputs = (logits,)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.task_num_labels[task_num]), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class AutoMaskedLMWithClassificationHead(nn.Module):
    """
    Adds an extra classification layer to a transformer language model head
    """
    def __init__(self, transformer_model: AnyStr, n_labels: int = 2, from_scratch: bool = False, cls_weights: torch.FloatTensor = None, loss_weights = [1.0, 1.0]):
        super(AutoMaskedLMWithClassificationHead, self).__init__()
        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        if from_scratch:
            self.mlm_model = AutoModelForMaskedLM.from_config(config)
        else:
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(transformer_model)

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, n_labels)
        self.transformer_model = transformer_model
        self.n_labels = n_labels
        self.cls_weights = cls_weights
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=cls_weights)
        self.loss_weights = loss_weights

    def forward(self, input_ids, attention_mask, labels=None, cls_labels=None, unmasked_ids=None):

        mlm_outputs = self.mlm_model(input_ids, attention_mask, output_hidden_states=True)
        mlm_logits = mlm_outputs[0]
        if unmasked_ids == None:
            # Get cls embedding of the last hidden layer
            cls_output = mlm_outputs[1][-1][:,0,:]
        else:
            unmasked_outputs = self.mlm_model(unmasked_ids, attention_mask, output_hidden_states=True)
            cls_output = unmasked_outputs[1][-1][:, 0, :]
        # Pass through pooler
        cls_output = self.dropout(self.act(self.pooler(cls_output)))
        cls_logits = self.classifier(cls_output)
        outputs = (cls_logits, mlm_logits)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            mlm_loss = self.loss_weights[0] * loss_fn(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (mlm_loss,) + outputs

        if cls_labels is not None:
            loss_fn = self.cls_loss_fn
            cls_loss = self.loss_weights[1] * loss_fn(cls_logits.view(-1, self.n_labels), cls_labels.view(-1))
            outputs = (cls_loss,) + outputs

        return outputs

    def save_pretrained(self, model_dir):
        self.mlm_model.save_pretrained(model_dir)


class AutoTransformerForSentenceSequenceModeling(nn.Module):
    """
       Implements a transformer which performs sequence classification on a sequence of sentences
    """

    def __init__(self, transformer_model: AnyStr, num_labels: int = 2, sep_token_id: int = 2):
        super(AutoTransformerForSentenceSequenceModeling, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()

        # Create the classifier heads
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sep_token_id = sep_token_id

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            lam=1.0
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask,

        )

        # Gather all of the SEP hidden states
        hidden_states = outputs['last_hidden_state'].reshape(-1, self.config.hidden_size)
        locs = (input_ids == self.sep_token_id).view(-1)
        #(n * seq_len x d) -> (n * sep_len x d)
        sequence_output = hidden_states[locs]
        assert sequence_output.shape[0] == sum(locs)
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling(sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.classifier(pooled_output)

        outputs = (logits,)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return {'loss': loss, 'logits': logits}


class AutoMaskedLMWithSentenceSequenceClassificationHead(nn.Module):
    """
    Adds an extra classification layer to a transformer language model head
    """
    def __init__(self, transformer_model: AnyStr, n_labels: int = 2, from_scratch: bool = False, cls_weights: torch.FloatTensor = None, loss_weights = [1.0, 1.0]):
        super(AutoMaskedLMWithSentenceSequenceClassificationHead, self).__init__()
        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        if from_scratch:
            self.mlm_model = AutoModelForMaskedLM.from_config(config)
        else:
            self.mlm_model = AutoModelForMaskedLM.from_pretrained(transformer_model)

        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, n_labels)
        self.transformer_model = transformer_model
        self.n_labels = n_labels
        self.cls_weights = cls_weights
        self.cls_loss_fn = nn.CrossEntropyLoss(weight=cls_weights)
        self.loss_weights = loss_weights

    def forward(self, input_ids, attention_mask, labels=None, cls_labels=None, unmasked_ids=None):

        mlm_outputs = self.mlm_model(input_ids, attention_mask, output_hidden_states=True)
        mlm_logits = mlm_outputs['logits']
        if unmasked_ids is not None:
            # Gather all of the SEP hidden states
            hidden_states = mlm_outputs['hidden_states'][-1].reshape(-1, self.config.hidden_size)
            locs = (unmasked_ids == self.sep_token_id).view(-1)
            # (n * seq_len x d) -> (n * sep_len x d)
            sequence_output = hidden_states[locs]
            assert sequence_output.shape[0] == sum(locs)
            assert sequence_output.shape[1] == self.config.hidden_size
            # if unmasked_ids == None:
            #     # Get cls embedding of the last hidden layer
            #     cls_output = mlm_outputs[1][-1][:,0,:]
            # else:
            #     unmasked_outputs = self.mlm_model(unmasked_ids, attention_mask, output_hidden_states=True)
            #     cls_output = unmasked_outputs[1][-1][:, 0, :]
            # Pass through pooler
            cls_output = self.dropout(self.act(self.pooler(sequence_output)))
            cls_logits = self.classifier(cls_output)
        else:
            cls_logits = None
        outputs = (cls_logits, mlm_logits)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            mlm_loss = self.loss_weights[0] * loss_fn(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (mlm_loss,) + outputs

        if cls_labels is not None:
            loss_fn = self.cls_loss_fn
            cls_loss = self.loss_weights[1] * loss_fn(cls_logits.view(-1, self.n_labels), cls_labels.view(-1))
            outputs = (cls_loss,) + outputs

        return outputs

    def save_pretrained(self, model_dir):
        self.mlm_model.save_pretrained(model_dir)


class AutoTransformerForSentenceSequenceModelingMultiTask(nn.Module):
    """
       Implements a transformer which performs sequence classification on a sequence of sentences
    """

    def __init__(self, transformer_model: AnyStr, task_num_labels: List, sep_token_id: int = 2):
        super(AutoTransformerForSentenceSequenceModelingMultiTask, self).__init__()

        config = AutoConfig.from_pretrained(transformer_model)
        self.config = config
        self.xformer = AutoModel.from_pretrained(transformer_model, config=config)

        # Pooling layers
        self.pooling = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in task_num_labels])
        self.act = nn.Tanh()

        # Create the classifier heads
        self.task_classifiers = nn.ModuleList([nn.Linear(config.hidden_size, n_labels) for n_labels in task_num_labels])
        self.task_num_labels = task_num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sep_token_id = sep_token_id

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            task_num=0,
            lam=1.0
    ):
        outputs = self.xformer(
            input_ids,
            attention_mask=attention_mask,

        )

        # Gather all of the SEP hidden states
        hidden_states = outputs['last_hidden_state'].reshape(-1, self.config.hidden_size)
        locs = (input_ids == self.sep_token_id).view(-1)
        # (n * seq_len x d) -> (n * sep_len x d)
        sequence_output = hidden_states[locs]
        assert sequence_output.shape[0] == sum(locs)
        assert sequence_output.shape[1] == self.config.hidden_size

        pooled_output = self.pooling[task_num](sequence_output)
        pooled_output = self.dropout(self.act(pooled_output))

        logits = self.task_classifiers[task_num](pooled_output)

        outputs = (logits,)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = lam * loss_fct(logits.view(-1, self.task_num_labels[task_num]), labels.view(-1))
            outputs = (loss,) + outputs

        return {'loss': loss, 'logits': logits}
