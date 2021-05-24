import sys
import random
import numpy as np
import torch
import wandb
import argparse
from transformers import AutoTokenizer

from datareader import TransformerLanguageModelingDataset
from datareader import TransformerLanguageModelingMultiSentenceDataset
from model import AutoMaskedLMWithClassificationHead
from model import AutoMaskedLMWithSentenceSequenceClassificationHead
from train import TransformerMLMTrainer


def enforce_reproducibility(seed=1000):
    # Sets seed manually for both CPU and CUDA
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For atomic operations there is currently
    # no simple way to enforce determinism, as
    # the order of parallel operations is not known.
    # CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # System based
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Location of the training data", required=True, type=str)
    parser.add_argument("--validation_data", help="Location of the validation data", required=True, type=str)
    parser.add_argument("--model_name", help="The name of the model being tested. Can be a directory for a local model",
                        required=True, type=str)
    parser.add_argument("--model_dir", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)
    parser.add_argument("--gradient_accumulation", help="The number of gradient accumulation steps", type=int, default=1)
    parser.add_argument("--use_cite_objective", action="store_true", default=False,
                        help="Whether or not to use citation detection objective")
    parser.add_argument("--cite_only", action="store_true", default=False,
                        help="Whether or not to only train citation detection")
    parser.add_argument("--unmask_citation", action="store_true", default=False,
                        help="Whether or not to use unmasked inputs for citation detection objective")
    parser.add_argument("--mlm_weight", help="Weight to apply to the mlm loss", type=float, default=1.0)
    parser.add_argument("--cite_weight", help="Weight to apply to the citation detection loss", type=float, default=1.0)
    parser.add_argument("--multi_sentence", action="store_true", default=False,
                        help="Whether to train on single-sentence or multi-sentence task")



    args = parser.parse_args()

    seed = args.seed #1000
    lr = args.learning_rate #0.000001351
    warmup_steps = args.warmup_steps #300
    save_steps = 2000
    batch_size = args.batch_size #32
    weight_decay = args.weight_decay #0.01
    n_epochs = args.n_epochs #10
    # Always first
    enforce_reproducibility(seed)

    train_data_loc = args.train_data
    test_data_loc = args.validation_data #sys.argv[2]
    model_dir = args.model_dir #sys.argv[3]
    model_name = args.model_name #sys.argv[4]
    use_cite_objective = args.use_cite_objective
    unmask_citation = args.unmask_citation
    cite_only = args.cite_only

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    DatasetClass = TransformerLanguageModelingMultiSentenceDataset if args.multi_sentence else TransformerLanguageModelingDataset
    train_dset = DatasetClass(train_data_loc, tokenizer)
    valid_dset = DatasetClass(test_data_loc, tokenizer, ret_label=False)

    # model = AutoMaskedLMWithClassificationHead(model_name)
    # trainer = TransformerMLMTrainer(model, device, multi_gpu=True)

    labels = train_dset.getLabels().astype(np.int64)
    # weight = torch.tensor(len(labels) / (2 * np.bincount(labels)))
    # weight = weight.type(torch.FloatTensor).to(device)
    weight = None
    Model = AutoMaskedLMWithSentenceSequenceClassificationHead if args.multi_sentence else AutoMaskedLMWithClassificationHead
    model = Model(model_name, cls_weights=weight, loss_weights=[args.mlm_weight, args.cite_weight])
    trainer = TransformerMLMTrainer(model, device, multi_gpu=True)

    # Train it
    trainer.train(
        train_dset,
        valid_dset,
        weight_decay=weight_decay,
        model_file=model_dir,
        metric_name='loss',
        #logger=wandb,
        lr=lr,
        warmup_steps=warmup_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        save_steps=save_steps,
        with_cls_objective=use_cite_objective,
        unmask_cls_objective=unmask_citation,
        cls_objective_only=cite_only,
        gradient_accumulation=args.gradient_accumulation
    )
