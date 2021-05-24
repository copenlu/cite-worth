import os
import random
import numpy as np
import torch
import wandb
from bpemb import BPEmb
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import ipdb
from torch.utils.data import random_split

from datareader import CitationDetectionSingleDomainDataset, TransformerMultiSentenceDataset, \
    text_to_sequence_batch_transformer, CitationDetectionSingleDomainMultiSentenceDataset
from datareader import TransformerClassifierPUCDataset
from datareader import text_to_batch_transformer_bpemb
from datareader import text_to_batch_transformer
from train import TransformerClassificationTrainer
from model import TransformerClassifier, AutoTransformerForSentenceSequenceModeling


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


def get_transformer(ff_dim: int, n_layers: int, n_heads: int, dropout_prob: float):
    """
    Creates a new transformer and tokenizer using the given parameters
    :param ff_dim:
    :param n_layers:
    :param n_heads:
    :param dropout_prob:
    :return:
    """
    # Load english model with 25k word-pieces
    tokenizer = BPEmb(lang='en', dim=300, vs=25000)
    # Extract the embeddings and add a randomly initialized embedding for our extra [PAD] token
    pretrained_embeddings = np.concatenate([tokenizer.emb.vectors, np.zeros(shape=(1, 300))], axis=0)
    # Extract the vocab and add an extra [PAD] token
    vocabulary = tokenizer.emb.index2word + ['[PAD]']
    tokenizer.pad_token_id = len(vocabulary) - 1

    model = TransformerClassifier(
        torch.tensor(pretrained_embeddings).type(torch.FloatTensor),
        ff_dim=ff_dim,
        d_model=300,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout_prob=dropout_prob
    ).to(device)

    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Location of the training data", required=True, type=str)
    parser.add_argument("--validation_data", help="Location of the validation data", required=True, type=str)
    parser.add_argument("--test_data", help="Location of the test data", required=True, type=str)
    parser.add_argument("--train_domain", help="The domain to train on", required=True, type=str)
    parser.add_argument("--test_domain", help="The domain to test on ", required=True, type=str)
    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--tag", help="A tag to give this run (for wandb)", required=True, type=str)
    parser.add_argument("--model_name", help="The name of the model being tested. Can be a directory for a local model",
                        required=True, type=str)
    parser.add_argument("--model_dir", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--learning_rate", help="The learning rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", help="Amount of weight decay", type=float, default=0.0)
    parser.add_argument("--dropout_prob", help="The dropout probability", type=float, default=0.1)
    parser.add_argument("--n_epochs", help="The number of epochs to run", type=int, default=2)
    parser.add_argument("--ff_dim", help="The feedforward dimensionality of the network", type=int, default=256)
    parser.add_argument("--n_heads", help="Number of attention heads", type=int, default=1)
    parser.add_argument("--n_layers", help="Number of transformer layers", type=int, default=1)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--warmup_steps", help="The number of warmup steps", type=int, default=200)
    parser.add_argument("--balance_class_weight", action="store_true", default=False, help="Whether or not to use balanced class weights")
    parser.add_argument("--pu_learning", help="The type of PU learning to do", type=str, default=None)
    parser.add_argument("--pu_learning_model", help="A model to initialize for PU learning", type=str, default=None)
    parser.add_argument("--sequence_model", help="Indicates to use a sequence modeling setup", action="store_true", default=False)


    args = parser.parse_args()

    seed = args.seed
    lr = args.learning_rate
    weight_decay = args.weight_decay
    warmup_steps = args.warmup_steps
    dropout_prob = args.dropout_prob
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    class_weights = 'balanced' if args.balance_class_weight else None
    model_name = args.model_name
    pu_learning = args.pu_learning
    use_scheduler = True

    config = {
            "epochs": n_epochs,
            "learning_rate": lr,
            "warmup": warmup_steps,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "bert_model": model_name,
            "seed": seed,
            "use_scheduler": use_scheduler,
            "balance_class_weight": args.balance_class_weight,
            "pu_learning": pu_learning,
            "train_domain": args.train_domain,
            "test_domain": args.test_domain
        }

    # Always first
    enforce_reproducibility(seed)

    train_data_loc = args.train_data
    valid_data_loc = args.validation_data
    test_data_loc = args.test_data

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")
    DatareaderClass = CitationDetectionSingleDomainDataset
    train_dset = None
    if model_name == 'scratch_transformer':
        ff_dim = args.ff_dim
        n_heads = args.n_heads
        n_layers = args.n_layers
        model, tokenizer = get_transformer(ff_dim, n_layers, n_heads, dropout_prob)
        tokenizer_fn = text_to_batch_transformer_bpemb
        use_scheduler = False
        config['ff_dim'] = ff_dim
        config['n_heads'] = n_heads
        config['n_layers'] = n_layers
        config['use_scheduler'] = False
    elif args.sequence_model:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        DatareaderClass = CitationDetectionSingleDomainMultiSentenceDataset
        tokenizer_fn = text_to_sequence_batch_transformer
        model = AutoTransformerForSentenceSequenceModeling(
            model_name,
            num_labels=2,
            sep_token_id=tokenizer.sep_token_id
        ).to(device)
    else:
        model = model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_fn = text_to_batch_transformer

    # Create the datasets and trainer
    #valid_dset = TransformerSingleSentenceDataset(valid_data_loc, tokenizer, tokenizer_fn=tokenizer_fn)
    trainer = TransformerClassificationTrainer(model, device, num_labels=2, tokenizer=tokenizer)
    # if pu_learning != None:
    #     assert args.pu_learning_model is not None, "Must pass a model to initialize for PU learning"
    #     # Load the model
    #     base_trainer = TransformerClassificationTrainer(model, device, num_labels=2)
    #     base_trainer.load(args.pu_learning_model)
    #     #Create the training dataset
    #     base_train_dset = TransformerSingleSentenceDataset(train_data_loc, tokenizer, tokenizer_fn=tokenizer_fn)
    #     train_dset = TransformerClassifierPUCDataset(base_train_dset, valid_dset, base_trainer.model, device, puc=(pu_learning == 'puc'))
    #     class_weights = 'sample_based_weight'
    # else:
    #     train_dset = TransformerSingleSentenceDataset(train_data_loc, tokenizer, tokenizer_fn=tokenizer_fn)
    base_dset = DatareaderClass(
        [args.train_data, args.validation_data, args.test_data],
        tokenizer,
        tokenizer_fn=tokenizer_fn,
        domain=args.train_domain
    )
    if args.train_domain == args.test_domain:
        train_size = int(len(base_dset) * 0.8)
        val_size = int((len(base_dset) - train_size) / 2)
        test_size = len(base_dset) - (train_size + val_size)
        base_dsets = random_split(base_dset, [train_size, val_size, test_size])
        train_dset = base_dsets[0]
        valid_dset = base_dsets[1]
    else:
        train_size = int(len(base_dset) * 0.8)
        base_dsets = random_split(base_dset, [train_size, len(base_dset) - train_size])
        train_dset = base_dsets[0]
        valid_dset = base_dsets[1]


    # wandb initialization
    run = wandb.init(
        project="scientific-citation-detection",
        name=args.run_name,
        config=config,
        reinit=True,
        tags=[args.tag, 'domain-adaptation-all']
    )

    # Create a new directory to save the model
    wandb_path = Path(wandb.run.dir)
    model_dir = f"{args.model_dir}/{wandb_path.name}"
    # Create save directory for model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Train it
    trainer.train(
        train_dset,
        valid_dset,
        weight_decay=weight_decay,
        model_file=f"{model_dir}/model_{args.train_domain}_{args.test_domain}.pth",
        class_weights=class_weights,
        metric_name='F1',
        logger=wandb,
        lr=lr,
        warmup_steps=warmup_steps,
        n_epochs=n_epochs,
        batch_size=batch_size,
        use_scheduler=use_scheduler
    )

    # Test on the final test data
    if args.train_domain == args.test_domain:
        test_dset = base_dsets[2]
    else:
        test_dset = DatareaderClass(
            [args.train_data, args.validation_data, args.test_data],
            tokenizer,
            tokenizer_fn=tokenizer_fn,
            domain=args.test_domain
        )
    (test_loss, acc, P, R, F1) = trainer.evaluate(test_dset)
    wandb.run.summary[f'test-loss'] = test_loss
    wandb.run.summary[f'test-acc'] = acc
    wandb.run.summary[f'test-P'] = P
    wandb.run.summary[f'test-R'] = R
    wandb.run.summary[f'test-F1'] = F1

    print(F1)