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

from datareader import TransformerSingleSentenceDataset
from datareader import TransformerClassifierPUCDataset
from datareader import text_to_batch_transformer_bpemb
from datareader import text_to_batch_transformer
from train import TransformerClassificationTrainer
from model import CNN


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


def get_cnn(in_channels, out_channels, kernel_heights, stride, padding, dropout_prob):
    """
    Creates a new CNN and tokenizer using the given parameters
    :return:
    """
    # Load english model with 25k word-pieces
    tokenizer = BPEmb(lang='en', dim=300, vs=25000)
    # Extract the embeddings and add a randomly initialized embedding for our extra [PAD] token
    pretrained_embeddings = np.concatenate([tokenizer.emb.vectors, np.zeros(shape=(1, 300))], axis=0)
    # Extract the vocab and add an extra [PAD] token
    vocabulary = tokenizer.emb.index2word + ['[PAD]']
    tokenizer.pad_token_id = len(vocabulary) - 1

    model = CNN(
        torch.tensor(pretrained_embeddings).type(torch.FloatTensor),
        n_labels=2,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_heights=kernel_heights,
        stride=stride,
        padding=padding,
        dropout=dropout_prob
    ).to(device)

    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Location of the training data", required=True, type=str)
    parser.add_argument("--validation_data", help="Location of the validation data", required=True, type=str)
    parser.add_argument("--test_data", help="Location of the test data", required=True, type=str)
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
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=100)
    parser.add_argument("--kernel_heights", help="filter windows", type=int, nargs='+', default=[2, 4, 5])
    parser.add_argument("--stride", help="stride", type=int, default=1)
    parser.add_argument("--padding", help="padding", type=int, default=0)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--balance_class_weight", action="store_true", default=False, help="Whether or not to use balanced class weights")


    args = parser.parse_args()

    seed = args.seed
    lr = args.learning_rate
    weight_decay = args.weight_decay
    dropout_prob = args.dropout_prob
    in_channels = args.in_channels
    out_channels = args.out_channels
    kernel_heights = args.kernel_heights
    stride = args.stride
    padding = args.padding
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    class_weights = 'balanced' if args.balance_class_weight else None
    model_name = args.model_name
    use_scheduler = False

    config = {
            "epochs": n_epochs,
            "learning_rate": lr,
            "warmup": 0,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "bert_model": model_name,
            "seed": seed,
            "use_scheduler": use_scheduler,
            "balance_class_weight": args.balance_class_weight,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_heights": kernel_heights,
            "stride": stride,
            "padding": padding
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

    model, tokenizer = get_cnn(in_channels, out_channels, kernel_heights, stride, padding, dropout_prob)
    tokenizer_fn = text_to_batch_transformer_bpemb


    # Create the datasets and trainer
    valid_dset = TransformerSingleSentenceDataset(valid_data_loc, tokenizer, tokenizer_fn=tokenizer_fn)
    trainer = TransformerClassificationTrainer(model, device, num_labels=2, tokenizer=tokenizer)
    train_dset = TransformerSingleSentenceDataset(train_data_loc, tokenizer, tokenizer_fn=tokenizer_fn)

    # wandb initialization
    run = wandb.init(
        project="scientific-citation-detection",
        name=args.run_name,
        config=config,
        reinit=True,
        tags=args.tag
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
        model_file=f"{model_dir}/model.pth",
        class_weights=class_weights,
        metric_name='F1',
        logger=wandb,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size,
        use_scheduler=use_scheduler
    )

    # Test on the final test data
    test_dset = TransformerSingleSentenceDataset(test_data_loc, tokenizer, tokenizer_fn=tokenizer_fn)
    (test_loss, acc, P, R, F1) = trainer.evaluate(test_dset)
    wandb.run.summary[f'test-loss'] = test_loss
    wandb.run.summary[f'test-acc'] = acc
    wandb.run.summary[f'test-P'] = P
    wandb.run.summary[f'test-R'] = R
    wandb.run.summary[f'test-F1'] = F1