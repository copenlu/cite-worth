import sys
import matplotlib as mpl
import random
import os
import argparse
import json
import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import ipdb
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.mixture import GaussianMixture
import scipy
from scipy.spatial.distance import euclidean
from collections import defaultdict

import seaborn as sns
sns.set(font_scale=1.4)
import matplotlib.pyplot as plt

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModel


colors = ['red', 'orange', 'blue', 'purple', 'green', 'pink', 'brown', 'gray', 'black']


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


def make_ellipses(gmm, ax, clusters_to_classes):
    """
    Adds Ellipses to ax according to the gmm clusters.
    Taken from https://github.com/roeeaharoni/unsupervised-domain-clusters/blob/master/src/domain_clusters.ipynb
    """

    for n in sorted(list(clusters_to_classes.keys())):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        class_id = clusters_to_classes[n]
        class_color = colors[n]
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=class_color, linewidth=0)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.4)
        ax.add_artist(ell)
        ax.set_aspect('equal', 'datalim')


def map_clusters_to_classes_by_majority(y_train, y_train_pred):
    """
    Maps clusters to classes by majority to compute the Purity metric.
    Taken from https://github.com/roeeaharoni/unsupervised-domain-clusters/blob/master/src/domain_clusters.ipynb
    """
    cluster_to_class = {}
    for cluster in np.unique(y_train_pred):
        # run on indices where this is the cluster
        original_classes = []
        for i, pred in enumerate(y_train_pred):
            if pred == cluster:
                original_classes.append(y_train[i])
        # take majority
        cluster_to_class[cluster] = max(set(original_classes), key = original_classes.count)
    return cluster_to_class



def read_data(jsonl_file, domains):
    """
    Reads in the jsonl file for citation detection and returns a dataframe for a single-sentence dataset
    :param jsonl_file: Location of the dataset
    :return:
    """
    with open(jsonl_file) as f:
        data = [json.loads(l.strip()) for l in f]

    # Get sentences and labels
    dataset = [[s['text'], d['mag_field_of_study'][0]] for d in data for s in d['samples'] if len(domains) == 0 or d['mag_field_of_study'][0] in domains]

    return dataset


def collate_batch(pad_token_id: int, input_data):
    input_ids = [i[0] for i in input_data]
    masks = [i[1] for i in input_data]
    labels = [i[2] for i in input_data]

    max_length = max([len(i) for i in input_ids])

    input_ids = [(i + [pad_token_id] * (max_length - len(i))) for i in input_ids]
    masks = [(m + [0] * (max_length - len(m))) for m in masks]

    assert (all(len(i) == max_length for i in input_ids))
    assert (all(len(m) == max_length for m in masks))
    return torch.tensor(input_ids), torch.tensor(masks), labels


def text_to_batch_transformer(text, tokenizer):
    """Turn a piece of text into a batch for transformer model

    :param text: The text to tokenize and encode
    :param tokenizer: The tokenizer to use
    :return: A list of IDs and a mask
    """
    max_length = min(512, tokenizer.model_max_length)
    input_ids = [tokenizer.encode(t, add_special_tokens=True, max_length=max_length, truncation=True, verbose=False) for t in text]

    masks = [[1] * len(i) for i in input_ids]

    return input_ids, masks


class TransformerDataset(Dataset):

    def __init__(self, data, tokenizer):

        self.dataset = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        # Calls the text_to_batch function
        input_ids, masks = text_to_batch_transformer([row[0]], self.tokenizer)
        label = row[1]
        return input_ids[0], masks[0], label



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Location of the training data", required=True, type=str)
    parser.add_argument("--validation_data", help="Location of the validation data", required=True, type=str)
    parser.add_argument("--test_data", help="Location of the test data", required=True, type=str)
    parser.add_argument("--model_name", help="The name of the model being tested. Can be a directory for a local model",
                        required=True, type=str)
    parser.add_argument("--domains", nargs='+', help='A list of domains to use for training', default=[])
    parser.add_argument("--output_dir", help="Top level directory to save the models", required=True, type=str)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--batch_size", help="The batch size", type=int, default=8)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--new_vectors", action="store_true", help="Whether or not to get new vectors")

    args = parser.parse_args()
    enforce_reproducibility(args.seed)

    # See if CUDA available
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # Get the model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # Combine all the data
    data = read_data(args.train_data, args.domains) + read_data(args.validation_data, args.domains) + read_data(args.test_data, args.domains)

    # Run all the data through the model
    dset = TransformerDataset(data, tokenizer)
    collate_fn = partial(collate_batch, tokenizer.pad_token_id)
    dloader = DataLoader(dset, collate_fn=collate_fn, batch_size=args.batch_size)

    # Create numpy memory map
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(len(dset))

    if args.new_vectors:
        memmap = np.memmap(f'{args.output_dir}/vectors.dat', dtype='float32', mode='w+', shape=(len(dset), config.hidden_size))
        i = 0
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dloader):
                input_ids = batch[0].to(device)
                masks = batch[1].to(device)
                fields = batch[2]

                outputs = model(input_ids)
                # avg pooled
                hidden_states = outputs[0] * masks.unsqueeze(-1)
                seq_lens = masks.sum(-1, keepdim=True)
                avg_pooled = hidden_states.sum(1) / seq_lens

                # max pooled
                # hidden_states = outputs[0]
                # hidden_states[masks == 0] = float('-inf')
                # max_pooled = hidden_states.max(dim=1)[0]

                memmap[i:i+args.batch_size, :] = avg_pooled.detach().cpu().numpy()
                i += args.batch_size
    else:
        memmap = np.memmap(f'{args.output_dir}/vectors.dat', dtype='float32', mode='r', shape=(len(dset), config.hidden_size))

    on_mem_data = np.array(memmap)
    pca_plotting = PCA(n_components=50)
    plot_data = pca_plotting.fit_transform(on_mem_data)
    cats = [d[1] for d in data]


    # GMM
    estimator = GaussianMixture(n_components=5, covariance_type='full', max_iter=150, random_state=args.seed, verbose=True)
    estimator.fit(plot_data)
    y_pred = estimator.predict(plot_data)
    clusters_to_classes = map_clusters_to_classes_by_majority(cats, y_pred)
    classes_to_cluster = {v:k for k,v in clusters_to_classes.items()}

    # Get representative points and measure pairwise distances
    centers = np.empty(shape=(estimator.n_components, plot_data.shape[1]))
    for i in range(estimator.n_components):
        density = scipy.stats.multivariate_normal(cov=estimator.covariances_[i], mean=estimator.means_[i]).logpdf(
            plot_data)
        centers[i, :] = plot_data[np.argmax(density)]

    # Calculate purity
    count = 0
    for i, pred in enumerate(y_pred):
        if clusters_to_classes[pred] == cats[i]:
            count += 1
    train_accuracy = float(count) / len(y_pred) * 100
    print(f"Purity: {train_accuracy}")

    # Plot
    fig = plt.figure(figsize=(10, 8))
    pca_plotting = PCA(n_components=2)
    plot_data = pca_plotting.fit_transform(on_mem_data)
    palette = {k: colors[v] for k,v in classes_to_cluster.items()}
    scatter = sns.scatterplot(plot_data[:,0], plot_data[:,1], hue=cats, palette=palette, markers=',', s=1.0, linewidth=0, edgecolor='none')
    scatter.set(xticklabels=[])
    scatter.set(yticklabels=[])
    make_ellipses(estimator, scatter, clusters_to_classes)
    fig.tight_layout()
    fig.savefig(f'{args.output_dir}/plot.png')

    da_scores = {
        'Chemistry': {'Chemistry': 67.58, 'Engineering': 66.62, 'Computer Science': 65.05, 'Psychology': 65.49, 'Biology': 66.59},
        'Engineering': {'Chemistry': 58.41, 'Engineering': 60.25, 'Computer Science': 59.36, 'Psychology': 58.03, 'Biology': 58.80},
        'Computer Science': {'Chemistry': 58.86, 'Engineering': 60.11, 'Computer Science': 61.99, 'Psychology': 56.69, 'Biology': 58.22},
        'Psychology': {'Chemistry': 62.35, 'Engineering': 64.02, 'Computer Science': 63.85, 'Psychology': 65.10, 'Biology': 64.54},
        'Biology': {'Chemistry': 68.23, 'Engineering': 68.07, 'Computer Science': 66.72, 'Psychology': 68.27, 'Biology': 69.12}
    }

    # Get cluster distances
    dists = defaultdict(dict)
    for i in range(centers.shape[0]):
        for j in range(centers.shape[0]):
            dists[clusters_to_classes[i]][clusters_to_classes[j]] = euclidean(centers[i], centers[j])

    # Flatten for each domain
    pearsons = []
    rhos = []
    for test_domain in da_scores:
        print(test_domain)
        scores = []
        dists_flat = []
        for train_domain in da_scores:
            # for j in da_scores[i]:
            #     scores.append(da_scores[i][j])
            #     dists_flat.append(dists[i][j])
            scores.append(da_scores[test_domain][train_domain])
            dists_flat.append(dists[test_domain][train_domain])
        pearsons.append(scipy.stats.pearsonr(scores, dists_flat))
        rhos.append(scipy.stats.spearmanr(scores, dists_flat))
        print(f"Pearsons's correlation: {pearsons[-1]}")
        print(f"Spearman rho: {rhos[-1]}")
        print()

    print(f"Avg pearsons: {np.mean([p[0] for p in pearsons])}")
    print(f"Avg rho: {np.mean([r[0] for r in rhos])}")
