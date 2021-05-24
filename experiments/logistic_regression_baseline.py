import sys
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import ipdb
import argparse
import wandb
from sklearn.metrics import precision_recall_fscore_support


from datareader import read_citation_detection_jsonl_single_line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", help="Location of the training data", required=True, type=str)
    parser.add_argument("--validation_data", help="Location of the validation data", required=True, type=str)
    parser.add_argument("--test_data", help="Location of the test data", required=True, type=str)
    parser.add_argument("--run_name", help="A name for this run", required=True, type=str)
    parser.add_argument("--tag", help="A tag to give this run (for wandb)", required=True, type=str)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--C", help="The value of C", type=float, default=0.1)


    args = parser.parse_args()

    seed = args.seed
    C = args.C

    random.seed(seed)
    np.random.seed(seed)

    # wandb initialization
    run = wandb.init(
        project="scientific-citation-detection",
        name=args.run_name,
        config={
            'C': C,
            'warm_start': True,
            'bert_model': 'logistic_regression',
            'balance_class_weights': True
        },
        reinit=True,
        tags=args.tag
    )

    train_data_loc = args.train_data
    dev_data_loc = args.validation_data
    test_data_loc = args.test_data

    # The .sample part shuffles the dataframe
    train_data = read_citation_detection_jsonl_single_line(train_data_loc).sample(frac=1).reindex()

    y_train = train_data.values[:,1].astype(np.int32)

    test_data = read_citation_detection_jsonl_single_line(test_data_loc)
    y_test = test_data.values[:, 1].astype(np.int32)

    cv = TfidfVectorizer()
    cv.fit(train_data.values[:, 0])

    X_train = cv.transform(train_data.values[:, 0])
    # Do some grid search to get good parameters
    # For hyperparameter search

    classifier = LogisticRegression(penalty='l2', C=C, warm_start=True, class_weight='balanced')

    classifier.fit(X_train, y_train)

    X_test = cv.transform(test_data.values[:, 0])

    preds = classifier.predict(X_test)
    P,R,F1,_ = precision_recall_fscore_support(y_test, preds, average='binary')
    wandb.run.summary[f'test-P'] = P
    wandb.run.summary[f'test-R'] = R
    wandb.run.summary[f'test-F1'] = F1