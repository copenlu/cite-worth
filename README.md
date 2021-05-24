# CiteWorth
Data and code for the paper "CiteWorth: Cite-Worthiness Detection for Improved Scientific Document Understanding"

Dustin Wright and Isabelle Augenstein

In Findings of ACL 2021

Scientific document understanding is challenging as the data is highly domain specific and diverse. However, datasets for tasks with scientific text require expensive manual annotation and tend to be small and limited to only one or a few fields. At the same time, scientific documents contain many potential training signals, such as citations, which can be used to build large labelled datasets. Given this, we present an in-depth study of cite-worthiness detection in English, where a sentence is labelled for whether or not it cites an external source. To accomplish this, we introduce CiteWorth, a large, contextualized, rigorously cleaned labelled dataset for cite-worthiness detection built from a massive corpus of extracted plain-text scientific documents. We show that CiteWorth is high-quality, challenging, and suitable for studying problems such as domain adaptation. Our best performing cite-worthiness detection model is a paragraph-level contextualized sentence labelling model based on Longformer, exhibiting a 5 F1 point improvement over SciBERT which considers only individual sentences. Finally, we demonstrate that language model fine-tuning with cite-worthiness as a secondary task leads to improved performance on downstream scientific document understanding tasks.

## Getting the data
The data is available for download via this link: [https://drive.google.com/drive/folders/1j4B1rQFjjqnRzKsf15ur2\_rCaBh5TJKD?usp=sharing](https://drive.google.com/drive/folders/1j4B1rQFjjqnRzKsf15ur2\_rCaBh5TJKD?usp=sharing)

The data is derived from the [S2ORC dataset](https://github.com/allenai/s2orc), specifically the 20200705v1 release of the data. It is licensed under the [CC By-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) license.   

## Pretrained models
We releases two pretrained models from the paper, available in the [HuggingFace model hub](https://huggingface.co/copenlu). The two models are:
 - `copenlu/citebert`: SciBERT trained on citation detection and masked language modeling from CiteWorth data
 - `copenlu/citebert-cite-only`: SciBERT trained on citation detection only from CiteWorth data

## Environment setup
We recommend using Anaconda to create your environment. After installing conda, run the following to create the environment.
```[bash]
$ conda env create -f environment.yml python=3.8
$ conda activate citeworth
$ python -m spacy download en_core_web_sm
$ pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz   
``` 

## Building a dataset from scratch
We have released the code to build a dataset using the same filtering and cleaning as CiteWorth in the `dataset_creation` directory. To do so, do the following:

First, download the [S2ORC dataset](https://github.com/allenai/s2orc) and place it in `dataset_creation/data/s2orc_full`. Then, run the following commands:
```[bash]
$ cd dataset_creation
$ python filter_papers_full.py
$ python get_filtered_pdf_line_offsets.py
$ python build_dataset.py
```

Filtering the data may take a while depending on your computing infrastructure. You should then have a file `dataset_creation/data/citation_needed_data_contextualized_with_removal_v1.jsonl`. 


## Running experiments
The code for our experiments can be found under `experiments`. To run them, first create a directory `experiments/data` and place all of the CiteWorth data in this directory. Then, go to `experiments/experiment_scripts` and run any of the experiments given there.


## Citing
Please use the following citation when referencing this work or using the data:

```
@inproceedings{wright2021citeworth,
    title={{CiteWorth: Cite-Worthiness Detection for Improved Scientific Document Understanding}},
    author={Dustin Wright and Isabelle Augenstein},
    booktitle = {Findings of ACL-IJCNLP},
    publisher = {Association for Computational Linguistics},
    year = 2021
}
```
