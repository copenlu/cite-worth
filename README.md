# CiteWorth
Code for the paper "CiteWorth: Cite-Worthiness Detection for Improved Scientific Document Understanding"

The code from our experiments is currently being prepared for release, in addition to scripts which can be used to extract all of the data from S2ORC. 

## Getting the data
The data is available for download via this link: [https://drive.google.com/drive/folders/1j4B1rQFjjqnRzKsf15ur2\_rCaBh5TJKD?usp=sharing](https://drive.google.com/drive/folders/1j4B1rQFjjqnRzKsf15ur2\_rCaBh5TJKD?usp=sharing)

The data is derived from the [S2ORC dataset](https://github.com/allenai/s2orc), specifically the 20200705v1 release of the data. It is licensed under the [CC By-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/) license.   

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
