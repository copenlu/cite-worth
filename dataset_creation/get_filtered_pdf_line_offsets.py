from tqdm import tqdm
import json
import gzip
import time
from multiprocessing import Lock
from multiprocessing import Pool
from multiprocessing.managers import SyncManager
from functools import partial
from collections import Counter

from collections import defaultdict
from pathlib import Path


data_loc = Path('./data/s2orc_full/20200705v1/full/')


def index_metadata(ab):
    article_id_to_metadata = {}
    with gzip.open(f"{data_loc}/filtered_metadata/metadata_{ab}.jsonl.gz") as f:
        for i, l in enumerate(f):
            metadata = json.loads(l.strip())
            article_id_to_metadata[metadata['paper_id']] = metadata
    with gzip.open(f"{data_loc}/pdf_parses/pdf_parses_{ab}.jsonl.gz") as f:
        offset = 0
        for i, l in enumerate(f):
            pdf = json.loads(l.strip())
            if pdf['paper_id'] in article_id_to_metadata:
                article_id_to_metadata[pdf['paper_id']]['file_line_offset'] = offset
            offset += len(l)
    return ab,list(article_id_to_metadata.values())

if __name__ == "__main__":
    # Get all of the statistics for venues, also time how long it takes to iterate through all the data
    start = time.time()

    pool = Pool(8)
    venue_frequencies = defaultdict(int)
    for vf in tqdm(pool.imap_unordered(index_metadata, range(100)), total=100):
        with gzip.open(f"{data_loc}/filtered_metadata/metadata_{vf[0]}.jsonl.gz", 'wt') as f:
            for l in vf[1]:
                f.write(json.dumps(l) + '\n')
    pool.close()
    pool.join()
