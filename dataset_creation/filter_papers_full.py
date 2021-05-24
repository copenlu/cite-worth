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


def article_allowed(metadata):
    if metadata['mag_field_of_study'] and len(metadata['mag_field_of_study']) > 0 \
        and metadata['has_inbound_citations'] \
        and metadata['has_pdf_parse'] and metadata['has_pdf_parsed_abstract'] \
        and metadata['has_pdf_parsed_body_text'] and metadata['has_pdf_parsed_bib_entries'] \
        and metadata['has_pdf_parsed_ref_entries'] and metadata['abstract'] \
        and (metadata['journal'] or metadata['venue'] or metadata['arxiv_id'] or \
            metadata['acl_id'] or metadata['pubmed_id'] or metadata['pmc_id']):

        return True

    return False


def filter_by_metadata(ab):
    venues = []
    with gzip.open(ab) as f:
        for i, l in enumerate(f):
            metadata = json.loads(l.strip())
            if article_allowed(metadata):
                venues.append((ab,l))
    return venues


if __name__ == "__main__":
    # Get all of the statistics for venues, also time how long it takes to iterate through all the data
    start = time.time()
    (data_loc/"filtered_metadata").mkdir(exist_ok=True)

    article_bundles = []
    for article_bundle in data_loc.glob(f"metadata/*.gz"):
        article_bundles.append(article_bundle)

    pool = Pool(8)
    venue_frequencies = defaultdict(int)
    for vf in tqdm(pool.imap_unordered(filter_by_metadata, article_bundles), total=100):
        with gzip.open(f"{data_loc}/filtered_metadata/{vf[0][0].name}", 'w') as f:
            for l in vf:
                f.write(l[1])
    pool.close()
    pool.join()
