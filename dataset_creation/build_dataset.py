from tqdm import tqdm
import json
import gzip
import time
from multiprocessing import Pool
import scispacy
import spacy
nlp = spacy.load('en_core_sci_sm')
import re
import string
from collections import defaultdict
from pathlib import Path

data_loc = Path('../data/s2orc_full/20200705v1/full/')

def valid_sentence(sentence_text):
    """
    Determines if a sentence starts with a capital letter and ends with a punctuation mark
    :param sentence_text:
    :return: {bool}
    """

    return re.search("[A-Z]", sentence_text[0]) is not None and sentence_text[-1] in '.?!'

def get_true_token_index(chars_to_tokens, index, val=0):
    """
    Gets the true token index depending on if the character is whitespace or not and if it should point
    to the previous or next token
    :param chars_to_tokens:
    :param index:
    :param val: 0 for previous, 1 for next
    :return:
    """
    if type(chars_to_tokens[index]) is tuple:
        return chars_to_tokens[index][val]
    else:
        return chars_to_tokens[index]


def get_permissible_title_list():
    with open('./data/permissible_section_titles.txt') as f:
        titles = [l.strip() for l in f]
    return titles


def extract_sections(pdf, permissible_titles):
    sections = []
    indices = []
    total_cite_spans = 0
    cite_spans_valid = False
    for j,pgr in enumerate(pdf['body_text']):
        if pgr['section'].lower() in permissible_titles and len(pgr['cite_spans']) > 0:
            sections.append(pgr)
            indices.append(j)
            if not cite_spans_valid:
                if valid_citation_format(pgr['cite_spans'][0]['text']):
                    cite_spans_valid = True
                # Skip this paper if the citation format is invalid
                else:
                    return [], []
            total_cite_spans += len(pgr['cite_spans'])

    return (indices, sections) if total_cite_spans > 5 else ([], [])


def create_doc_and_index(text):
    """
    Get spacy doc of text as well as a map from char indices to token indices
    :param text: The original text
    :return: (doc, dict): The spacy parsed document and a map from characters to token indices
    """
    doc = nlp(text, disable=['tagger', 'ner'])
    chars_to_tokens = {}
    largest = 0
    last_idx = 0
    for token in doc:
        while token.idx > largest:
            chars_to_tokens[largest] = (last_idx, last_idx + 1)
            largest += 1
        for i in range(token.idx, token.idx + len(token.text)):
            chars_to_tokens[i] = token.i
            last_idx = token.i
            largest = i + 1
    while largest < len(text):
        chars_to_tokens[largest] = last_idx
        largest += 1
    return doc, chars_to_tokens


def combine_span_indices(text, chars_to_tokens, cite_spans):
    final_spans = [{'start': cite_spans[0]['start'], 'end': cite_spans[0]['end'], 'text': cite_spans[0]['text'],
                    'ref_ids': [cite_spans[0]['ref_id']]}]
    for cs in cite_spans[1:]:
        # If it is the next or current token, combine them
        if get_true_token_index(chars_to_tokens, cs['start'], 0) - \
                get_true_token_index(chars_to_tokens, final_spans[-1]['end'], 0) < 2:
            final_spans[-1]['end'] = cs['end']
            final_spans[-1]['text'] = text[final_spans[-1]['start']:cs['end']]
            final_spans[-1]['ref_ids'].append(cs['ref_id'])
        else:
            final_spans.append({'start': cs['start'], 'end': cs['end'], 'text': cs['text'],
                            'ref_ids': [cs['ref_id']]})

    return final_spans


def not_end_of_sentence(text, sent, span):
    # If anything after the last token of the citation is not punctuation or whitespace, its in the middle of the
    # sentence
    return re.search(f"[^{string.punctuation}]\s+", text[span['end']:sent.end_char])


def brackets_only(span_text):
    # If anything after the last token of the citation is not punctuation or whitespace, its in the middle of the
    # sentence
    return (span_text[0] == '[' and span_text[-1] == ']') or (span_text[0] == '(' and span_text[-1] == ')')


def hanging_citation(text):
    """
    Checks if a sentence ends with a certain list of prepositions and ignores those that do
    :param text:
    :return:
    """
    return re.search("\s+\(?(\(\s*\)|like|reference|including|include|with|for instance|for example|see also|at|following|of|from|to|in|by|see|as|e\.?g\.?(,)?|viz(\.)?(,)?)\s*(,)*(-)*[\)\]]?\s*[.?!]\s*$", text.lower()) is not None


def valid_citation_format(text):
    """
    Returns true if the citation text is in a format that we accept
    :param text:
    :return:
    """
    return re.search("\[([0-9]+\s*[,-;]*\s*)*[0-9]+\s*\]", text) is not None \
           or re.search("\(?[12][0-9]{3}[a-z]?\s*\)", text) is not None

def ignore_sentence(text, sent, span=None, sentence_text=None):
    """
    :param text:
    :param span: A sample span containing the format of interest, or None to check for any citation type
    :return:
    """
    if sentence_text is None:
        sentence_text = text[sent.start_char:span['start']]

    if span is None:
        return re.search("\[([0-9]+\s*[,-;]*\s*)*[0-9]+\s*\]", sentence_text) is not None \
               or re.search("\(?[12][0-9]{3}[a-z]?\s*\)|et( )?al(\.?)", sentence_text) is not None
    elif re.search("\[([0-9]+\s*[,-;]*\s*)*[0-9]+\s*\]", span['text']):
        return re.search("\[([0-9]+\s*[,-;]*\s*)*[0-9]+\s*\]", sentence_text) is not None \
               or re.search("\(?[12][0-9]{3}[a-z]?\s*\)|et( )?al(\.?)", sentence_text) is not None
    elif re.search("\(?[12][0-9]{3}[a-z]?\s*\)", span['text']):
        # Ignore a cite span if no author name is in it or the first character isn't a parenthesis
        return re.search("\(?[12][0-9]{3}[a-z]?\s*\)|et( )?al(\.?)", sentence_text) is not None \
               or re.search("\[([0-9]+\s*[,-;]*\s*)*[0-9]+\s*\]", sentence_text) is not None \
               or not re.search("[A-Z]", span['text']) or span['text'][0] != '('
    # We are going to ignore sentences which have a different citation format
    else:
        return True


def next_char_punctuation(text):
    """
    Determines if the next non-whitespace character is punctuation
    :param text:
    :return:
    """
    return re.search(f"^\s*[{string.punctuation}]", text) is not None


def remove_citations(text, sent, spans):
    """
    Removes all of the citation text for a given sentence
    :param text:
    :param span:
    :return:
    """
    # Make sure the spans are sorted
    sorted_spans = list(sorted(spans, key=lambda x: x['start']))
    final_text = text[sent.start_char:sorted_spans[0]['start']]
    for i in range(1,len(sorted_spans)):
        t = text[sorted_spans[i-1]['end']:sorted_spans[i]['start']]
        if next_char_punctuation(t):
            final_text = final_text.rstrip()
        final_text += t.lstrip()
    t = text[sorted_spans[-1]['end']:sent.end_char]
    if next_char_punctuation(t):
        final_text = final_text.rstrip()
    final_text += t.lstrip()
    return final_text


def dataset_worker(ab):
    dataset = []
    permissible_titles = get_permissible_title_list()
    with gzip.open(f"{data_loc}/filtered_metadata/metadata_{ab}.jsonl.gz") as f, \
         gzip.open(f"{data_loc}/pdf_parses/pdf_parses_{ab}.jsonl.gz") as g:
        for i, l in enumerate(f):
            metadata = json.loads(l.strip())
            #Seek to line in pdfs file
            g.seek(metadata['file_line_offset'])
            pdf = json.loads(g.readline())
            # Text indices are the indices into the "body_text" section of the parsed PDF json
            (text_indices, sections) = extract_sections(pdf, permissible_titles)
            for index, sec in zip(text_indices, sections):

                # Create the doc and index all of the characters to tokens
                doc, c_to_t  = create_doc_and_index(sec['text'])

                # Skip this paragraph if the number of sentences is less than 3
                if len(list(doc.sents)) < 3:
                    continue

                # Combine citation spans
                final_spans = combine_span_indices(sec['text'], c_to_t, sec['cite_spans'])
                j = 0

                # We're going to store all of the sentences in succession
                final_samples = []
                paragraph_failed = False
                for sent in doc.sents:
                    context_only = False
                    sentence_citation_spans = []
                    # If there are still citation spans and the current sentence starts after the current span,
                    # there was a sentence parsing issue where the character spans
                    # don't align with sentence boundaries and a citation was parsed incorrectly, so ignore this paragraph
                    if j < len(final_spans) and sent.start_char > final_spans[j]['start']:
                        paragraph_failed = True
                        break
                    # While we still have citation spans and this sentence contains the current span
                    while j < len(final_spans) and sent.start_char <= final_spans[j]['start'] \
                            and final_spans[j]['end'] <= sent.end_char:
                        sentence_citation_spans.append(final_spans[j])
                        # Seeing what the dataset looks like if we take any sentence where all the citations are bracketed
                        context_only = context_only \
                                       or not brackets_only(final_spans[j]['text'].strip()) \
                                       or not_end_of_sentence(sec['text'], sent, final_spans[j])

                        j += 1
                    if len(sentence_citation_spans) > 0:
                        # Make sure all the spans are in this sentence
                        assert all(sent.start_char <= sp['start']
                                   and sp['end'] <= sent.end_char
                                   and sp['text'] in sent.text for sp in sentence_citation_spans), \
                            "Not good, this isn't the correct span"

                        # Remove all of the citation text
                        final_sentence = remove_citations(sec['text'], sent, sentence_citation_spans)

                        # Remove punctuation
                        while len(final_sentence) > 20 and final_sentence[-2] in ' @&:;/-,\n\t':
                            final_sentence = final_sentence[:-2] + final_sentence[-1]
                        # Fail out immediately if there were missing citations of the sentence appears to have been
                        # parsed incorrectly
                        if len(final_sentence) > 20 \
                                and valid_sentence(final_sentence) \
                                and not ignore_sentence(sec['text'], sent, sentence_text=final_sentence) \
                                and not hanging_citation(final_sentence):

                            # Create json
                            final_samples.append({
                                'text': final_sentence,
                                'label': 'context-only' if context_only else 'check-worthy',
                                'original_text': sent.text,
                                'ref_ids': [sp['ref_ids'] for sp in sentence_citation_spans],
                                'citation_text': [sp['text'] for sp in sentence_citation_spans]
                            })
                        else:
                            #print(f"Invalid sentence: {final_sentence}\n")
                            paragraph_failed = True
                            break
                    else:
                        # Check here if we should ignore the sentence; basically we're ignoring all docs which
                        # have a strange sentence format
                        if not ignore_sentence(sec['text'], sent, sentence_text=sent.text) \
                                and valid_sentence(sent.text) and len(sent.text) > 20 \
                                and not hanging_citation(sent.text):
                            final_samples.append({
                                'text': sent.text,
                                'label': 'non-check-worthy',
                                'original_text': sent.text,
                                'ref_ids': None
                            })
                        else:
                            paragraph_failed = True
                            break

                if paragraph_failed:
                    final_samples = []

                if len(final_samples) > 0:
                    dataset.append({
                        'paper_id': metadata['paper_id'],
                        'section_index': index,
                        'file_index': f"{ab}",
                        'file_offset': metadata['file_line_offset'],
                        'mag_field_of_study': metadata['mag_field_of_study'],
                        'original_text': sec['text'],
                        'section_title': sec['section'],
                        'samples': final_samples
                    })
    return ab,dataset


if __name__ == "__main__":
    # Get all of the statistics for venues, also time how long it takes to iterate through all the data
    start = time.time()
    pool = Pool(8)

    # with open(f"{data_loc}/citation_needed_data_v4.tsv", 'wt') as f:
    #     f.write("text\toriginal_citation\tlabel\n")
    version = 1
    completed = []
    with open(f'data/citation_needed_data_contextualized_with_removal_v{version}_completed.txt') as f:
        completed = set([int(l.strip()) for l in f])

    run_list = [i for i in range(100) if i not in completed and i != 12] # skip 12 since it never finishes
    print(f"{len(run_list)} files to go")
    for result in tqdm(pool.imap_unordered(dataset_worker, run_list), total=len(run_list)):
        # Mark it as completed first so we don't accidentally duplicate data
        with open(f'data/citation_needed_data_contextualized_with_removal_v{version}_completed.txt', 'at+') as f:
            f.write(f"{result[0]}\n")

        with open(f"{data_loc}/citation_needed_data_contextualized_with_removal_v{version}.jsonl", 'at+') as f:
            for record in result[1]:
                f.write(f"{json.dumps(record)}\n")

    pool.close()
    pool.join()
