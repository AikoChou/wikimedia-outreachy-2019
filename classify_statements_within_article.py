import re
import argparse
import pickle
import numpy as np
import pandas as pd
import mwparserfromhell
import requests

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))

'''
    Set up the arguments and parse them.
'''

def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to determinee whether a statement needs a citation or not.')
    parser.add_argument('-i', '--input', help='The input file contains the titles of English Wikipedia articles.', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=True)
    parser.add_argument('-m', '--model', help='The path to the model which we use for classifying the statements.', required=True)
    parser.add_argument('-v', '--vocab', help='The path to the vocabulary of words we use to represent the statements.', required=True)
    parser.add_argument('-s', '--sections', help='The path to the vocabulary of section with which we trained our model.', required=True)
    parser.add_argument('-l', '--lang', help='The language that we are parsing now.', required=False, default='en')

    return parser.parse_args()

'''
    Retrieve the article of the given title
'''

def parse(API_URL, title):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "rvlimit": 1,
        "titles": title,
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": "My-Bot-Name/1.0"}
    req = requests.get(API_URL, headers=headers, params=params)
    res = req.json()
    #print(res)
    revision = res["query"]["pages"][0]["revisions"][0]
    text = revision["slots"]["main"]["content"]
    return mwparserfromhell.parse(text)

'''
    Parsing the article

    Return: a content dictionary
            -keys: section name, including MAIN_SECTION and other top level headdings
            -values: a list of paragraphs in the section
'''
def get_content_dict(text):
    brace_open = False
    contents = {'MAIN_SECTION': []}
    section = 'MAIN_SECTION'

    entries = re.split("\n+", text)
    heading_p = re.compile('(?<=^\={2})\s*\w+[\s|,|\w]*')
    for entry in entries:
        ## ignore {{...}} at the beginning e.g. infobox
        if entry[:2] == '{{':
            brace_open = True
            if entry[-2:] == '}}':
                brace_open = False
                continue
        if entry[:2] == '}}':
            brace_open = False
            continue
        if brace_open == True:
            continue

        ## search top level headding == ... ==
        match = heading_p.search(entry)
        if match:
            section = match.group(0).strip()
            #print(section)
            contents[section] = []
            continue

        contents[section].append(entry)
    return contents

'''
    Parsing paragraphs in each section

    Return: a sentence dictionary
            -keys: section name
            -values: a list of tuple (sentence_words_list, citation_or_not)

            **citation_or_not: True if <ref>...</ref> or {{citation_needed|...}} found in the sentence.
'''

def get_sentence_dict(content_dict):
    sentence_dict = {}
    sentences_p = re.compile('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')

    for section in content_dict.keys():
        ## ignore sections which content dont need citations
        if section in ["See also", "References", "External links", "Further reading", "Notes"]:
            continue

        sentence_dict[section] = []
        for parag in content_dict[section]:
            ## ignore list & minor headding
            if parag[0] in  '*=<|{! ':
                continue
            if parag[:3] == "'''" and parag[-3:] == "'''":
                continue

            ## replace <ref>...</ref> or {{citation needed}} part with a <ref> tag
            parag = remove_reflink(parag)
            ## remove {{...}} in the paragraph
            parag = remove_notetag(parag)

            ## split paragraph into sentences
            sentences = sentences_p.split(parag)
            for s in sentences:
                cite = 0
                if "<ref>" in s:
                    cite = 1
                ## handle abbreviation, special characters and transform the text into a word list
                text, wordlist = clean_text(s)
                sentence_dict[section].append((text, wordlist, cite))
    return sentence_dict


def remove_reflink(text):
    text = re.sub("(<ref>.+?</ref>)+", "<ref>", text)
    text = re.sub("(<ref\sname=\"*.+?\"*>.+?</ref>)+", "<ref>", text)
    text = re.sub("<ref\sname=\"*.+\"*/>", "<ref>", text)
    text = re.sub("{{citation needed.+?}}", "<ref>", text)
    text = re.sub("\.\"*\'*\s*(<ref>)+", " <ref>.", text)
    return text

def remove_notetag(text):
    text = re.sub("{{.+?}}", "", text)
    return text

def clean_text(text):
    text = re.sub(r"<ref>", "", text)
    raw_text = text

    text = str(text).lower()
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()
    return raw_text, text
'''


'''

def construct_instance_reasons(statement_dict, section_dict_path, vocab_w2v_path, max_len=-1):
    # load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'))
    # load the section dictionary
    section_dict = pickle.load(open(section_dict_path, 'rb'), encoding='latin1')

    # construct the training data
    X = []
    sections = []
    y = []
    outstring = []
    for section in statement_dict.keys():
        try:
            section_id = section_dict.get(section.strip().lower(), 0)
            for (text, wordlist, label) in statement_dict[section]:
                X_inst = []
                for w in wordlist:
                    if max_len != -1 and len(X_inst) >= max_len:
                        break
                    if w not in vocab_w2v:
                        X_inst.append(vocab_w2v['UNK'])
                    else:
                        X_inst.append(vocab_w2v[w])
                X.append(X_inst)
                sections.append(section_id)
                y.append(label)
                outstring.append(text)
        except Exception as e:
            print(section)
            print(e)

    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)
    y = to_categorical(y)
    return X, np.array(sections), y, encoder, outstring



if __name__ == '__main__':
    p = get_arguments()

    # load the model
    model = load_model(p.model)
    max_seq_length = model.input[0].shape[1].value

    # read the input file
    with open(p.input, 'r') as f:
        titles = [l.strip() for l in f.readlines()]
    print("==========================================")
    print("INPUT: ", titles)
    print()

    API_URL = "https://en.wikipedia.org/w/api.php"
    outstr = "Text\tPredictProb[0]\tPredictProb[1]\tLabel\n"

    # process title one by one
    for title in titles:
        print("Processing article: ", title)

        # retrieve the article
        article = str(parse(API_URL, title))

        # get the content dictionary
        content = get_content_dict(article)

        # get the sentence dict dictionary
        sentence_dict = get_sentence_dict(content)
        print("  Number of section: ", len(sentence_dict))

        # construct input data for model
        X, sections, y, encoder, outstring = construct_instance_reasons(sentence_dict, p.sections, p.vocab, max_seq_length)
        print("  Number of sentence: ", len(X))

        # classify the data
        pred = model.predict([X, sections])
        # sort by predicted score
        pred_df = pd.DataFrame(pred).sort_values(by=[0])

        # save the prediction
        for idx, y_pred in pred_df.iterrows():
            outstr += outstring[idx]+'\t'+str(y_pred[0])+'\t'+str(y_pred[1])+'\t'+str(y[idx])+ '\n'

    # save all result to file
    output_path = p.out_dir + '/' + 'prediction_result.tsv'
    with open(output_path, 'wt') as f:
        f.write(outstr)
    print("Save to file: ", output_path)
