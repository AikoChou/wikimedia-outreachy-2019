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


def text_to_word_list(text):

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
    text = re.sub(r"-", " ", text) #
    text = re.sub(r";", " ", text) #
    text = re.sub(r":", " ", text) #
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()
    return text

'''
    Construct input data for model from the wikicode retrieved

    Return: -X: padded words vector, shape: (num_sentences, max_len)
            -sections: array of sections vector, shape: (num_sentences, 1)
            -outstring: list of raw texts
'''
def construct_instance_reasons(wikicode, section_dict_path, vocab_w2v_path, max_len=-1, DISCARD_SECTIONS=[]):
    # load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'))
    # load the section dictionary
    section_dict = pickle.load(open(section_dict_path, 'rb'), encoding='latin1')

    # construct the training data
    X = []
    sections = []
    outstring = []

    # consider lead section and sections within level 2 headding
    for section in wikicode.get_sections(levels=[2], include_lead=True):
        if not section.filter_headings(): # no heading -> is lead section
            section_name = 'MAIN_SECTION'
            headings = []

        elif section.filter_headings()[0].title in DISCARD_SECTIONS:
            # ignore sections which content dont need citations
            continue

        else:
            section_name = section.filter_headings()[0].title
            # store all (sub)heading names
            headings = [h.title for h in section.filter_headings()]

        # split section content into paragraphs
        paragraphs = re.split("\n+", section.strip_code())

        for paragraph in paragraphs:
            # clean hyperlinks which strip_code() did not remove
            paragraph = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", paragraph)
            # split paragraph into sentences
            sentences = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)

            for sentence in sentences:
                if sentence in headings: # discard all (sub)heading name
                    continue
                # handle abbreviation, special characters and transform the text into a word list
                text = text_to_word_list(sentence)

                 # construct the word vector for word list
                X_inst = []
                for word in text:
                    if max_len != -1 and len(X_inst) >= max_len:
                        break
                    X_inst.append(vocab_w2v.get(word, vocab_w2v['UNK']))

                X.append(X_inst)
                sections.append(section_dict.get(section_name.strip().lower(), 0))
                outstring.append(sentence)
    # pad all word vectors to max_len
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    return X, np.array(sections), outstring


if __name__ == '__main__':
    p = get_arguments()

    # load the model
    model = load_model(p.model)
    max_seq_length = model.input[0].shape[1].value

    # read the input file
    with open(p.input, 'r') as f:
        titles = [l.strip() for l in f.readlines() if l[0] != '#']
    print("==========================================")
    print("INPUT: ", titles)
    print()

    API_URL = "https://en.wikipedia.org/w/api.php"
    outstr = "Text\tPredictProb[0]\tPredictProb[1]\n"

    DISCARD_SECTIONS = ["See also", "References", "External links", "Further reading", "Notes"]

    # process title one by one
    for title in titles:
        print("Processing article: ", title)

        # retrieve the wikicode obj of the article
        wikicode = parse(API_URL, title)

        # check if it is a redirect page
        if "#REDIRECT" in str(wikicode):
            redict_to = wikicode.filter_wikilinks()[0].title
            print("* Redirect to: ", redict_to)
            wikicode = parse(API_URL, redict_to)
        # check if it is a disambiguation page
        if "{{disambiguation" in str(wikicode):
            print("* {} is an ambiguous title.".format(title))
            continue

        # construct input data for model
        X, sections, outstring = construct_instance_reasons(wikicode, p.sections, p.vocab, max_seq_length, DISCARD_SECTIONS)
        #print("  Number of section: ", len(np.unique(sections)))
        print("  Number of sentence: ", len(X))

        # classify the data
        pred = model.predict([X, sections])
        # sort by predicted score
        pred_df = pd.DataFrame(pred).sort_values(by=[0])

        # save the prediction
        for idx, y_pred in pred_df.iterrows():
            outstr += outstring[idx]+'\t'+str(y_pred[0])+'\t'+str(y_pred[1])+ '\n'

    # save all result to file
    output_path = p.out_dir + '/' + 'prediction_result_2.tsv'
    with open(output_path, 'wt') as f:
        f.write(outstr)
    print()
    print("Save to file: ", output_path)
