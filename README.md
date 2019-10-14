# Wikimedia-outreachy-2019
Repository for Wikimedia project of Outreachy 2019

## Second task

The *classify_statements_within_article.py* script takes as input a text file containing titles of articles to be retrieved, and gives as ouput a "citation need" score of each sentence in the articles.

Run the script using the following command:
```
python classify_statements_within_article.py -i input_file.txt  -m models/model.h5 -v embeddings/word_dict.pck -s embeddings/section_dict.pck -o output_folder
```

Where:
- **'-i', '--input'**, is the input .txt file from which we read the titles, one title per line. Lines starting with `#` will be ignored.

- **'-o', '--out_dir'**, is the output directory where we save the results.

- **'-m', '--model'**, is the path to the model which we use for classifying the statements.

- **'-v', '--vocab'**, is the path to the vocabulary of words we use to represent the statements.

- **'-s', '--sections'**, is the path to the vocabulary of section with which we trained our model.

- **'-l', '--lang'**, is the language that we are parsing now, e.g. "en", or "it". Default is 'en'.

### System Requirement

- Python 3.5 or 3.6
- Dependencies in `requirements.txt`




