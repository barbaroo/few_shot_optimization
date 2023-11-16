import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Constants
NUMBER_OF_EXAMPLES = 4
MODEL_NAME = 'intfloat/multilingual-e5-large'
FILE_PATH = 'data'
EXAMPLES_FILE = f'{FILE_PATH}/Sprotin_sentences.txt'
TRANSLATION_FILE = f'{FILE_PATH}/Faroese_Latn.devtest'
OUTPUT_FILE = f"indexes/output_{NUMBER_OF_EXAMPLES}.json"




def calculate_embeddings(sentences):
    """Calculate sentence embeddings."""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(sentences)

def find_most_similar(emb_translations, emb_examples, num_examples):
    """Find most similar sentences."""
    scores = np.zeros((len(emb_translations), len(emb_examples)))

    for i, translation in enumerate(emb_translations):
        for j, example in enumerate(emb_examples):
            scores[i, j] = cos_sim(translation, example)

    most_similar = {}
    for index, row_scores in enumerate(scores):
        sorted_indexes = np.argsort(row_scores)[::-1][:num_examples]
        most_similar[index] = sorted_indexes.tolist()

    return most_similar

def main():
    df_examples = pd.read_csv(EXAMPLES_FILE, header = None )
    list_examples = list(df_examples[1])[0:100]
    df_translation_data = pd.read_csv(TRANSLATION_FILE, sep='delimiter', header=None, engine='python')
    translation_sentences = list(df_translation_data[0])[0:2]

    emb_examples = calculate_embeddings(list_examples)
    emb_translation = calculate_embeddings(translation_sentences)

    dict_most_similar = find_most_similar(emb_translation, emb_examples, NUMBER_OF_EXAMPLES)

    try:
        with open(OUTPUT_FILE, 'w') as json_file:
            json.dump(dict_most_similar, json_file, indent=4)
        print(f"JSON data has been saved to {OUTPUT_FILE}")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

if __name__ == "__main__":
    main()
