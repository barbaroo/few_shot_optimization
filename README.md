# Faroese-English Translation Project

This project is aimed at investigating the role of semantic similarity between examples and query in few shot translation performance. 
In order to do so, we perform Faroese-English translations fo the devtest split of the FLORES dataset using the OpenAI API. Examples based on semantic similarity from the Sprotin parallel dataset.

## Components

The project consists of two main Python scripts:

1. **Pick_examples.py**: This script is used to select few-shot learning examples from a dataset based on semantic similarity. It utilizes the `SentenceTransformer` library for calculating sentence embeddings.

2. **prompt_similarity_fewshot.py**: This script handles the actual translation task. It constructs prompts for translation by including semantically similar examples and then uses the OpenAI API to perform the translation.

## Setup and Requirements

- Python 3.x
- Libraries: `openai`, `pandas`, `numpy`, `sentence_transformers`
- OpenAI API key

## Usage

1. **Pick_examples.py**: Run this script first to select and prepare few-shot learning examples. It reads sentences from a specified dataset and calculates their embeddings to find the most semantically similar examples. The index of the examples selected for each query is then output in JSON format (output_<n_examples>.json). Choose the number of most similar example as a hyperparameter.

2. **prompt_similarity_fewshot.py**: After preparing the examples, use this script to perform the translations. It constructs prompts that include the few-shot examples and communicates with the OpenAI API to obtain translations. Translations are exported in JSON format.

## Configuration

- Set the OpenAI API key in your environment variables.
- Adjust the paths and model names as required in the scripts.


