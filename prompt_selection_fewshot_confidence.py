from openai import OpenAI
import openai
import os
import pandas as pd
import json
import random


# Constants
SOURCE_LANG = 'English'
TARGET_LANG = 'Faroese'
DATA_PATH = 'data'
RESULT_PATH = 'results'
NUM_EXAMPLES = 1
EXAMPLE_FILE = f'{DATA_PATH}/output_sprotin_new.csv'
INDEX_FILE = "indexes/output_21_English.json"
EXAMPLE_SELECTION_MODE = 'random'  # Set to 'random' or 'fixed'


def create_prompt(example, src_lang, trg_lang):
    return f'Translate this sentence from {src_lang} into {trg_lang}: {example}'

def translate(client, prompt, src_lang, trg_lang, translation_examples):
    #client, prompt, SOURCE_LANG, TARGET_LANG, examples
        # Constructing the examples part of the prompt
    examples_str = "\n\n".join(
        [f"'{src_lang}': '{example[src_lang]}', '{trg_lang}': '{example[trg_lang]}'" for example in translation_examples]
    )

    #Full system prompt with examples
    system_prompt = f"""You are an expert in Faroese language. 
                    Here are some examples of translations from {src_lang} to {trg_lang}:

                    {examples_str}

                    When I give you a sentence in {src_lang}, you translate it into {trg_lang}.  
                    Respond with a JSON object with exactly three keys: '{src_lang}' for the original sentence, '{trg_lang}' for the translation, and 'confidence' for a continuous confidence score in the translation ranging from 0 to 1. 
                    The translations should be of excellent quality, and the confidence score should reflect how certain you are of the translation's accuracy."""

    

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def load_data(file_path):
    """Load data from a file."""
    try:
        df = pd.read_csv(file_path, sep='delimiter', header=None, engine='python')
        return list(df[0])[248:]
    except Exception as e:
        print(f"Error loading data: {e}")
        return []


def load_json(file_path):
    """Load JSON data from a file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}
    
def select_examples_randomly(example_fo, example_eng):
    """Select a random set of translation examples."""
    indexes = random.sample(range(len(example_fo)), NUM_EXAMPLES)
    return [{'Faroese': example_fo[i], 
             'English': example_eng[i]} for i in indexes]



def export_json_list_to_file(json_list, file_path):
    # Serialize the list of JSON objects to a JSON formatted string
    json_string = json.dumps(json_list, indent=4)
    # Write the JSON formatted string to the specified file
    with open(file_path, 'w') as json_file:
        json_file.write(json_string)

def main():
    try:
        # Configurations
        api_key = os.environ.get("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")

        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)

        # Load data
        sentences = load_data(f'{DATA_PATH}/{SOURCE_LANG}_Latn.devtest')

   
        df_examples = pd.read_csv(EXAMPLE_FILE)
        example_fo = list(df_examples['Faroese'])
        example_eng = list(df_examples['English'])
        #pythonprint(example_fo)

        # Load indexes
        indexes = load_json(INDEX_FILE)

        
        # Translate sentences
        translations = []
        for idx, sentence in enumerate(sentences):
            prompt = create_prompt(sentence, SOURCE_LANG, TARGET_LANG)


            #example_indexes = indexes[str(idx)]
            #examples = [{'Faroese': example_fo[i], 'English': example_eng[i]} for i in example_indexes]
            if EXAMPLE_SELECTION_MODE == 'random':
                examples = select_examples_randomly(example_fo, example_eng)
                print("rnd")
            else:
                example_indexes = indexes[str(idx)][:NUM_EXAMPLES]  # Limit to NUM_EXAMPLES
                examples = [{'Faroese': example_fo[i].replace('"', ''), 
                             'English': example_eng[i].replace('"', '')} for i in example_indexes]



            translation = translate(client, prompt, SOURCE_LANG, TARGET_LANG, examples)
            translation_json = json.loads(translation)
            print(translation_json)
            translations.append(translation_json)

        # Export translations
        export_json_list_to_file(translations, f'{RESULT_PATH}/{SOURCE_LANG}_to_{TARGET_LANG}_{EXAMPLE_SELECTION_MODE}{NUM_EXAMPLES}.json')

    except Exception as e:
        print(f"An error occurred: {e}")
        # Export whatever has been processed so far
        export_json_list_to_file(translations, f'{RESULT_PATH}/{SOURCE_LANG}_to_{TARGET_LANG}_{EXAMPLE_SELECTION_MODE}{NUM_EXAMPLES}_partial.json')

if __name__ == "__main__":
    main()

