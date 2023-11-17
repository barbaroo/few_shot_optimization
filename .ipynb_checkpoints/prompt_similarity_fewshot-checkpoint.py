from openai import OpenAI
import openai
import os
import pandas as pd
import json

def create_prompt(example, src_lang, trg_lang):
    return f'Translate this sentence from {src_lang} into {trg_lang}: {example}'

def translate(prompt, src_lang, trg_lang, translation_examples, client):
        # Constructing the examples part of the prompt
    examples_str = "\n\n".join(
        [f"'{src_lang}': '{example[src_lang]}', '{trg_lang}': '{example[trg_lang]}'" for example in translation_examples]
    )

    # Full system prompt with examples
    system_prompt = f"""You are an expert in Faroese language. 
                        Here are some examples of translations from {src_lang} to {trg_lang}:

                        {examples_str}

                        When I give you a sentence in {src_lang}, you translate it into {trg_lang}.  
                        Respond with a JSON object with two keys: '{src_lang}' for the original sentence and '{trg_lang}' for the translation.
                        The translations should be of excellent quality."""
    
    print(system_prompt)

    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def load_data(path, src_lang):
    file_name = f'{src_lang}_Latn.devtest'
    file_path = os.path.join(path, file_name)
    df = pd.read_csv(file_path, sep='delimiter', header=None, engine='python')
    return list(df[0])[:2]


def export_json_list_to_file(json_list, file_path):
    # Serialize the list of JSON objects to a JSON formatted string
    json_string = json.dumps(json_list, indent=4)
    # Write the JSON formatted string to the specified file
    with open(file_path, 'w') as json_file:
        json_file.write(json_string)


def main():
    # Configurations
    api_key = os.environ.get("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")
    data_path = 'data'
    source_lang = 'Faroese'
    target_lang = 'English'
    result_path = 'results'
    example_file = f'{data_path}/Sprotin_sentences.txt'
    file_name_indexes = "indexes/output_4.json"
    output_file_name = f'{source_lang}_to_{target_lang}.json'
    output_file_path = os.path.join(result_path, output_file_name)

    # Initialize OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Load data
    sentences = load_data(data_path, source_lang)
    df_examples = pd.read_csv(example_file, header = None )
    list_examples_fo = list(df_examples[1])
    list_examples_eng = list(df_examples[0])
    print(list_examples_fo[0])
    print(list_examples_eng[0])
    # File name from which to load the JSON data


# Loading the dictionary from the JSON file
    with open(file_name_indexes, 'r') as json_file:
        dict_indexes = json.load(json_file)



    # Translate sentences
    combined_translations = []
    for iter, sentence in enumerate(sentences):
        prompt = create_prompt(sentence, source_lang, target_lang)

        index_examples = dict_indexes[f'{iter}']

        examples_fo = []
        examples_eng = []



        example = []
        for ind in index_examples:
            fo = list_examples_fo[ind]
            eng = list_examples_eng[ind]
            example.append({'Faroese': fo, 'English': eng})

        print(example)

        translation = translate(prompt, source_lang, target_lang, example, client)
        translation_json = json.loads(translation)
        combined_translations.append(translation_json)

    # Export to JSON
    export_json_list_to_file(combined_translations, output_file_path)

if __name__ == "__main__":
    main()