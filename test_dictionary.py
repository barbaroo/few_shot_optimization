import json

# File name from which to load the JSON data
file_name = "indexes/output_4.json"

# Loading the dictionary from the JSON file
with open(file_name, 'r') as json_file:
    loaded_dict = json.load(json_file)

print("Dictionary loaded from JSON file:")
print(type(loaded_dict['0'][0]))