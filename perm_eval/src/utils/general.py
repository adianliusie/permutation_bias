import os

import json
import jsonlines
import pickle

from pathlib import Path

#== Saving and Loading Utils =============================================================#
def save_pickle(data, path:str):
    with open(path, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(path:str):
    with open(path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def save_json(data:dict, path:str):
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=2)

def load_json(path:str)->dict:
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    return data

def load_jsonl(path):
    with open(path, 'r') as json_file:
        json_list = list(json_file)
    return [json.loads(json_str) for json_str in json_list]

def save_jsonl(data:list, path:str):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(data)

def load_text_file(path:str)->str:
    with open(path, 'r') as f:
        output = f.read()
    return output

#== Location utils ================================================================================#
def _join_paths(base_path:str, relative_path:str):
    path = os.path.join(base_path, relative_path)
    path = str(Path(path).resolve()) #convert base/x/x/../../src to base/src
    return path 

def get_base_dir():
    """ automatically gets root dir of framework (parent of src) """
    #gets path of the src folder 
    cur_path = os.path.abspath(__file__)
    base_path = cur_path.split('/src')[0] 
    src_path = base_path.split('/src')[0] + '/src'

    #can be called through a symbolic link, if so go out one more dir.
    if os.path.islink(src_path):
        symb_link = os.readlink(src_path)
        src_path = _join_paths(base_path, symb_link)
        base_path = src_path.split('/src')[0]   

    return base_path

#== File Combination Utils =============================================================#
def combine_files(path:str)->dict:
    combined ={}
    for file in os.listdir(path):
        if file == 'combined.json':
            continue
        file_id = file.split('.')[0]
        file_path = os.path.join(path, file)
        data = load_json(file_path)
        combined[file_id] = data
    return combined

def save_combined_json(path:str):
    combined_path = f"{path}/combined.json"
    combined = combine_files(path)
    
    # if existing combined file exists, load it
    if os.path.isfile(combined_path):
        current = load_json(combined_path)
        combined = {**current, **combined}

    combined = {k:v for k, v in sorted(combined.items())}
    save_json(combined, combined_path)

def delete_leftover_files(path):
    combined_path = f"{path}/combined.json"
    combined = load_json(combined_path)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_id = file.split('.')[0]

        if file == 'combined.json':     
            continue

        if file_id in combined.keys():
            os.remove(file_path)
    
