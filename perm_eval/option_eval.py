import os
import random
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict

from src.data import get_dataset_category, load_mcqa_data, load_nlg_data
from src.models.llama import PromptedLlama2System
from src.models.flant5 import PromptedFlanT5System
from src.utils.general import save_pickle

from src.prompts import get_prompt_template, get_label_words, SCORE_TO_ADJECTIVE

def run_bias_search(
    dataset:str,
    system_name:str,
    score_type:str=None,
    peft_path:str=None,
    device:str='cuda'
):
    #== define prompt templates for task ==========================================================#
    prompt_templates = get_prompt_template(dataset)
    decoder_prefix, label_words = get_label_words(dataset)
    dataset_category = get_dataset_category(dataset)

    if 'llama' in system_name:
        prompt_templates = [template + '\n\nAnswer:' for template in prompt_templates]
    
    #== data depending on task ====================================================================#
    assert dataset_category == 'multiple_choice'

    _, _, data = load_mcqa_data(dataset)

    #== load LLM system ===========================================================================#
    if 'flan' in system_name:
        system = PromptedFlanT5System(
            system_name=system_name,
            decoder_prefix=decoder_prefix,
            label_words=[' '+x for x in label_words],
            device=device
        )

    elif ('llama' in system_name) or ('vicuna' in system_name):
        system = PromptedLlama2System(
            system_name=system_name,
            decoder_prefix=decoder_prefix,
            label_words=label_words,
            device=device
        )
    
    if peft_path:
        system.load_peft_model(peft_path)

    #== Make Dir if not already exists ============================================================#
    if not os.path.isdir(f"outputs/{dataset}"):
        os.makedirs(f"outputs/{dataset}")
    
    #== Go through all selected examples ==========================================================#
    for p_num, prompt_template in enumerate(prompt_templates):
        if not peft_path:
            output_path = f"outputs/{dataset}/{system_name}_p{p_num}.pk"
        else:
            peft_str = peft_path.replace('peft_models/', '').replace('/', '-')
            output_path = f"outputs/{dataset}/{peft_str}_p{p_num}.pk"

        #skip files already processed
        if os.path.isfile(output_path):
            continue

        #== code for multiplechoice ===============================================================#
        output_json = []

        for ex in tqdm(data):
            ex_output = {}
            filled_prompt = prompt_template.format(
                context=ex.context, 
                question=ex.question,
                option_1=ex.options[0],
                option_2=ex.options[1],
                option_3=ex.options[2],
                option_4=ex.options[3]
            )

            options = [filled_prompt + option for option in ex.options]
            print('--'*50)
            for x in options:
                print(x)
            print('--'*50)
            output = [system.text_loglikelihood(input_text=option) for option in options]
            print(output)
            ex_output['q_id'] = ex.ex_id.split('-')[0]
            ex_output['logits'] = np.array([output.logits])
            ex_output['raw_probs'] = np.array([output.raw_probs])
            ex_output['labels'] = [ex.label]
            output_json.append(ex_output)

        save_pickle(output_json, output_path)

def create_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--system-name', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--score-type', type=str, default=None, help='which transformer to use')
    parser.add_argument('--peft-path', type=str, default=None, help='weights to use (if provided)')
    parser.add_argument('--device', type=str, default=None, help='the device to use for generate')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    run_bias_search(
        dataset=args.dataset,
        system_name=args.system_name,
        score_type=args.score_type,
        peft_path=args.peft_path,
        device=args.device
    )

#python mcrc_bias.py --dataset race-M --system-name llama2-7b-chat --device cuda:3