import os
import random
import numpy as np
import argparse

from tqdm import tqdm
from collections import defaultdict

from src.data import load_mcrc_data
from src.models.llama import PromptedLlama2System
from src.models.flant5 import PromptedFlanT5System
from src.utils.general import save_pickle

from src.prompts import MCRC_TEMPLATES

def run_mcrc_bias_search(
    dataset:str,
    system_name:str,
    peft_path:str=None,
    device:str='cuda'
):
    #== define prompt templates for task ======================================================================#
    prompt_templates = MCRC_TEMPLATES
    if 'llama' in system_name:
        prompt_templates = [template + '\n\nAnswer:' for template in prompt_templates]
        
    #== prepare MCRC dataset ================================================================================#
    _, _, data = load_mcrc_data(dataset, all_perm=True)
    
    grouped_data = defaultdict(list)
    for ex in data:
        grouped_data[ex.ex_id.split('-')[0]] += [ex]
        
    #returns the middle question id
    q_id = lambda ex_id: int(ex_id.split('_')[-1].split('-')[0])

    grouped_data = [v for v in grouped_data.values()]
    #random.Random(4).shuffle(grouped_data)
    #grouped_data = sorted(grouped_data[:500], key=lambda x:(x[0].ex_id[0], q_id(x[0].ex_id)))

    #== load LLM system ==============================================================================#
    if 'flan' in system_name:
        system = PromptedFlanT5System(
            system_name=system_name,
            decoder_prefix='',
            label_words=[' '+x for x in label_words],
            device=device
        )

    elif ('llama' in system_name) or ('vicuna' in system_name):
        system = PromptedLlama2System(
            system_name=system_name,
            decoder_prefix='',
            label_words=label_words,
            device=device
        )
    
    if peft_path:
        system.load_peft_model(peft_path)

    #== Make Dir if not already exists ===============================================================#
    if not os.path.isdir(f"outputs/{dataset}"):
        os.makedirs(f"outputs/{dataset}")
    
    #== Go through all selected examples =============================================================#
    for p_num, prompt_template in enumerate(prompt_templates):
        if not peft_path:
            output_path = f"outputs/{dataset}/{system_name}_p{p_num}.pk"
        else:
            peft_str = peft_path.replace('peft_models/', '').replace('/', '-')
            output_path = f"outputs/{dataset}/{peft_str}_p{p_num}.pk"

        #skip files already processed
        if os.path.isfile(output_path):
            continue

        output_json = []
        for question in tqdm(grouped_data):
            ex_output = {}
            labels = []
            logits = []
            raw_probs = []
            selected_dist=np.zeros(4)
            for ex in question:
                filled_prompt = prompt_template.format(
                    context=ex.context, 
                    question=ex.question,
                    option_1=ex.options[0],
                    option_2=ex.options[1],
                    option_3=ex.options[2],
                    option_4=ex.options[3]
                )

                output = system.prompt_classifier_response(input_text=filled_prompt)
                selected_dist[output.pred] += 1

                logits += [output.logits]
                raw_probs += [output.raw_probs]
                labels.append(ex.label)

            ex_output['q_id'] = ex.ex_id.split('-')[0]
            ex_output['logits'] = np.array(logits)
            ex_output['raw_probs'] = np.array(raw_probs)
            ex_output['labels'] = labels

            #context_output['detailed_info'] = detailed

            output_json.append(ex_output)

        save_pickle(output_json, output_path)

def create_parser():
    """ Build Argument Parser """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--system-name', type=str, default='flant5-large', help='which transformer to use')
    parser.add_argument('--peft-path', type=str, default=None, help='weights to use (if provided)')
    parser.add_argument('--device', type=str, default=None, help='the device to use for generate')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    run_mcrc_bias_search(
        dataset=args.dataset,
        system_name=args.system_name,
        peft_path=args.peft_path,
        device=args.device
    )

#python mcrc_bias.py --dataset race-M --system-name llama2-7b-chat --device cuda:3