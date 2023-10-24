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

from src.prompts import STORY_COMPLETION_TEMPLATES

# PROMPT_TEMPLATES = [
#         'Complete the following with the best ending choice:\n\n{context}\n\n\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}',
#         'Select which of the following options best completes the context\n\nContext: {context}\n\nOptions:\nA:{option_1}\nB:{option_2}\nC:{option_3}\nD:{option_4}',
#         '{context}\n\nWhich option best completes the previous context?\nA){option_1}\nB){option_2}\nC){option_3}\nD){option_4}'
#  ]

def run_completeion_bias_search(
    dataset:str,
    system_name:str,
    device:str='cuda'
):
    #== define prompt templates for task ======================================================================#
    prompt_templates = STORY_COMPLETION_TEMPLATES
    if 'llama' in system_name:
        prompt_templates = [template + '\n\nAnswer:' for template in prompt_templates]
    
    label_words = ['A', 'B', 'C', 'D']
    
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
        
    #== Make Dir if not already exists ===============================================================#
    if not os.path.isdir(f"outputs/{dataset}"):
        os.makedirs(f"outputs/{dataset}")
    
    #== Go through all selected examples =============================================================#
    for p_num, prompt_template in enumerate(prompt_templates):
        output_path = f"outputs/{dataset}/{system_name}_p{p_num}.pk"
        #skip files already processed
        if os.path.isfile(output_path):
            continue

        output_json = []
        for k1, question in enumerate(tqdm(grouped_data)):
            ex_output = {}
            labels = []
            logits = []
            raw_probs = []
            selected_dist=np.zeros(4)
            for k2, ex in enumerate(question):
                filled_prompt = prompt_template.format(
                    context=ex.context, 
                    option_1=ex.options[0],
                    option_2=ex.options[1],
                    option_3=ex.options[2],
                    option_4=ex.options[3]
                )
                
                if k1<5 and k2==0:
                    print(filled_prompt)
                    
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
    parser.add_argument('--device', type=str, default=None, help='the device to use for generate')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    run_completeion_bias_search(
        dataset=args.dataset,
        system_name=args.system_name,
        device=args.device
    )

#python mcrc_bias.py --dataset race-M --system-name llama2-7b-chat --device cuda:3