from datasets import load_dataset
from types import SimpleNamespace
from typing import List

def load_summeval()->List[SimpleNamespace]:
    output = []
    summ_eval = load_dataset('mteb/summeval')['test']
    for k, row in enumerate(summ_eval):
        ex = SimpleNamespace(
            context_id=str(k),
            context=row['text'],
            responses=row['machine_summaries'],
            reference=row['human_summaries'][0],
            scores={
                'coherency':row['coherence'],
                'fluency':row['fluency'],
                'consistency':row['consistency'],
                'relevance':row['relevance']
            }
        )
        output.append(ex)
    return output

def load_podcast()->List[SimpleNamespace]:
    podcast_data = load_dataset("potsawee/podcast_summary_assessment")['evaluation']
    system_ids = ['R1'] + [f"E{k}" for k in range(1,4)] + [f"A{k}" for k in range(1,17)]
    system2id = {v:k for k, v in enumerate(system_ids)}

    episodes = set(row['episode_id'] for row in podcast_data)
    episode2id = {v:str(k) for k, v in enumerate(episodes)}

    # splitting 3580 -> 179 * 20
    podcast_179 = {}
    score_mapping = {'B':0, 'F': 1, 'G': 2, 'E': 3} # Bad, Fair, Good, Excellent
    for row in podcast_data:
        episode_id = row['episode_id']
        system_id = row['system_id']
        if episode_id not in podcast_179:
            podcast_179[episode_id] = SimpleNamespace(
                context_id=episode2id[row['episode_id']],
                #context_id=row['episode_id'],
                context=row['transcript'],
                responses=[None for _ in range(20)],
                scores={'overall': [None for _ in range(20)]},
            )
        podcast_179[episode_id].responses[system2id[system_id]] = row['summary']
        podcast_179[episode_id].scores['overall'][system2id[system_id]] = score_mapping[row['score']]

    podcast_179 = [v for v in podcast_179.values()]
    return podcast_179
