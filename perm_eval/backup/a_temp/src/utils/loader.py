import os
import scipy
import numpy as np
import re

from collections import defaultdict
from typing import Dict

from src.utils import load_json

class SystemLoader:
    @classmethod
    def load_comparisons(cls, path, lim=None):
        self = cls()
        self.comparisons, self.comparisons_M = self._load_comparisons(path, lim=lim)
        self.ratings = self.comparisons_to_ratings(self.comparisons, self.comparisons_M)
        return self
    
    @classmethod
    def load_comparisons_probs(cls, path, lim=None, balanced=False):
        self = cls()
        self.comparisons_probs, self.comparisons_M = self._load_comparison_probs(path, lim=lim)
        t = self.get_balanced_thresholds(self.comparisons_probs, self.comparisons_M) if balanced else 0.5
        self.comparisons = (self.comparisons_probs > t).astype(int)
        self.ratings = self.comparisons_to_ratings(self.comparisons, self.comparisons_M)
        return self

    #== Load Files by category =======================================================#
    @staticmethod
    def _load_comparisons(path:str, lim:int=None)->Dict[str, int]:
        data = load_json(path)

        ex_ids = [ex_id.split('-') for ex_id in data.keys()] # ex_id="docid-candid1-candid2"
        num_docs  = max([int(x[0]) for x in ex_ids]) + 1
        num_cands = max([int(x[1]) for x in ex_ids] + [int(x[2]) for x in ex_ids]) + 1

        comparisons = np.zeros((num_docs, num_cands, num_cands))
        comparisons_M = np.zeros((num_docs, num_cands, num_cands))

        for ex_id, output in data.items():
            doc_id, sys_id1, sys_id2 = [int(i) for i in ex_id.split('-')]

            output_text = output['output_text']
            if (' A' in output_text) and (not ' B' in output_text):
                comparisons[doc_id, sys_id1, sys_id2] = 1
            elif (' B' in output_text) and (not ' A' in output_text):
                comparisons[doc_id, sys_id1, sys_id2] = 0
            else:
                comparisons[doc_id, sys_id1, sys_id2] = 0.5

            comparisons_M[doc_id, sys_id1, sys_id2] = 1
        
        # check how often the scores were invalid
        fails = np.sum(comparisons==0.5)
        total = np.sum(comparisons_M)
        print(f"loaded ratings with {fails} failures out of {total}")
        
        return comparisons, comparisons_M

    @staticmethod
    def _load_comparison_probs(path:str, lim:int):
        data = load_json(path)

        response_ids = [ex_id.split('-') for ex_id in data.keys()] # ex_id="docid-candid1-candid2"
        num_docs  = max([int(x[0]) for x in response_ids]) + 1
        num_cands = max([int(x[1]) for x in response_ids] + [int(x[2]) for x in response_ids]) + 1

        comparisons_probs = np.zeros((num_docs, num_cands, num_cands))
        comparisons_M = np.zeros((num_docs, num_cands, num_cands))

        for ex_id, output in data.items():
            doc_id, sys_id1, sys_id2 = [int(i) for i in ex_id.split('-')]
            output_logits = output['logits']#[:2]
            probs = scipy.special.softmax(output_logits, axis=0)
            comparisons_probs[doc_id, sys_id1, sys_id2] = probs[0]
            comparisons_M[doc_id, sys_id1, sys_id2] = 1

        return comparisons_probs, comparisons_M

    #== Methods to Convert ===========================================================#
    @staticmethod
    def ratings_to_comparisons(ratings):
        comparisons = ratings[:, None, :] > ratings[:, :, None]
        comparison_M = np.ones_like(comparisons)
        return comparisons, comparison_M

    @staticmethod
    def comparisons_to_ratings(comparisons, comparisons_M):
        wins = (comparisons*comparisons_M).sum(axis=-1) + ((1-comparisons)*(comparisons_M)).sum(axis=-2)
        #games = comparisons_M.sum(axis=-1) + comparisons_M.sum(axis=-2)
        #win_ratio = wins/games
        return wins

    #== Methods dealing with prompt-based classifier =================================#
    @staticmethod
    def probs_to_comparisons(comparisons_logits, t:np.ndarray=None):
        if t is None: t=0
        comparisons = {}
        for ex_id, logits in comparisons_logits.items():
            weighted_logits = logits + np.array([0,t])
            #print(weighted_logits)
            #import time; time.sleep(2)
            pred = np.argmax(weighted_logits, axis=0)
            comparisons[ex_id] = pred
        return comparisons

    @staticmethod
    def get_balanced_thresholds(comparison_probs, comparisons_M):
        for t in np.arange(0,1,0.001):
            if np.mean(comparison_probs[comparisons_M==1] < t) >= 0.5:
                break 
        return t
    
    #== Analysis Functions ==========================================================#
    @property
    def perm_bias(self):
        P_A = self.comparisons.sum()/self.comparisons_M.sum()
        return P_A
    