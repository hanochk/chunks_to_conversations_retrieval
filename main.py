import sys
import requests
import os
from scipy.optimize import linear_sum_assignment
import pickle
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('qtagg')
import ipympl
import matplotlib.pyplot as plt
import re
import json
import copy
import subprocess

import numpy as np
import torch
import tqdm
# import spacy
# import nltk
# from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from sentence_transformers import SentenceTransformer

import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from eval_metrices import roc_plot, p_r_plot

def flatten(lst):
    return [x for l in lst for x in l]

def cosine_sim(x,y):
    return np.dot(x,y) / (np.linalg.norm(x)*np.linalg.norm(y))

def compare_cross_lists(l1, l2):
    return np.any([x in l2 for x in l1])

def triplet_from_rel(rel):
    return (rel.subject.names[0], rel.predicate, rel.object.names[0])

class SimilarityManager:
    def __init__(self):
        # nlp = spacy.load('en_core_web_sm')
        # nlp = spacy.load('en_core_web_lg')
        # nlp.add_pipe("spacy_wordnet", after='tagger')
        # nlp.add_pipe("spacy_wordnet", after='tagger', config={'lang': nlp.lang})
        # self.nlp = nlp
        self.similarity_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        # SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # more recent
        if torch.cuda.device_count() > 0:
            self.similarity_model.cuda()

    def similarity(self, src, target):
        rc = []
        s1 = self.nlp(src)
        s2 = self.nlp(target)
        for w in s1:
            if w.pos_ not in ['NOUN', 'ADJ', 'ADV', 'VERB', 'PROPN'] and len(s1) > 1:
                continue
            rc.append(max([w.similarity(x) for x in s2]))
        return np.mean(rc)

    # def compare_cross_synsets(self, text1, text2):
    #     t1 = self.nlp(text1)
    #     t2 = self.nlp(text2)
    #     return compare_cross_lists([x._.wordnet.synsets() for x in t1], [x._.wordnet.synsets() for x in t2])

    def compare_triplet(self, t1, t2, method='bert', invert_src=False, invert_dst=False, **kwargs):
        if len(t1) != len(t2):
            return 0.
        sim = 1.
        if method == 'bert':
            if len(t1) == 2:  # attribute tuple, invert
                if invert_src:
                    t1 = [t1[1], t1[0]]
                if invert_dst:
                    t2 = [t2[1], t2[0]]
            embs = self.similarity_model.encode([' '.join(t1).lower(), ' '.join(t2).lower()])
            sim = cosine_sim(*embs)
        elif method == 'meteor':
            return nltk.translate.meteor_score.single_meteor_score(' '.join(t1).lower().split(),
                                                                   ' '.join(t2).lower().split())
        else:
            for x, y in zip(t1, t2):
                if method == 'wordnet':
                    sim *= self.compare_cross_synsets(x, y)
                elif method == 'spacy':
                    sim *= self.similarity(x, y)
                else:
                    print("Unknown similarity method: {}".format(method))
        return sim


def greedy_match(similarity_matrix):
    A = similarity_matrix
    indices = []
    while np.any(A > 0):
        ind = np.unravel_index(np.argmax(A, axis=None), A.shape)
        indices.append(ind)
        A[ind[0]] = 0
        A[:, ind[1]] = 0

    return indices


def hungarian_match(similarity_matrix):
    B = similarity_matrix
    return list(zip(*linear_sum_assignment(-B))) #inimum weight matching in bipartite graphs


class VGEvaluation:
    def __init__(self):
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        self.smanager = SimilarityManager()

    def recall_triplet(self, src, dst, **kwargs):
        if not dst:
            return 0.
        scores = [self.smanager.compare_triplet(src, x, **kwargs) for x in dst]
        return max(scores)

    # src: A list of triplets
    # dst: A list of triplets
    def recall_triplets(self, src, dst, **kwargs):
        rc = [self.recall_triplet(x, dst, **kwargs) for x in src]
        return rc
        # return np.mean(rc)

    def compute_triplet_scores(self, src, dst, **kwargs):
        scores_matrix = [[self.smanager.compare_triplet(x, y, **kwargs) for y in dst] for x in src]
        return np.array(scores_matrix)

    def encode_tokens (self, t1: list):
        embs = self.smanager.similarity_model.encode([t.lower() for t in t1])
        return embs

    def sm_similarity(self, t1: str, t2: str):
        embs = self.smanager.similarity_model.encode([t1.lower(), t2.lower()])
        sim = cosine_sim(*embs)
        return sim

    def compute_scores(self, src, dst,
                       **kwargs):  # sm = sm_similarity(tuple([obj_gt['label']]), tuple([obj_det['label']])) # get into the format of ('token',)
        scores_matrix = [[self.sm_similarity(x, y) for y in dst] for x in src]
        return np.array(scores_matrix)

    def compute_precision_recall(self, src, dst, assignment_method="hungarian",
                                 debug_print=False, **kwargs) -> (float, float):
        if not src or not dst:
            return (0., 0.)

        chunk_2_paragraph = kwargs.pop('chunk_2_paragraph', False)
        if chunk_2_paragraph:


            A = self.compute_scores(src, dst, **kwargs)
        else:
            A = self.compute_triplet_scores(src, dst, **kwargs)


        if assignment_method == 'greedy':
            func = greedy_match
        elif assignment_method == 'hungarian':
            func = hungarian_match
        elif assignment_method == 'relaxed':
            recall = self.recall_triplets_mean(dst, src, **kwargs)
            precision = self.recall_triplets_mean(src, dst, **kwargs)
            return (precision, recall)
        else:
            raise "compute_precision_recall: Unknown method"
        res = func(A.copy())
        if debug_print:
            for ind in res:
                print("{} --- {} ({})".format(src[ind[0]], dst[ind[1]], A[ind]))
        return np.sum([A[x] for x in res]) / A.shape

    def recall_triplets_mean(self, src, dst, **kwargs):
        rc = self.recall_triplets(src, dst, **kwargs)
        if not rc:
            return 0.
        return np.mean(rc)

    def total_recall_triplets(self, src_triplets, dst_triplets, methods=('bert', 'bert', 'bert')):
        total_recall = []
        for i in [1, 2, 3]:
            dst_i = [x for x in dst_triplets if len(x) == i]
            src_i = [x for x in src_triplets if len(x) == i]
            total_recall.extend(self.recall_triplets(src_i, dst_i, method=methods[i - 1]))
        return total_recall

def training_zc(df_ref, result_dir, evaluator):
    embeds_dialog_summ = list()
    for idx, (dialog, summ) in enumerate(tqdm.tqdm(zip(df_ref['dialogue'], df_ref['summary']))):
        chunks = flatten([t.strip().split('.') for t in summ.strip().split(',')])[:-1]
        chunks = [c.strip() for c in chunks]
        if 0:
            string = re.sub('\r\n', '', dialog)
        ff = [dialog]
        ff.extend(chunks)
        embs = evaluator.encode_tokens(ff)
        embeds_dialog_summ.append(embs)
        if idx % 10 == 0:
            with open(os.path.join(result_dir, 'training.pkl'), 'wb') as f:
                pickle.dump(embeds_dialog_summ, f)

    with open(os.path.join(result_dir, 'training.pkl'), 'wb') as f:
        pickle.dump(embeds_dialog_summ, f)

def evaluation_ac(result_dir, pkl_file='training1.pkl'):
    with open(os.path.join(result_dir, pkl_file), 'rb') as f:
        embeds_dialog_summ = pickle.load(f)

    all_dist_positive = list()
    all_dist_neg = list()
    for ix, embs in enumerate(tqdm.tqdm(embeds_dialog_summ)):
        cos_dists_pos = [cosine_sim(embs[0], t2) for t2 in embs[1:]]
        all_dist_positive.extend(cos_dists_pos)
        neg_chunks = [embs[1:] for idx, embs in enumerate(embeds_dialog_summ) if idx != ix]
        cos_dists_neg = [cosine_sim(embs[0], t2) for t2 in np.concatenate(neg_chunks)]
        all_dist_neg.extend(cos_dists_neg)

    all_predictions = all_dist_positive + all_dist_neg
    all_targets = np.concatenate((np.ones_like(all_dist_positive), np.zeros_like(all_dist_neg)))
    # all_targets_one_hot = label_binarize(all_targets, classes=[0, 1])
    roc_plot(all_targets, all_predictions, positive_label=1, save_dir=result_dir,
             unique_id='chunks to dialogs classifier')


    neg_hist, neg_bins_edges = np.histogram(all_dist_neg, bins=50, density=True)
    pos_hist, pos_bins_edges = np.histogram(all_dist_positive, bins=50, density=True)
    pos_bins = 0.5 * (pos_bins_edges[:-1] + pos_bins_edges[1:])
    neg_bins = 0.5 * (neg_bins_edges[:-1] + neg_bins_edges[1:])
    plt.plot(pos_bins, pos_hist, 'r', label='positives')
    plt.plot(neg_bins, neg_hist, 'b', label='negatives')
    plt.title("Cosine similarity distribution of chunks to dialogs; support={}".format(len(embeds_dialog_summ)))
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig(
            os.path.join(result_dir, pkl_file.split('.')[0]+'.jpg'))

def main():
    # Use a breakpoint in the code line below to debug your script.
    result_dir = r'C:\Users\h00633314\HanochWorkSpace\Projects\chunk_back_to_summary\chunks_to_conversations'
    evaluator = VGEvaluation()




    if 0:
        df_ref = pd.read_csv('reference.csv')
        training_zc(df_ref, result_dir, evaluator)
    else:
        evaluation_ac(result_dir=result_dir, pkl_file='training.pkl')

    df_chunks = pd.read_csv('summary_pieces.csv')
    df_dialog = pd.read_csv('dialogues.csv')


    with open(os.path.join(result_dir, 'training.pkl'), 'rb') as f:
        results_meta = pickle.load(f)
    pr_gpt, re_gpt = evaluator.compute_precision_recall([df_ref['summary'].loc[0]],
                                                        [df_ref['dialogue'].loc[0]],
                                                        assignment_method='hungarian',
                                                        debug_print=True,
                                                        chunk_2_paragraph=True)
    # pr_gpt, re_gpt = evaluator.compute_precision_recall(df_chunks['summary_piece'].to_list(),
    #                                                     df_dialog['dialogue'].to_list(),
    #                                                     assignment_method='hungarian',
    #                                                     debug_print=True,
    #                                                     chunk_2_paragraph=True)
    print('ka')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
