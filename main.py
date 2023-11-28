import sys

import numpy
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
        # self.similarity_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

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
    return list(zip(*linear_sum_assignment(-B))) #minimum weight matching in bipartite graphs


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

    def compute_scores(self, src_embed, dst_embed,
                       **kwargs):  # sm = sm_similarity(tuple([obj_gt['label']]), tuple([obj_det['label']])) # get into the format of ('token',)
        scores_matrix = [[cosine_sim(x, y) for y in dst_embed] for x in src_embed]
        return np.array(scores_matrix)

    def compute_precision_recall(self, src: str, dst: str, assignment_method="hungarian",
                                 debug_print=False, **kwargs) -> (float, float):

        if not src or not dst:
            return (0., 0.)

        chunk_2_paragraph = kwargs.pop('chunk_2_paragraph', False)
        preliminary_embds = kwargs.pop('preliminary_embds', False)
        if chunk_2_paragraph:
            if preliminary_embds:
                embds_chunks = kwargs.pop('embds_chunks', None)
                embds_seg_dialog = kwargs.pop('embds_seg_dialog', None)

                # isinstance(src, np.ndarray)
                # isinstance(dst, np.ndarray)
                A = self.compute_scores(src_embed=embds_chunks, dst_embed=embds_seg_dialog, **kwargs)
            else:
                A = self.compute_scores(src=src, dst=dst, **kwargs)
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
        return np.sum([A[x] for x in res]) / A.shape ,res

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

def training_zs(df_ref, result_dir, evaluator):
    max_seq_len = int(evaluator.smanager.similarity_model.max_seq_length *0.6)#WordPeice tokens to
    embds_seg_dialog_summ = list()
    all_embs_chunks = list()
    all_chunk = list()
    chunk_2_summ_id = dict()
    dialog_2_seg_id = dict()
    len_chunk_prev = 0
    len_dialog_seg_prev = 0
    all_seg_dialogs = list()

    for idx, (dialog, summ) in enumerate(tqdm.tqdm(zip(df_ref['dialogue'], df_ref['summary']))):
        chunks = flatten([t.strip().split('.') for t in summ.strip().split(',')])[:-1]
        chunks = [c.strip() for c in chunks]
        all_chunk.append(chunks)
        all_segmented_dialog = paragraph_seg(dialog, max_seq_len)
        all_seg_dialogs.append(all_segmented_dialog)

        embs_chunks = evaluator.encode_tokens(chunks)
        all_embs_chunks.append(embs_chunks)
        segmented_dialog_embs = evaluator.encode_tokens(all_segmented_dialog)
        embds_seg_dialog_summ.append(segmented_dialog_embs)

        chunk_2_summ_id[idx] = list(np.arange(len_chunk_prev, len_chunk_prev + len(chunks)))
        # assert ( not (any([True if len(flatten(chunk)) > max_seq_len else False for chunk in chunks])))
        len_chunk_prev += len(chunks)

        dialog_2_seg_id[idx] = list(np.arange(len_dialog_seg_prev, len_dialog_seg_prev + len(all_segmented_dialog)))
        len_dialog_seg_prev += len(all_segmented_dialog)


    with open(os.path.join(result_dir, 'training_embds.pkl'), 'wb') as f:
        pickle.dump((embds_seg_dialog_summ, all_embs_chunks), f)


    with open(os.path.join(result_dir, 'training_chunk_2_summ_id.pkl'), 'wb') as f:
        pickle.dump(chunk_2_summ_id, f)

    with open(os.path.join(result_dir, 'training_dialog_2_seg_id.pkl'), 'wb') as f:
        pickle.dump(dialog_2_seg_id, f)

    with open(os.path.join(result_dir, 'training_all_chunk.pkl'), 'wb') as f:
        pickle.dump(all_chunk, f)

    with open(os.path.join(result_dir, 'training_all_seg_dialogs.pkl'), 'wb') as f:
        pickle.dump(all_seg_dialogs, f)


def create_dialog_seg_embds(dialogue, result_dir, evaluator, tag='test'):
    max_seq_len = int(evaluator.smanager.similarity_model.max_seq_length *0.6)#WordPeice tokens to

    embds_seg_dialog_summ = list()
    all_embs_chunks = list()
    dialog_2_seg_id = dict()
    len_dialog_seg_prev = 0
    all_seg_dialogs = list()

    for idx, (dialog) in enumerate(tqdm.tqdm(dialogue)):
        all_segmented_dialog = paragraph_seg(dialog, max_seq_len)
        all_seg_dialogs.append(all_segmented_dialog)
        segmented_dialog_embs = evaluator.encode_tokens(all_segmented_dialog)
        embds_seg_dialog_summ.append(segmented_dialog_embs)

        dialog_2_seg_id[idx] = list(np.arange(len_dialog_seg_prev, len_dialog_seg_prev + len(all_segmented_dialog)))
        len_dialog_seg_prev += len(all_segmented_dialog)


    # [evaluator.compute_scores(x,y) for x, y in zip(embds_seg_dialog_summ, all_embs_chunks)]
    with open(os.path.join(result_dir, tag + '_embds.pkl'), 'wb') as f:
        pickle.dump((embds_seg_dialog_summ), f)

    with open(os.path.join(result_dir, tag + '_dialog_2_seg_id.pkl'), 'wb') as f:
        pickle.dump(dialog_2_seg_id, f)

    with open(os.path.join(result_dir, tag + '_all_seg_dialogs.pkl'), 'wb') as f:
        pickle.dump(all_seg_dialogs, f)

def paragraph_seg(dialog, max_seq_len):
    all_segmented_dialog = list()
    ptr_beg = 0
    ptr_end = 0
    while ptr_beg < len(dialog):
        delimiter = [(m.start(), m.end()) for m in re.finditer('\r', dialog[ptr_beg:]) if m.start() < max_seq_len]
        if delimiter != []:
            ptr_end = delimiter[-1][0]
        else:
            ptr_end = max_seq_len  # brute force

        if ptr_end <= 1: # when no delimiter in the range then brute force
            ptr_end = max_seq_len # brute force
        window_dialog = re.sub('\r\n', ' ', dialog[ptr_beg:ptr_beg + ptr_end]).strip()
        ptr_beg += ptr_end
        all_segmented_dialog.append(window_dialog)

    return all_segmented_dialog

def embeddings_extract(sentence: list, key_tag:str , result_dir: str, evaluator):
    batch_size = 32

    if len(sentence) % batch_size != 0:  # all images size are Int multiple of batch size
        pad = batch_size - len(sentence) % batch_size
    else:
        pad = 0

    all_embeds_dialog = list()
    bns = len(sentence)//batch_size

    for idx in np.arange(bns):
        batch_sent = sentence[idx * batch_size: (idx + 1) * batch_size]
        batch_sent = [re.sub('\r\n', '', dialog) for dialog in batch_sent] # TODO HK@@
        embs = evaluator.encode_tokens(batch_sent)
        all_embeds_dialog.append(embs)

        if idx % 10 == 0:
            with open(os.path.join(result_dir, str(key_tag) + '.pkl'), 'wb') as f:
                pickle.dump(all_embeds_dialog, f)
    if pad != 0:
        batch_sent = sentence[batch_size * (len(sentence)//batch_size): len(sentence)]
        embs = evaluator.encode_tokens(batch_sent)
        all_embeds_dialog.append(embs)

    all_embeds_dialog = np.concatenate(all_embeds_dialog)

    with open(os.path.join(result_dir, str(key_tag) + '.pkl'), 'wb') as f:
        pickle.dump(all_embeds_dialog, f)

    return all_embeds_dialog

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
    analyse_reference = False
    train_data = False
    pre_compute_embeddings = True

    print("Max Sequence Length:", evaluator.smanager.similarity_model.max_seq_length)

    if analyse_reference:
        if 1:
            df_ref = pd.read_csv('reference.csv')
            training_zs(df_ref, result_dir, evaluator)
        else:
            evaluation_ac(result_dir=result_dir, pkl_file='training.pkl')
        return
# TODO check "" vs. ''
    if pre_compute_embeddings:
        if train_data:
            df_ref = pd.read_csv('reference.csv')
            # dialog_list = df_ref['dialogue'].to_list()
            # summary = df_ref['summary']
            # max_seq_len = int(evaluator.smanager.similarity_model.max_seq_length * 0.6)  # WordPeice tokens to

            training_zs(df_ref, result_dir, evaluator)


            # chunk_2_summ_id = dict()
            # chunk_list = list()
            # len_chunk_prev = 0
            # for idx, summ in enumerate(tqdm.tqdm(summary)):
            #     chunks = flatten([t.strip().split('.') for t in summ.strip().split(',')])[:-1]
            #     chunks = [c.strip() for c in chunks]
            #     print(len(chunks), list(np.arange(len_chunk_prev,len_chunk_prev+len(chunks))))
            #     chunk_2_summ_id[idx] = list(np.arange(len_chunk_prev, len_chunk_prev+len(chunks)))
            #     assert(len(flatten(chunks)) <= max_seq_len)
            #     len_chunk_prev += len(chunks)
            #     chunk_list.extend(chunks)
            #
            # key_tag_d = 'dialogue_train'
            # key_tag_chunk = 'chunks_train'
            #
            # with open(os.path.join(result_dir, 'chunk_2_summ_id.pkl'), 'wb') as f:
            #     pickle.dump(chunk_2_summ_id, f)

            # chunks = flatten([t.strip().split('.') for t in summ.strip().split(',') for summ in summary])[:-1]
            # chunk_list = [c.strip() for c in chunks]


        else: # test data
            df_chunks = pd.read_csv('summary_pieces.csv')
            df_dialog = pd.read_csv('dialogues.csv')
            dialog_list = df_dialog['dialogue'].to_list()
            chunk_list = df_chunks['summary_piece'].to_list()
            # key_tag_d = 'dialogue'
            key_tag_chunk = 'test_chunks'

            # all_embds_dialog = embeddings_extract(dialog_list,
            #                     key_tag=key_tag_d,
            #                     result_dir=result_dir,
            #                     evaluator=evaluator)
            create_dialog_seg_embds(dialog_list, result_dir, evaluator)

            all_embds_chunks = embeddings_extract(chunk_list,
                                key_tag=key_tag_chunk,
                                result_dir=result_dir,
                                evaluator=evaluator)
    else:
        if train_data:
            # key_tag_d = 'dialogue_train'
            # key_tag_chunk = 'chunks_train'
            with open(os.path.join(result_dir, 'training_chunk_2_summ_id.pkl'), 'rb') as f:
                chunk_2_summ_id = pickle.load(f)
            with open(os.path.join(result_dir, 'training_dialog_2_seg_id.pkl'), 'rb') as f:
                dialog_2_seg_id = pickle.load(f)

            with open(os.path.join(result_dir, 'training_embds.pkl'), 'rb') as f:
                (all_embds_seg_dialog, all_embds_chunks) = pickle.load(f)

            with open(os.path.join(result_dir, 'training_all_chunk.pkl'), 'rb') as f:
                all_chunk = pickle.load(f)

            with open(os.path.join(result_dir, 'training_all_seg_dialogs.pkl'), 'rb') as f:
                all_seg_dialogs = pickle.load(f)


        else:
            key_tag_d = 'dialogue'
            key_tag_chunk = 'chunks'

            with open(os.path.join(result_dir, str(key_tag_chunk) + '.pkl'), 'rb') as f:
                all_embds_chunks = pickle.load(f)

            with open(os.path.join(result_dir, str(key_tag_d) + '.pkl'), 'rb') as f:
                all_embds_seg_dialog = pickle.load(f)

    _, res = evaluator.compute_precision_recall(src=flatten(all_chunk), dst=flatten(all_seg_dialogs),
                                                        assignment_method='hungarian',
                                                        debug_print=False,
                                                        chunk_2_paragraph=True,
                                                        preliminary_embds=True,
                                                        embds_chunks = np.concatenate(all_embds_chunks),
                                                        embds_seg_dialog = np.concatenate(all_embds_seg_dialog))

    # pr_gpt, re_gpt = evaluator.compute_precision_recall(df_chunks['summary_piece'].to_list(),
    #                                                     df_dialog['dialogue'].to_list(),
    #                                                     assignment_method='hungarian',
    #                                                     debug_print=True,
    #                                                     chunk_2_paragraph=True)

    tp = 0
    all_errors = list()
    for tup in res:
        summ_ind = [key for key, val in chunk_2_summ_id.items() if tup[0] in val][0]
        dialog_ind = [key for key, val in dialog_2_seg_id.items() if tup[1] in val][0]
        if dialog_ind == summ_ind:
            tp += 1
        else:
            all_errors.append((tup, (summ_ind, dialog_ind),
                               chunk_2_summ_id[summ_ind].index(tup[0]),
                               dialog_2_seg_id[dialog_ind].index(tup[1]),
                               all_chunk[summ_ind][chunk_2_summ_id[summ_ind].index(tup[0])],
                               all_seg_dialogs[dialog_ind][dialog_2_seg_id[dialog_ind].index(tup[1])]))
    print("Recall {}".format(tp/len(res)))


    print('ka')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
"""
cosine_sim(*evaluator.encode_tokens(["This is an example sentence", "Each sentence is converted"]))
"""