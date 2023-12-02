import sys

import os
from scipy.optimize import linear_sum_assignment
import pickle
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('qtagg')
import matplotlib.pyplot as plt
import re
import copy

import numpy as np
import torch
import tqdm
# np.seterr(all='raise')
from sentence_transformers import SentenceTransformer

import pandas as pd
import emoji
from abbrev_dict import *
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
        # self.similarity_model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

        if torch.cuda.device_count() > 0:
            self.similarity_model.cuda()


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
        self.smanager = SimilarityManager()

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

                if not isinstance(embds_chunks, np.ndarray):
                    embds_chunks = np.concatenate(embds_chunks)

                if not isinstance(embds_seg_dialog, np.ndarray):
                    embds_seg_dialog = np.concatenate(embds_seg_dialog)

                A = self.compute_scores(src_embed=embds_chunks, dst_embed=embds_seg_dialog, **kwargs)
            else:
                A = self.compute_scores(src=src, dst=dst, **kwargs)
        else:
            A = self.compute_triplet_scores(src, dst, **kwargs)


        if assignment_method == 'greedy':
            func = greedy_match
        elif assignment_method == 'hungarian':
            func = hungarian_match
        else:
            raise "compute_precision_recall: Unknown method"
        res = func(A.copy())
        if debug_print:
            for ind in res:
                print("{} --- {} ({})".format(src[ind[0]], dst[ind[1]], A[ind]))
        return np.sum([A[x] for x in res]) / A.shape ,res


def training_zs(df_ref, result_dir: str, evaluator, clean_noisy_segment: bool=True):
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
        all_segmented_dialog = paragraph_seg(dialog, max_seq_len, clean_noisy_segment=clean_noisy_segment)
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

    tag = ['norm_noisy_segment_' if clean_noisy_segment else ''][0]
    with open(os.path.join(result_dir, tag + 'training_embds.pkl'), 'wb') as f:
        pickle.dump((embds_seg_dialog_summ, all_embs_chunks), f)


    with open(os.path.join(result_dir, tag + 'training_chunk_2_summ_id.pkl'), 'wb') as f:
        pickle.dump(chunk_2_summ_id, f)

    with open(os.path.join(result_dir, tag + 'training_dialog_2_seg_id.pkl'), 'wb') as f:
        pickle.dump(dialog_2_seg_id, f)

    with open(os.path.join(result_dir, tag + 'training_all_chunk.pkl'), 'wb') as f:
        pickle.dump(all_chunk, f)

    with open(os.path.join(result_dir, tag  +'training_all_seg_dialogs.pkl'), 'wb') as f:
        pickle.dump(all_seg_dialogs, f)


def create_dialog_seg_embds(dialogue, result_dir, evaluator, clean_noisy_segment, tag='test'):
    max_seq_len = int(evaluator.smanager.similarity_model.max_seq_length *0.6)#WordPeice tokens to

    embds_seg_dialog_summ = list()
    all_embs_chunks = list()
    dialog_2_seg_id = dict()
    len_dialog_seg_prev = 0
    all_seg_dialogs = list()

    for idx, (dialog) in enumerate(tqdm.tqdm(dialogue)):
        all_segmented_dialog = paragraph_seg(dialog, max_seq_len, clean_noisy_segment=clean_noisy_segment)
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

def paragraph_seg(dialog: str, max_seq_len: int, clean_noisy_segment: bool=False):
    all_segmented_dialog = list()
    ptr_beg = 0
    ptr_end = 0
    all_dirty_window = list()
    while ptr_beg < len(dialog):
        delimiter = [(m.start(), m.end()) for m in re.finditer('\r', dialog[ptr_beg:]) if m.start() < max_seq_len]
        if delimiter != []:
            ptr_end = delimiter[-1][0]
        else:
            ptr_end = max_seq_len  # brute force

        if ptr_end <= 1: # when no delimiter in the range then brute force
            ptr_end = max_seq_len # brute force

        if clean_noisy_segment:
            window_dialog = re.sub('\r\n', ' . ', dialog[ptr_beg:ptr_beg + ptr_end]).strip()
            all_dirty_window.append(window_dialog) # for debug
            if 1:
                window_dialog = ' '.join([convert_abbrev(x) for x in window_dialog.split(' ')])
            window_dialog = ' '.join([remove_abbrev(x) for x in window_dialog.split(' ')])
            window_dialog = ' '.join([remove_abbrev_post_comma(x) for x in window_dialog.split(' ')])
            window_dialog = ' '.join([remove_abbrev_post_period(x) for x in window_dialog.split(' ')])
            # window_dialog = remove_abbrev_dialog(window_dialog)
            window_dialog = window_dialog.replace(': ', ' said ') # replace : after name with action
            window_dialog = window_dialog.replace(' . ', ' ') # replace : after name with action

        else:
            window_dialog = re.sub('\r\n', ' ', dialog[ptr_beg:ptr_beg + ptr_end]).strip()

        ptr_beg += ptr_end
        all_segmented_dialog.append(window_dialog)
        # [(x, y) for x, y in zip(all_segmented_dialog, all_dirty_window)]
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
    result_dir = r'C:\Users\h00633314\HanochWorkSpace\Projects\chunk_back_to_summary\chunks_to_conversations\bin'
    evaluator = VGEvaluation()

    train_data = True
    pre_compute_embeddings = False
    clean_noisy_segment = False
    denoise_tag = ['norm_noisy_segment_' if clean_noisy_segment else ''][0]

    print("Max Sequence Length:", evaluator.smanager.similarity_model.max_seq_length)

# TODO check "" vs. ''
    if pre_compute_embeddings:
        print("precompute embeddings saved to pickles")
        compute_embeddings(clean_noisy_segment, evaluator, result_dir, train_data)
        return
    else:
        all_chunk, all_embds_chunks, all_embds_seg_dialog, all_seg_dialogs, chunk_2_summ_id, dialog_2_seg_id = load_pre_computed_embeddings(
            denoise_tag, result_dir, train_data)
    if 0:
        with open(os.path.join(result_dir, denoise_tag + 'training_bipartite_matching.pkl'), 'rb') as f:
            res = pickle.load(f)

    else:
        print('Running minimum cost assignment based matching')
        _, res = evaluator.compute_precision_recall(src=flatten(all_chunk), dst=flatten(all_seg_dialogs),
                                                        assignment_method='hungarian',
                                                        debug_print=False,
                                                        chunk_2_paragraph=True,
                                                        preliminary_embds=True,
                                                        embds_chunks = all_embds_chunks,
                                                        embds_seg_dialog = all_embds_seg_dialog)

    if train_data:
        tp = 0
        all_errors = list()
        chunk_2_dialog_match = dict()
        for tup in res:
            summ_ind = [key for key, val in chunk_2_summ_id.items() if tup[0] in val][0]
            dialog_ind = [key for key, val in dialog_2_seg_id.items() if tup[1] in val][0]
            if not (chunk_2_dialog_match.get(dialog_ind, None)):
                chunk_2_dialog_match[dialog_ind] = [np.concatenate(all_chunk)[tup[0]]]
            else:
                chunk_2_dialog_match[dialog_ind].append(np.concatenate(all_chunk)[tup[0]])

            if dialog_ind == summ_ind:
                tp += 1
            else:
                all_errors.append((tup, (summ_ind, dialog_ind),
                                   chunk_2_summ_id[summ_ind].index(tup[0]),
                                   dialog_2_seg_id[dialog_ind].index(tup[1]),
                                   all_chunk[summ_ind][chunk_2_summ_id[summ_ind].index(tup[0])],
                                   all_seg_dialogs[dialog_ind][dialog_2_seg_id[dialog_ind].index(tup[1])]))
        print("Recall {}".format(tp/len(res)))

        pd.DataFrame.from_dict(chunk_2_dialog_match, orient='index').transpose().to_csv(
            os.path.join(result_dir, denoise_tag + 'training_bipartite_matching_results.csv'))

        with open(os.path.join(result_dir, denoise_tag + 'training_bipartite_matching.pkl'), 'wb') as f:
            pickle.dump(res, f)
    # ind = 0
    # evaluator.sm_similarity(all_errors[ind][-2], all_errors[ind][-1])
    # all_seg_dialogs[all_errors[ind][1][0]]
    # df_ref = pd.read_csv('reference.csv')
    # src_dialog = df_ref['dialogue']
    # src_dialog[all_errors[ind][1][0] ]
    # evaluator.sm_similarity(all_errors[0][-2], all_seg_dialogs[all_errors[0][1][0]][3])
    else:
        chunk_2_dialog_match = dict()
        for tup in res:
            dialog_ind = [key for key, val in dialog_2_seg_id.items() if tup[1] in val][0] + 1 #index in csv starts from 1
            if not (chunk_2_dialog_match.get(dialog_ind, None)):
                chunk_2_dialog_match[dialog_ind] = [all_chunk[tup[0]]]
            else:
                chunk_2_dialog_match[dialog_ind].append(all_chunk[tup[0]])

        pd.DataFrame.from_dict(chunk_2_dialog_match, orient='index').to_csv(
            os.path.join(result_dir, 'results', denoise_tag + 'test_bipartite_matching_results.csv'))

        with open(os.path.join(result_dir, denoise_tag + 'test_bipartite_matching.pkl'), 'wb') as f:
            pickle.dump(res, f)


def load_pre_computed_embeddings(denoise_tag, result_dir, train_data):
    if train_data:
        # key_tag_d = 'dialogue_train'
        # key_tag_chunk = 'chunks_train'
        with open(os.path.join(result_dir, denoise_tag + 'training_chunk_2_summ_id.pkl'), 'rb') as f:
            chunk_2_summ_id = pickle.load(f)
        with open(os.path.join(result_dir, denoise_tag + 'training_dialog_2_seg_id.pkl'), 'rb') as f:
            dialog_2_seg_id = pickle.load(f)

        with open(os.path.join(result_dir, denoise_tag + 'training_embds.pkl'), 'rb') as f:
            (all_embds_seg_dialog, all_embds_chunks) = pickle.load(f)

        with open(os.path.join(result_dir, denoise_tag + 'training_all_chunk.pkl'), 'rb') as f:
            all_chunk = pickle.load(f)

        with open(os.path.join(result_dir, denoise_tag + 'training_all_seg_dialogs.pkl'), 'rb') as f:
            all_seg_dialogs = pickle.load(f)


    else:
        key_tag_chunk = 'test_chunks'
        tag = 'test'

        with open(os.path.join(result_dir, denoise_tag + str(key_tag_chunk) + '.pkl'), 'rb') as f:
            all_embds_chunks = pickle.load(f)

        with open(os.path.join(result_dir, denoise_tag + tag + '_embds.pkl'), 'rb') as f:
            all_embds_seg_dialog = pickle.load(f)

        with open(os.path.join(result_dir, denoise_tag + tag + '_dialog_2_seg_id.pkl'), 'rb') as f:
            dialog_2_seg_id = pickle.load(f)

        with open(os.path.join(result_dir, denoise_tag + tag + '_all_seg_dialogs.pkl'), 'rb') as f:
            all_seg_dialogs = pickle.load(f)

        df_chunks = pd.read_csv('summary_pieces.csv')
        all_chunk = df_chunks['summary_piece'].to_list()
    return all_chunk, all_embds_chunks, all_embds_seg_dialog, all_seg_dialogs, chunk_2_summ_id, dialog_2_seg_id


def compute_embeddings(clean_noisy_segment, evaluator, result_dir, train_data):
    if train_data:
        df_ref = pd.read_csv('reference.csv')
        training_zs(df_ref, result_dir, evaluator, clean_noisy_segment=clean_noisy_segment)


    else:  # test data
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
        create_dialog_seg_embds(dialog_list, result_dir, evaluator, clean_noisy_segment=clean_noisy_segment)

        all_embds_chunks = embeddings_extract(chunk_list,
                                              key_tag=key_tag_chunk,
                                              result_dir=result_dir,
                                              evaluator=evaluator)
        return all_embds_chunks


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
"""
cosine_sim(*evaluator.encode_tokens(["This is an example sentence", "Each sentence is converted"]))
"""