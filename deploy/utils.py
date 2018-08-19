import sys

import re
import numpy as np
import requests
import time

from nltk.tokenize import sent_tokenize

RE_INT_MATCHER = re.compile('^\d+$')
RE_FLOAT_MATCHER = re.compile('^\d+\.\d+$')
ENTITY_STRING = 'ENTITY'

ncbi_url = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/RESTful/tmTool.cgi/BioConcept/"

def equalize_len(data, max_size):
    l = len(data)
    if l >= max_size:
        return data[:max_size]
    else:
        for _ in range(max_size - l):
            data.append(0)
        return data


def parse_flags(flag_file, flags):
    with open(flag_file, 'r') as f:
        for line in f:
            if line.startswith('Loading'):
                break
            k, v = line.split(':', 1)
            k, v = k.strip(), v.strip()

            int_match = RE_INT_MATCHER.match(v)
            float_match = RE_FLOAT_MATCHER.match(v)

            if k == 'null_label':
                v = v
            elif int_match:
                v = int(v)
            elif float_match:
                v = float(v)
            elif v == 'False':
                v = False
            elif v == 'True':
                v = True

            flags.__dict__[k] = v

    return flags


def parse_pubtator(pid, model, tokenize=None, str_id_maps=None, max_sent_len=200):
    url_submit = ncbi_url + str(pid) + "/PubTator/"
    try:
        urllib_result = requests.get(url_submit, timeout=1)
    except requests.exceptions.Timeout:
        print('Pubtator URL Not responding, exiting.......')
        sys.exit(1)

    result = urllib_result.text.strip().split('\n')
    title = result[0].split('|')[2]
    abst = result[1].split('|')[2]
    token_str_id_map = str_id_maps['token_str_id_map']
    sent_list = []
    sent_list.append(title)
    sent_list.extend(sent_tokenize(abst))
    orig_tokenized_sentences = [tokenize(sent) for sent in sent_list]
    unequal_tokens = [[token_str_id_map.get(token, 1) for token in sent] for sent in orig_tokenized_sentences]
    tokens = []
    for sent in unequal_tokens:
        tokens.append(equalize_len(sent, max_sent_len))
    tokens = np.array(tokens)

    seq_len = [len(s) for s in orig_tokenized_sentences]
    print(seq_len)

    ner_labels, entities, e, e_dist = make_example(result, seq_len, str_id_maps=str_id_maps, tokenize=tokenize,
                                                   max_sent_len=max_sent_len)

    final_e1 = []
    final_e2 = []
    final_e1_dist = []
    final_e2_dist = []
    final_tokens = []
    final_seq_len = []
    for sent_idx in range(len(e)):
        for e1_sent in e[sent_idx]:
            for e2_sent in e[sent_idx]:
                final_e1.append(e1_sent)
                final_e2.append(e2_sent)
                final_tokens.append(tokens[sent_idx])
                final_seq_len.append(seq_len[sent_idx])

        for e1_dist_sent in e_dist[sent_idx]:
            for e2_dist_sent in e_dist[sent_idx]:
                final_e1_dist.append(e1_dist_sent)
                final_e2_dist.append(e2_dist_sent)

    final_e1 = np.array(final_e1, dtype=np.int32)
    final_e2 = np.array(final_e2, dtype=np.int32)
    final_e1_dist = np.array(final_e1_dist, dtype=np.int32)
    final_e2_dist = np.array(final_e2_dist, dtype=np.int32)
    final_tokens = np.array(final_tokens, dtype=np.int32)
    final_seq_len = np.array(final_seq_len, dtype=np.int32)
    rel = np.zeros_like(final_e1)
    ep = np.zeros_like(final_e1)
    doc_ids = np.ones_like(final_e1) * pid

    ep_dist = np.full((final_e1_dist.shape[0], final_e1_dist.shape[1], final_e1_dist.shape[1]), -1e8)
    ep_indices = [(b, ei, ej) for b in range(final_e1_dist.shape[0])
                  for ei in np.where(final_e1_dist[b] == 1)[0]
                  for ej in np.where(final_e2_dist[b] == 1)[0]]
    b, r, c = zip(*ep_indices)
    ep_dist[b, r, c] = 0.0

    pos_encode = [range(1, final_tokens.shape[1] + 1) for i in range(final_tokens.shape[0])]
    label_batch = rel
    ex_loss = np.ones_like(final_e1)

    feed_dict = {model.text_batch: final_tokens,
                 model.e1_dist_batch: final_e1_dist,
                 model.e2_dist_batch: final_e2_dist,
                 model.seq_len_batch: final_seq_len,
                 model.label_batch: label_batch,
                 model.pos_encode_batch: pos_encode,
                 model.ep_batch: ep,
                 model.kb_batch: rel,
                 model.e1_batch: final_e1,
                 model.e2_batch: final_e2,
                 model.example_loss_weights: ex_loss,
                 model.ep_dist_batch: ep_dist}

    return feed_dict, final_e1.shape[0], doc_ids


def make_example(text_list, seq_len, str_id_maps=None, tokenize=None, max_sent_len=200):
    line_num = 0

    line = text_list[line_num].strip()
    pub_id, _, title_list = line.split('|', 2)
    line_num += 1
    title = ''.join(title_list)
    abstract = ''.join(text_list[line_num].strip().split('|')[2:])
    line_num += 1
    current_pub = '%s %s' % (title, abstract)
    label_annotations = {}
    current_annotations = []
    ner_label_str_id_map = str_id_maps['ner_label_str_id_map']
    entity_str_id_map = str_id_maps['entity_str_id_map']

    line = text_list[line_num].strip()
    line_num += 1
    while line_num <= len(text_list) and line:

        parts = line.split('\t')
        if len(parts) == 4:
            pub_id, rel, e1, e2 = parts
            label_annotations[(e1, e2, pub_id)] = rel
        elif len(parts) == 6:
            pub_id, start, end, mention, label, kg_ids = parts
            if label in ['Chemical', 'Disease']:
                for kg_id in kg_ids.split('|'):
                    current_annotations.append((pub_id, int(start), int(end), mention, label, kg_id))
        elif len(parts) == 7:
            pub_id, start, end, mention, label, kg_ids, split_mentions = parts
            if label in ['Chemical', 'Disease']:
                for kg_id in kg_ids.split('|'):
                    current_annotations.append((pub_id, int(start), int(end), mention, label, kg_id))

        if line_num < len(text_list):
            line = text_list[line_num].strip()
        line_num += 1

    # do something with last annotations
    sorted_annotations = sorted(current_annotations, key=lambda tup: tup[1])
    replaced_text = []
    last = 0
    annotation_map = {}
    for i, (pub_id, start, end, mention, label, kg_id) in enumerate(sorted_annotations):
        mention = current_pub[start:end]
        dummy_token = '%s%d_' % (ENTITY_STRING, i)
        replaced_text.append(' %s %s ' % (current_pub[last:start].strip(), dummy_token))
        last = end
        annotation_map[dummy_token] = (mention, label, kg_id)

    # add text that occurs after the last entity
    replaced_text.append(current_pub[end:])
    abstract = ''.join(replaced_text).replace('  ', ' ')
    sentences = sent_tokenize(abstract)
    for sent in sentences:
        print(len(sent))
    tokenized_sentences = [[w for w in tokenize(s)] for s in sentences]

    out_sentence = [[tokenize(annotation_map[token][0]) if token.startswith(ENTITY_STRING) else [token]
                     for token in sentence]
                    for sentence in tokenized_sentences]

    # get the token offsets in the sentence for each entity
    token_lens = [[len(token) if type(token) is list else 1 for token in sentence]
                  for sentence in out_sentence]
    token_offsets = [np.cumsum(lens) for lens in token_lens]

    entity_offsets = [[(annotation_map[token], offset[i] - length[i], offset[i])
                       for (i, token) in enumerate(sentence) if token.startswith(ENTITY_STRING)]
                      for length, offset, sentence
                      in zip(token_lens, token_offsets, tokenized_sentences)]
    print(entity_offsets)
    print(len(entity_offsets))

    num_sentences = len(entity_offsets)
    ner_labels = np.zeros((num_sentences, max_sent_len))
    entities = np.zeros((num_sentences, max_sent_len))
    e1_abst = []
    e1_dist_abst = []

    for sent_idx, sent_offsets in enumerate(entity_offsets):
        print(sent_idx)
        sent_len = seq_len[sent_idx]
        token_idx = 0
        prev_end = 0
        e1_sent = []
        e1_dist_sent = []

        if len(sent_offsets) == 0:
            ner_labels[sent_idx, :sent_len] = ner_label_str_id_map['O']
            continue

        for (mention, label, ent_str), start, end in sent_offsets:
            e1_dist = np.zeros(max_sent_len, dtype=np.int32)
            e1_dist[start:end] = 1
            e1_dist_sent.append(e1_dist)

            for _ in range(prev_end, start):
                ner_labels[sent_idx, token_idx] = ner_label_str_id_map['O']
                entities[sent_idx, token_idx] = 0
                token_idx += 1

            ner_labels[sent_idx, token_idx] = ner_label_str_id_map['B-' + label]

            if ent_str not in entity_str_id_map:
                mesh_str = 'MESH:' + ent_str
                chebi_str = 'CHEBI:' + ent_str
                if mesh_str in entity_str_id_map:
                    ent_id = entity_str_id_map.get(mesh_str)
                if chebi_str in entity_str_id_map:
                    ent_id = entity_str_id_map.get(chebi_str)
            else:
                ent_id = entity_str_id_map.get(ent_str)

            entities[sent_idx, token_idx] = ent_id
            e1_sent.append(ent_id)

            token_idx += 1
            for _ in range(start + 1, end):
                ner_labels[sent_idx, token_idx] = ner_label_str_id_map['I-' + label]
                token_idx += 1
            prev_end = end

        for _ in range(end, sent_len):
            ner_labels[sent_idx, token_idx] = ner_label_str_id_map['O']
            token_idx += 1

        e1_abst.append(e1_sent)
        e1_dist_abst.append(e1_dist_sent)

    return ner_labels, entities, e1_abst, e1_dist_abst


def export_predictions(sess, model, FLAGS, pid, string_int_maps, tokenize=None, threshold_map=None, max_sent_len=500):
    print('Evaluating')
    null_label_set = set([int(l) for l in FLAGS.null_label.split(',')])

    result_list = [model.probs, model.label_batch, model.e1_batch, model.e2_batch]

    feed_dict, batch_size, doc_ids = parse_pubtator(pid, model, tokenize=tokenize, str_id_maps=string_int_maps,
                                                    max_sent_len=max_sent_len)

    probs, labels, e1, e2 = sess.run(result_list, feed_dict=feed_dict)
    labeled_scores = [(l, np.argmax(s), np.max(s), _e1, _e2, did)
                      for s, l, _e1, _e2, did in zip(probs, labels, e1, e2, doc_ids)]

    final_out = []
    for label_id in range(FLAGS.num_classes):
        if label_id not in null_label_set:
            label_str = string_int_maps['kb_id_str_map'][label_id]
            threshold = threshold_map[label_id] if threshold_map \
                else [float(t) for t in FLAGS.thresholds.split(',')]

            # label, prediction, confidence
            predictions = [(_e1, _e2, did) for label, pred, conf, _e1, _e2, did in labeled_scores
                           if conf >= threshold and pred == label_id and _e1 != _e2]
            mapped_predictions = [(string_int_maps['entity_id_str_map'][_e1],
                                   string_int_maps['entity_id_str_map'][_e2], did)
                                  for _e1, _e2, did in predictions]
            out_lines = ['%s\t%s\tArg1:%s\tArg2:%s\n' % (did, label_str, _e1, _e2)
                         for _e1, _e2, did in mapped_predictions]
            final_out.extend(out_lines)

    return set(final_out)
