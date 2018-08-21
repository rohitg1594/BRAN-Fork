import random
import gzip
from os.path import join
import os

import argparse

from src.data_utils import *
from src.models.classifier_models import *
from src.feed_dicts import *
from src.processing.utils.word_piece_tokenizer import WordPieceTokenizer

from utils import export_predictions, parse_flags

tf.logging.set_verbosity('ERROR')

FLAGS = tf.app.flags.FLAGS
ENTITY_STRING = 'ENTITY'

THRESHOLD_MAP = {1: 0.3,
                 2: 0.1,
                 3: 0.1,
                 4: 0.5,
                 5: 0.3,
                 6: 0.1,
                 7: 0.1}

parser = argparse.ArgumentParser(description='Script to use bran with PUBMEDID.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--pid', type=int, help='PUBMED ID to predict on.')
parser.add_argument('-o', '--output', type=str, help='Output file for predictions.')

bran_dir = os.getenv("BRAN_DIR")
args = parser.parse_args()
wpt = WordPieceTokenizer(join(bran_dir, 'deploy/bpe.vocab'), entity_str=ENTITY_STRING)
tokenize = wpt.tokenize

FLAGS = parse_flags(bran_dir, FLAGS)
if not args.output:
    args.output = '{}.tsv'.format(args.pid)

if ('transformer' in FLAGS.text_encoder or 'glu' in FLAGS.text_encoder) and FLAGS.token_dim == 0:
    FLAGS.token_dim = FLAGS.embed_dim-(2*FLAGS.position_dim)


def predict(pid):
    # read in str <-> int vocab maps
    with open(FLAGS.vocab_dir + '/rel.txt', 'r') as f:
        kb_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        kb_id_str_map = {i: s for s, i in kb_str_id_map.iteritems()}
        kb_vocab_size = FLAGS.kb_vocab_size
    with open(FLAGS.vocab_dir + '/token.txt', 'r') as f:
        token_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        if FLAGS.start_end:
            if '<START>' not in token_str_id_map:
                token_str_id_map['<START>'] = len(token_str_id_map)
            if '<END>' not in token_str_id_map:
                token_str_id_map['<END>'] = len(token_str_id_map)
        token_id_str_map = {i: s for s, i in token_str_id_map.iteritems()}
        token_vocab_size = len(token_id_str_map)

    with open(FLAGS.vocab_dir + '/entities.txt', 'r') as f:
        entity_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        entity_id_str_map = {i: s for s, i in entity_str_id_map.iteritems()}
        entity_vocab_size = len(entity_id_str_map)
    with open(FLAGS.vocab_dir + '/ep.txt', 'r') as f:
        ep_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
        ep_id_str_map = {i: s for s, i in ep_str_id_map.iteritems()}
        ep_vocab_size = len(ep_id_str_map)

    if FLAGS.ner_train != '':
        with open(FLAGS.vocab_dir + '/ner_labels.txt', 'r') as f:
            ner_label_str_id_map = {l.split('\t')[0]: int(l.split('\t')[1].strip()) for l in f.readlines()}
            if FLAGS.start_end:
                if '<START>' not in ner_label_str_id_map:
                    ner_label_str_id_map['<START>'] = len(ner_label_str_id_map)
                if '<END>' not in ner_label_str_id_map:
                    ner_label_str_id_map['<END>'] = len(ner_label_str_id_map)
            ner_label_id_str_map = {i: s for s, i in ner_label_str_id_map.iteritems()}
            ner_label_vocab_size = len(ner_label_id_str_map)
    else:
        ner_label_id_str_map = {}
        ner_label_str_id_map = {}
        ner_label_vocab_size = 1
    position_vocab_size = (2 * FLAGS.max_seq)

    label_weights = None
    if FLAGS.label_weights != '':
        with open(FLAGS.label_weights, 'r') as f:
            lines = [l.strip().split('\t') for l in f]
            label_weights = {kb_str_id_map[k]: float(v) for k, v in lines}

    ep_kg_labels = None

    e1_e2_ep_map = {}
    ep_e1_e2_map = {}

    string_int_maps = {'kb_str_id_map': kb_str_id_map, 'kb_id_str_map': kb_id_str_map,
                       'token_str_id_map': token_str_id_map, 'token_id_str_map': token_id_str_map,
                       'entity_str_id_map': entity_str_id_map, 'entity_id_str_map': entity_id_str_map,
                       'ep_str_id_map': ep_str_id_map, 'ep_id_str_map': ep_id_str_map,
                       'ner_label_str_id_map': ner_label_str_id_map, 'ner_label_id_str_map': ner_label_id_str_map,
                       'e1_e2_ep_map': e1_e2_ep_map, 'ep_e1_e2_map': ep_e1_e2_map, 'ep_kg_labels': ep_kg_labels,
                       'label_weights': label_weights}

    word_embedding_matrix = load_pretrained_embeddings(token_str_id_map, FLAGS.embeddings, int(FLAGS.token_dim),
                                                       token_vocab_size)
    entity_embedding_matrix = load_pretrained_embeddings(entity_str_id_map, FLAGS.entity_embeddings, int(FLAGS.embed_dim),
                                                         entity_vocab_size)

    with tf.Graph().as_default():
        tf.set_random_seed(int(FLAGS.random_seed))
        np.random.seed(int(FLAGS.random_seed))
        random.seed(int(FLAGS.random_seed))

        # initialize model
        if 'multi' in FLAGS.model_type and 'label' in FLAGS.model_type:
            model_type = MultiLabelClassifier
        elif 'entity' in FLAGS.model_type and 'binary' in FLAGS.model_type:
            model_type = EntityBinary
        else:
            model_type = ClassifierModel
        model = model_type(ep_vocab_size, entity_vocab_size, kb_vocab_size, token_vocab_size, position_vocab_size,
                           ner_label_vocab_size, word_embedding_matrix, entity_embedding_matrix, string_int_maps, FLAGS)

        if FLAGS.load_model != '':
            reader = tf.train.NewCheckpointReader(join(bran_dir, FLAGS.load_model))
            cp_list = set([key for key in reader.get_variable_to_shape_map()])
            # if variable does not exist in checkpoint or sizes do not match, dont load
            r_vars = [k for k in tf.global_variables() if k.name.split(':')[0] in cp_list
                      and k.get_shape() == reader.get_variable_to_shape_map()[k.name.split(':')[0]]]
            saver = tf.train.Saver(var_list=r_vars)
        else:
            saver = tf.train.Saver()
        sv = tf.train.Supervisor(logdir=FLAGS.logdir if FLAGS.save_model != '' else None,
                                 global_step=model.global_step,
                                 saver=None,
                                 save_summaries_secs=0,
                                 save_model_secs=0, )

        with sv.managed_session(FLAGS.master,
                                config=tf.ConfigProto(
                                    # log_device_placement=True,
                                    allow_soft_placement=True
                                )) as sess:
            if FLAGS.load_model != '':
                print("Deserializing model: %s" % FLAGS.load_model)
                saver.restore(sess, join(bran_dir, FLAGS.load_model))

            predictions, ent_type_map = export_predictions(sess, model, FLAGS, pid, string_int_maps,
                                             threshold_map=THRESHOLD_MAP, tokenize=tokenize)
            print('Done')

    predictions_dict = {}
    for i, prediction in enumerate(predictions):
        prediction = prediction.strip()
        parts = prediction.split('\t')
        theme = parts[1]

        entity_1 = parts[2].split(':')[2]
        print(entity_1)
        for k in ent_type_map.keys():
            if entity_1 in k:
                type_1 = ent_type_map[k]
                break

        entity_2 = parts[2].split(':')[2]
        print(entity_2)
        for k in ent_type_map.keys():
            if entity_2 in k:
                type_2 = ent_type_map[k]
                break

        predictions_dict['K{}'.format(i)] = {'theme': theme,
                                             'entities': [entity_1, entity_2],
                                             'entity_types': [type_1, type_2]}

    return predictions, predictions_dict


if __name__ == '__main__':
    predictions, predictions_dict = predict(args.pid)

    print(predictions_dict)

    if len(predictions) == 0:
        print("No relations found.")
    else:
        with open(args.output, 'w') as f:
            for prediction in predictions:
                f.write(prediction)
                print(prediction)

