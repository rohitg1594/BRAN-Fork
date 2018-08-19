import random
import gzip
from os.path import join

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
parser.add_argument('--bran_dir', type=str, help='path to bran')
parser.add_argument('--max_sent_len', type=int, default=100, help='Maximum number of tokens in a sentence.')
parser.add_argument('--pid', type=int, help='PUBMED ID to predict on.')

args = parser.parse_args()
wpt = WordPieceTokenizer(join(args.bran_dir, 'deploy/bpe.vocab'), entity_str=ENTITY_STRING)
tokenize = wpt.tokenize

FLAGS = parse_flags(join(args.bran_dir, 'deploy/gnltw_flags.txt'), FLAGS)

if ('transformer' in FLAGS.text_encoder or 'glu' in FLAGS.text_encoder) and FLAGS.token_dim == 0:
    FLAGS.token_dim = FLAGS.embed_dim-(2*FLAGS.position_dim)

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
if FLAGS.kg_label_file != '':
    kg_in_file = gzip.open(FLAGS.kg_label_file, 'rb') if FLAGS.kg_label_file.endswith('gz')\
        else open(FLAGS.kg_label_file, 'r')
    lines = [l.strip().split() for l in kg_in_file.readlines()]
    eps = [('%s::%s' % (l[0], l[1]), l[2]) for l in lines]
    ep_kg_labels = defaultdict(set)
    [ep_kg_labels[ep_str_id_map[_ep]].add(pid) for _ep, pid in eps if _ep in ep_str_id_map]

    kg_in_file.close()

e1_e2_ep_map = {}
ep_e1_e2_map = {}

# get entity <-> type maps for sampling negatives
entity_type_map, type_entity_map = {}, defaultdict(list)
if FLAGS.type_file != '':
    with open(FLAGS.type_file, 'r') as f:
        entity_type_map = {entity_str_id_map[l.split('\t')[0]]: l.split('\t')[1].strip().split(',') for l in
                           f.readlines() if l.split('\t')[0] in entity_str_id_map}
        for entity, type_list in entity_type_map.iteritems():
            for t in type_list:
                type_entity_map[t].append(entity)
        # filter
        type_entity_map = {k: v for k, v in type_entity_map.iteritems() if len(v) > 1}
        valid_types = set([t for t in type_entity_map.iterkeys()])
        entity_type_map = {k: [t for t in v if t in valid_types] for k, v in entity_type_map.iteritems()}
        entity_type_map = {k: v for k, v in entity_type_map.iteritems() if len(v) > 1}

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
        reader = tf.train.NewCheckpointReader(join(args.bran_dir, FLAGS.load_model))
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
            saver.restore(sess, join(args.bran_dir, FLAGS.load_model))

        threads = tf.train.start_queue_runners(sess=sess)
        fb15k_eval = None
        tac_eval = None
        out_file = ''
        predictions = export_predictions(sess, model, FLAGS, args.pid, string_int_maps, threshold_map=THRESHOLD_MAP,
                                         max_sent_len=args.max_sent_len, tokenize=tokenize)
        print('Done')

if len(predictions) == 0:
    print("No relations found.")
else:
    for prediction in predictions:
        print(prediction)
