#!/usr/bin/env bash

word_piece_vocab=${CDR_IE_ROOT}/data/cdr/word_piece_vocabs/just_train_2500/word_pieces.txt
input_dir=${CDR_IE_ROOT}/data/cdr
processed_dir=${input_dir}/processed/just_train_2500
proto_dir=${processed_dir}/protos
max_len=500000
# replace infrequent tokens with <UNK>
min_count=5


# process train, dev, and test data
mkdir -p ${processed_dir}
echo "Processing Training data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/CDR_TrainingSet.PubTator.txt.gz --output_dir ${processed_dir} --output_file_suffix CDR_train.txt --max_seq ${max_len} --full_abstract True --word_piece_codes ${word_piece_vocab}

echo "Processing Dev data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/CDR_DevelopmentSet.PubTator.txt.gz --output_dir ${processed_dir} --output_file_suffix CDR_dev.txt --max_seq ${max_len} --full_abstract True --word_piece_codes ${word_piece_vocab}

echo "Processing Test data"
python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_data.py --input_file ${input_dir}/CDR_TestSet.PubTator.txt.gz --output_dir ${processed_dir} --output_file_suffix CDR_test.txt --max_seq ${max_len} --full_abstract True --word_piece_codes ${word_piece_vocab}

for f in CDR_dev CDR_train CDR_test; do
python ${CDR_IE_ROOT}/src/processing/utils/filter_hypernyms.py -p ${processed_dir}/positive_0_${f}.txt -n ${processed_dir}/negative_0_${f}.txt -m ${CDR_IE_ROOT}/data/2017MeshTree.txt -o ${processed_dir}/negative_0_${f}_filtered.txt
done


# convert processed data to tensorflow protobufs
python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords.py --text_in_files ${processed_dir}/\*tive_\*CDR\* --out_dir ${proto_dir} --max_len ${max_len} --num_threads 10 --multiple_mentions --tsv_format --min_count ${min_count}

# rename dev files to match train regex
mv ${proto_dir}/negative_0_CDR_dev_filtered.txt.proto ${proto_dir}/negative_0_CDR_train_dev_filtered.txt.proto
mv ${proto_dir}/positive_0_CDR_dev.txt.proto ${proto_dir}/positive_0_CDR_train_dev.txt.proto

# convert ner data to tf protos
python ${CDR_IE_ROOT}/src/processing/ner_to_tfrecords.py --in_files ${processed_dir}/ner_\* --out_dir ${proto_dir} --load_vocab ${proto_dir} --num_threads 5
