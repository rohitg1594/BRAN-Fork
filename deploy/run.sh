#!/usr/bin/env bash
pid=$1
word_piece_vocab=${CDR_IE_ROOT}/deploy/bpe.vocab
processed_dir=${CDR_IE_ROOT}/deploy/processed
protos_dir=${CDR_IE_ROOT}/deploy/processed

mkdir -p ${processed_dir}
mkdir -p ${protos_dir}

python ${CDR_IE_ROOT}/deploy/download_pubtator.py --pid $pid --out_file ${CDR_IE_ROOT}/deploy/test.tsv

python ${CDR_IE_ROOT}/src/processing/utils/process_CDR_new.py --input_file ${CDR_IE_ROOT}/deploy/test.tsv --output_dir ${processed_dir} --output_file_suffix processed.txt --max_seq 2000000 --full_abstract True --encoding utf-8 --export_all_eps True --word_piece_codes ${word_piece_vocab}

python ${CDR_IE_ROOT}/src/processing/labled_tsv_to_tfrecords_single_sentences.py --text_in_files ${CDR_IE_ROOT}/deploy/*processed.txt --out_dir ${protos_dir} --max_len 2000000 --num_threads 10 --multiple_mentions --tsv_format --min_count 0 --sentence_window 0


# sentence segment NER
python ${CDR_IE_ROOT}/src/processing/utils/sentence_segment_conll.py -i ${processed_dir}/ner_processed.txt -o ${processed_dir}/ner_processed_sentence.txt -e  utf-8

# convert ner data to tf protos
python ${CDR_IE_ROOT}/src/processing/ner_to_tfrecords.py --in_files ${processed_dir}/ner_processed_sentence.txt --out_dir ${proto_dir} --load_vocab ${proto_dir} --num_threads 5
