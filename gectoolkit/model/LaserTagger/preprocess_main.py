# coding=utf-8

# Lint as: python3
# """Convert a dataset into the TFRecord format.
# The resulting TFRecord file will be used when training a LaserTagger model.
# """

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

from typing import Text

from gectoolkit.model.LaserTagger.utils import utils
from gectoolkit.model.LaserTagger import tagging_converter, bert_example


input_file = '../../../dataset/sighan15/trainset.json'
input_format = 'wikisplit'
output_record = 'tmp_output/train.record'
label_map_file = 'tmp_output/label_map.txt'
vocab_file = '../../properties/model/LaserTagger/vocab.txt'
max_seq_length = 40
do_lower_case = True
enable_swap_tag = False
output_arbitrary_targets_for_infeasible_examples = True


def _write_example_count(count: int) -> Text:
    count_fname = output_record + '.num_examples.txt'
    # with tf.io.gfile.GFile(count_fname, 'w') as count_writer:
    with open(count_fname, 'w') as count_writer:
        count_writer.write(str(count))
    return count_fname


def main():
    label_map = utils.read_label_map(label_map_file)
    converter = tagging_converter.TaggingConverter(
        tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
        enable_swap_tag)
    print('--------------------')
    print(converter)

    builder = bert_example.BertExampleBuilder(label_map, vocab_file,
                                              max_seq_length,
                                              do_lower_case, converter)

    num_converted = 0
    # with tf.io.TFRecordWriter(output_tfrecord) as writer:
    with open(output_record, 'w') as writer:
        for i, (sources, target) in enumerate(utils.yield_sources_and_targets(
                input_file, input_format)):
            print("sources:", sources)
            print("target:", target)
            # logging.log_every_n(
            #     logging.INFO,
            #     f'{i} examples processed, {num_converted} converted to tf.Example.',
            #     10000)
            example = builder.build_bert_example(
                sources, target,
                output_arbitrary_targets_for_infeasible_examples)
            if example is None:
                continue
            print('----------------------------')
            print(example.to_torch_example())
            writer.write(example.to_torch_example())

            num_converted += 1
    # logging.info(f'Done. {num_converted} examples converted to tf.Example.')
    count_fname = _write_example_count(num_converted)
    # logging.info(f'Wrote:\n{output_record}\n{count_fname}')


if __name__ == '__main__':
    main()

# class Input():
#     def getInput(self):
#
#         label_map = utils.read_label_map(label_map_file)
#         converter = tagging_converter.TaggingConverter(
#             tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
#             enable_swap_tag)
#         print('--------------------')
#         print(converter)
#
#         builder = bert_example.BertExampleBuilder(label_map, vocab_file,
#                                                   max_seq_length,
#                                                   do_lower_case, converter)
#
#         num_converted = 0
#         # with tf.io.TFRecordWriter(output_tfrecord) as writer:
#         with open(output_record, 'w') as writer:
#             for i, (sources, target) in enumerate(utils.yield_sources_and_targets(
#                     input_file, input_format)):
#                 # logging.log_every_n(
#                 #     logging.INFO,
#                 #     f'{i} examples processed, {num_converted} converted to tf.Example.',
#                 #     10000)
#                 example = builder.build_bert_example(
#                     sources, target,
#                     output_arbitrary_targets_for_infeasible_examples)
#                 if example is None:
#                     continue
#                 print('----------------------------')
#                 print(example.to_torch_example())
#                 writer.write(example.to_torch_example())
#
#                 num_converted += 1
#         # logging.info(f'Done. {num_converted} examples converted to tf.Example.')
#         count_fname = _write_example_count(num_converted)
#         # logging.info(f'Wrote:\n{output_record}\n{count_fname}')
#         return example.to_torch_example()
