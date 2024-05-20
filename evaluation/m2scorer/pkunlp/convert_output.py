#!/usr/bin/python
# encoding: utf-8

from __future__ import unicode_literals, print_function

from pkunlp import Segmentor, NERTagger, POSTagger
import argparse


def segment(input_file, output_file):
    segmentor = Segmentor("./pkunlp/feature/segment.feat", "./pkunlp/feature/segment.dic")
    output = open(output_file, 'w', encoding='utf-8')

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            segments = segmentor.seg_string(line)
            print(segments)
            for seg in segments:
                if seg != '\ufeff':
                    output.write(seg)
                    if seg != '\n':
                        output.write(' ')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        help='Path to the input file',
                        required=True)
    parser.add_argument('--output_file',
                        help='Path to the output file',
                        required=True)
    args = parser.parse_args()
    segment(args.input_file, args.output_file)
