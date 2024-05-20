import argparse
import json


def convert_for_errant(predict_data):
    """
    converts output files for ERRANT evaluation (akces, cowsl2h, falko)
    """
    source = open('errant/source.txt', 'w', encoding='utf-8')
    target = open('errant/target.txt', 'w', encoding='utf-8')
    predict = open('errant/predict.txt', 'w', encoding='utf-8')
    for i in predict_data:
        source.write(i['Wrong'].replace(' ##', '') + '\n')
        target.write(i['Target'].replace(' ##', '') + '\n')
        predict.write(i['Rewrite'].replace(' ##', '') + '\n')


def convert_for_cherrant(predict_data):
    """
    converts output files for ChERRANT evaluation (mucgec)
    """
    hyp = open('cherrant/hyp.txt', 'w', encoding='utf-8')

    for i, data in enumerate(predict_data):
        source = ''.join(data['Wrong'].strip().split())
        # target = ''.join(data['Target'].strip().split())
        predict = ''.join(data['Rewrite'].strip().split())
        hyp.write(str(i + 1) + '\t' + source + '\t' + predict + '\n')
        # ref.write(str(i + 1) + '\t' + source + '\t' + target + '\n')


def convert_for_m2scorer(predict_data, language):
    """
    converts output files for m2scorer (conll14, nlpcc18, fce)
    """
    with open('m2scorer/predict.txt', 'w', encoding='utf-8') as f:
        for data in predict_data:
            if language == 'English':
                f.write(data['Rewrite'].replace(' ##', '') + '\n')
            else:
                f.write(''.join(data['Rewrite'].strip().split()) + '\n')


def main(args):
    with open(args.predict_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if args.dataset in ['akces', 'cowsl2h', 'falko']:
        convert_for_errant(data)
    elif args.dataset in ['mucgec']:
        convert_for_cherrant(data)
    elif args.dataset in ['conll14', 'fce', 'nlpcc18']:
        if args.dataset == 'nlpcc18':
            convert_for_m2scorer(data, 'Chinese')
        else:
            convert_for_m2scorer(data, 'English')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_file', type=str,
                        help='path to predict file')
    parser.add_argument('--dataset', type=str,
                        choices=['akces', 'conll14', 'cowsl2h', 'falko', 'fce', 'nlpcc18', 'mucgec'],
                        help='specify your dataset')
    args = parser.parse_args()
    main(args)