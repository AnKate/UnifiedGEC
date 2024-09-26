# -*- encoding: utf-8 -*-
# @Author: Yunshi Lan
# @Time: 2023/01/30 11:06
# @File: evaluator.py


import os
import errant
import spacy

from collections import Counter
from gectoolkit.evaluate.cherrant import parallel_to_m2, compare_m2_for_evaluation
from gectoolkit.utils.enum_type import SpecialTokens
from gectoolkit.utils.compare_m2 import simplify_edits, process_edits, evaluate_edits, computeFScore


def save_to_txt_file(text, filename):
    """
    write text to file
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)


def get_errant_edits(src, tgt):
    """
    generate m2 edits
    """
    nlp = spacy.load("en_core_web_sm")
    annotator = errant.load("en", nlp)

    ret = ""
    orig = annotator.parse(src)
    ret += " ".join(["S"] + [token.text for token in orig]) + '\n'
    cor = annotator.parse(tgt)
    edits = annotator.annotate(orig, cor)
    if len(edits):
        for e in edits:
            ret += e.to_m2() + '\n'
    else:
        ret += "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n"
    return ret


class GECEvaluator(object):
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.tmp_save_path = './gectoolkit/evaluate/cherrant/samples'
        if not os.path.exists(self.tmp_save_path):
            os.mkdir(self.tmp_save_path)

        self.language = config["language"]
        self.model = config["model"]

        # ['<-PAD->', '<-CLS->', '<-SEP->', '<-MASK->']
        special_tokens = [SpecialTokens.PAD_TOKEN, SpecialTokens.CLS_TOKEN, SpecialTokens.SEP_TOKEN, SpecialTokens.MASK_TOKEN]
        # data_special_ids ：[16543, 16539, 16544, 16540]  pred_special_ids：[16543]
        if self.model != 'GECToR':
            self.data_special_ids = [tokenizer.convert_tokens_to_ids(w) for w in special_tokens]
            self.pred_special_ids = [tokenizer.convert_tokens_to_ids(w) for w in [SpecialTokens.PAD_TOKEN]]

    def measure(self, sources, labels, predicts):
        """
        gec的评测
        """

        if self.model != 'GECToR':
            sources = self.tokenizer.decode(sources)
            labels = self.tokenizer.decode(labels)
            predicts = self.tokenizer.decode([int(i) for i in predicts], skip_special_tokens=True)

        if self.language == "zh":
            if self.model == 'GECToR':
                sources = ''.join(sources)
                labels = ''.join(labels)
                predicts = ''.join(predicts)

            source_predict = sources + '\t' + predicts
            source_target = sources + '\t' + labels

            hyp = os.path.join(self.tmp_save_path, "hyp.para")
            save_to_txt_file(source_predict, hyp)
            ref = os.path.join(self.tmp_save_path, "ref.para")
            save_to_txt_file(source_target, ref)

            hyp_m2_char = os.path.join(self.tmp_save_path, "hyp.m2.char")
            ref_m2_char = os.path.join(self.tmp_save_path, "ref.m2.char")

            p2m_hyp_args = parallel_to_m2.Args(file=hyp, output=hyp_m2_char)
            parallel_to_m2.main(p2m_hyp_args)

            p2m_ref_args = parallel_to_m2.Args(file=ref, output=ref_m2_char)
            parallel_to_m2.main(p2m_ref_args)

            compare_args = compare_m2_for_evaluation.Args(hyp=hyp_m2_char, ref=ref_m2_char)

            TP, FP, FN, Prec, Rec, F = compare_m2_for_evaluation.main(compare_args)

            gec_evaluate_dict = {
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "Prec": Prec,
                "Rec": Rec,
                "F0.5": F
            }
            return Prec, Rec, F
        else:
            if self.model == 'GECToR':
                sources = ' '.join(sources)
                labels = ' '.join(labels)
                predicts = ' '.join(predicts)

            ref = get_errant_edits(sources, labels).strip()
            hyp = get_errant_edits(sources, predicts).strip()

            best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
            hyp_edits = simplify_edits(hyp)
            ref_edits = simplify_edits(ref)
            hyp_dict = process_edits(hyp_edits)
            ref_dict = process_edits(ref_edits)

            original_sentence = hyp[2:].split("\nA")[0]
            count_dict, _ = evaluate_edits(hyp_dict, ref_dict, best_dict, 0, original_sentence)
            best_dict += Counter(count_dict)

            p, r, f = computeFScore(best_dict["tp"], best_dict["fp"], best_dict["fn"])
            return p, r, f
