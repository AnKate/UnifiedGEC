import gectoolkit.model.GECToR.segment.tokenization as tokenization
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(dir_path, "vocab.txt"), do_lower_case=False)


def segment(lines):
    ret = []
    for line in lines:
        line = line.strip()
        line = line.replace(" ", "")
        line = tokenization.convert_to_unicode(line)
        if not line:
            continue
        tokens = tokenizer.tokenize(line)
        ret.append(tokens)
    return ret
