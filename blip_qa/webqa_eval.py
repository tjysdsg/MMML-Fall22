import numpy as np
from word2number import w2n
from bart_score import BARTScorer
import string, re
from collections import Counter
import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat", "parser"])
np.set_printoptions(precision=4)

color_set = {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite',
             'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey',
             'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet',
             'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross',
             'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube',
             'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot',
             'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape',
             'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight',
             'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical',
             'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
yesno_set = {'yes', 'no'}


def detectNum(l):
    result = []
    for w in l:
        try:
            result.append(str(int(w)))
        except:
            pass
    return result


def toNum(word):
    if word == 'point': return word
    try:
        return w2n.word_to_num(word)
    except:
        return word


def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):  # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1)  # remove '.' if it's not a decimal point

    def lower(text):
        return text.lower()

    def lemmatization(text):
        return " ".join([token.lemma_ for token in nlp(text)])

    if len(s.strip()) == 1:
        # accept article and punc if input is a single char
        return white_space_fix(lower(s))
    elif len(s.strip().split()) == 1:
        # accept article if input is a single word
        return lemmatization(white_space_fix(remove_punc(lower(s))))

    return lemmatization(white_space_fix(remove_articles(remove_punc(lower(s)))))


def _webqa_acc_approx(predction, ground_truth, domain=None):
    """VQA Eval (SQuAD style EM, F1)"""
    bow_pred = normalize_text(predction).split()
    bow_target = normalize_text(ground_truth).split()
    if domain == {"NUMBER"}:
        bow_pred = detectNum(bow_pred)
        bow_target = detectNum(bow_target)
    elif domain is not None:
        bow_pred = list(domain.intersection(bow_pred))
        bow_target = list(domain.intersection(bow_target))
    else:
        # TODO: fine-grained evaluation (e.g., content words) for text question types
        bow_pred = bow_pred
        bow_target = bow_target

    common = Counter(bow_target) & Counter(bow_pred)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = num_same / len(bow_pred)
    recall = num_same / len(bow_target)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, recall, precision


def webqa_acc_approx(prediction, ground_truth, Qcate="text"):
    f1, recall, precision = _webqa_acc_approx(
        prediction,
        ground_truth,
        domain={
            "color": color_set,
            "shape": shape_set,
            "YesNo": yesno_set,
            "number": {"NUMBER"},
            "text": None,
            "Others": None,
            "choose": None,
        }[Qcate],
    )
    if Qcate in ["color", "shape", "number", "YesNo"]:
        acc = f1
    else:
        acc = recall
    return {"acc_approx": acc}


TABLE = str.maketrans(dict.fromkeys(string.punctuation))


def normalize_text_for_bart(x):  # Light text normalization for WebQA eval: white space fix + punctuation removal
    return " ".join(x.translate(TABLE).split())


def compute_bartscore_ParaBank(c, a, model, switch=False):
    c_removepunc = [normalize_text_for_bart(x) for x in c]
    a_removepunc = [normalize_text_for_bart(x) for x in a]
    if switch:
        score = np.exp(model.score(c_removepunc, a_removepunc))
    else:
        score = np.exp(model.score(a_removepunc, c_removepunc))
    return score


def webqa_fl(predictions, ground_truths, bart_score_path='bart_score.pth'):  # Please change the path to bart.pth
    model = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
    model.load(path=bart_score_path)
    normalizer = compute_bartscore_ParaBank(ground_truths, ground_truths, model)
    score = compute_bartscore_ParaBank(predictions, ground_truths, model)
    norm_score = score / normalizer
    norm_score[norm_score > 1] = 1
    norm_score = norm_score.tolist()
    fl = sum(norm_score) / len(norm_score)
    return {'fl': fl}


if __name__ == '__main__':
    # 'd5cbd2580dba11ecb1e81171463288e9'
    # prediction = 'both the sanfran 14 trolleybus and the third light rail trolleybus run off of the ground'
    # ground_truth = 'Both the SanFran 14TrSF trolleybus and the Third light rail vehicle run off of electricity'

    # FL = 1
    # prediction = 'Both the SanFran 14 trolleybus and the Third light rail vehicle run off of electricity'
    # ground_truth = 'Both the SanFran fourteen trolleybus and the Third light rail vehicle run off of electricity'
    # fl = webqa_fl([prediction], [ground_truth])

    prediction = "the facade of the bakery sattin et fils in rethel, france, is red"
    ground_truth = "the facade of the bakery sattin et fils in rethel, france, is colored red"
    fl = webqa_fl([prediction], [ground_truth])

    # acc = webqa_acc_approx(prediction, ground_truth, Qcate="Others")
    acc = 0
    print(f'FL: {fl}, Acc {acc}')
