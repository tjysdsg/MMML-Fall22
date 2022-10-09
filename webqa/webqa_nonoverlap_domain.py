import json
import string, re
import spacy
from operator import ge
from collections import defaultdict, Counter
nlp = spacy.load("en_core_web_sm", disable=["ner","textcat","parser"])

def get_dataset(path):
    with open(path) as f:
        dataset = json.load(f)
    return dataset


def normalize_text(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text): # additional: converting numbers to digit form
        return " ".join([str(toNum(w)) for w in text.split()])

    def remove_punc(text):
        exclude = set(string.punctuation) - set(['.'])
        text1 = "".join(ch for ch in text if ch not in exclude)
        return re.sub(r"\.(?!\d)", "", text1) # remove '.' if it's not a decimal point

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


def detectNum(l):
    result = []
    for w in l:
        try: result.append(str(int(w)))
        except: pass
    return result

    
def toNum(word):
    if word == 'point': return word
    try: return w2n.word_to_num(word)
    except:
        return word


def noun_overlap(dataset, test_dataset, qa_choice):
    nlp = spacy.load("en_core_web_sm")
    test_noun = []
    for id, qa_data in test_dataset.items():
        if qa_choice == 'Q':
            n_question = nlp(normalize_text(qa_data['Q']))
            q_noun = [chunck.text for chunck in n_question.noun_chunks]
            test_noun += q_noun
        elif qa_choice == 'A':
            for answer in qa_data['A']:
                n_answer = nlp(normalize_text(answer))
                a_noun = [chunck.text for chunck in n_answer.noun_chunks]
                test_noun += a_noun
    test_noun = set(test_noun)

    def get_noun_set(split):
        noun = []
        for id, qa_data in dataset.items():
            if qa_data['split'] == split \
               and (qa_data['Qcate'] == 'choice' \
               or qa_data['Qcate'] == 'Others' \
               or qa_data['Qcate'] == 'text'):
                if qa_choice == 'Q':
                    n_question = nlp(normalize_text(qa_data['Q']))
                    q_noun = [chunck.text for chunck in n_question.noun_chunks]
                    noun += q_noun
                elif qa_choice == 'A':
                    for answer in qa_data['A']:
                        n_answer = nlp(normalize_text(answer))
                        a_noun = [chunck.text for chunck in n_answer.noun_chunks]
                        noun += a_noun
        return set(noun)
    
    train_noun = get_noun_set('train')
    valid_noun = get_noun_set('val')
    train_val_shared_noun = train_noun.intersection(valid_noun)
    train_test_shared_noun = train_noun.intersection(test_noun)

    return len(train_noun), len(valid_noun), len(test_noun), \
           len(train_val_shared_noun), len(train_test_shared_noun)


def domainset_overlap(dataset, split, qcate=None):
    color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
    shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
    yesno_set = {'yes', 'no'}
    color_dict = defaultdict(int)
    shape_dict = defaultdict(int)
    yesno_dict = defaultdict(int)
    match_num = 0
    total_num = 0
    color_nonmatch = []
    shape_nonmatch = []
    yesno_nonmatch = []
    for id, qa_data in dataset.items():
        if qa_data['split'] == split and qa_data['Qcate'] == qcate:
            total_num += 1
            ground_truth = qa_data['A'][0]
            bow_gth = normalize_text(ground_truth).split()
            if qcate == 'number':
                bow_gth = detectNum(bow_gth)
            elif qcate == 'YesNo':
                bow_gth = list(yesno_set.intersection(bow_gth))
            elif qcate == 'shape':
                bow_gth = list(shape_set.intersection(bow_gth))
            elif qcate == 'color':
                bow_gth = list(color_set.intersection(bow_gth))
            else: 
                # TODO: fine-grained evaluation (e.g., content words) for text question types
                bow_gth = bow_gth
            if len(bow_gth) > 0:
                match_num += 1
            else:
                if qcate == 'YesNo':
                    yesno_nonmatch.append(id)
                elif qcate == 'shape': 
                    shape_nonmatch.append(id)
                elif qcate == 'color':
                    color_nonmatch.append(id)
    
    if qcate == 'YesNo':
        output = yesno_nonmatch
    elif qcate == 'shape':
        output = shape_nonmatch
    else:
        output = color_nonmatch

    with open(f'nonoverlap/nonoverlap_{split}_{qcate}.json', 'w') as outfile:
        json_obj = json.dumps(output, indent=4)
        outfile.write(json_obj)

    return match_num / total_num


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


def webqa_metrics_approx(prediction, ground_truth, Qcate="text"):
    color_set= {'orangebrown', 'spot', 'yellow', 'blue', 'rainbow', 'ivory', 'brown', 'gray', 'teal', 'bluewhite', 'orangepurple', 'black', 'white', 'gold', 'redorange', 'pink', 'blonde', 'tan', 'turquoise', 'grey', 'beige', 'golden', 'orange', 'bronze', 'maroon', 'purple', 'bluere', 'red', 'rust', 'violet', 'transparent', 'yes', 'silver', 'chrome', 'green', 'aqua'}
    shape_set = {'globular', 'octogon', 'ring', 'hoop', 'octagon', 'concave', 'flat', 'wavy', 'shamrock', 'cross', 'cylinder', 'cylindrical', 'pentagon', 'point', 'pyramidal', 'crescent', 'rectangular', 'hook', 'tube', 'cone', 'bell', 'spiral', 'ball', 'convex', 'square', 'arch', 'h', 'cuboid', 'step', 'rectangle', 'dot', 'oval', 'circle', 'star', 'crosse', 'crest', 'octagonal', 'cube', 'triangle', 'semicircle', 'domeshape', 'obelisk', 'corkscrew', 'curve', 'circular', 'xs', 'slope', 'pyramid', 'round', 'bow', 'straight', 'triangular', 'heart', 'fork', 'teardrop', 'fold', 'curl', 'spherical', 'diamond', 'keyhole', 'conical', 'dome', 'sphere', 'bellshaped', 'rounded', 'hexagon', 'flower', 'globe', 'torus'}
    yesno_set = {'yes', 'no'}
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
        accuracy = f1
    else:
        accuracy = recall
    return {"acc_approx": accuracy}


def webqa_metrics_calc(dataset, qcate):
    total_num = 0
    accumulated_accuracy = 0
    for id, qa_data in dataset.items():
        if qa_data['split'] == 'val':
            acc = webqa_metrics_approx(qa_data['A'][0], qa_data['A'][0], Qcate=qcate)
            total_num += 1
            accumulated_accuracy += acc['acc_approx']
    return accumulated_accuracy / total_num


if __name__ == '__main__':
    # get dataset
    train_dev_dataset = get_dataset('../../WebQA_data_first_release/WebQA_train_val.json')
    test_dataset = get_dataset('../../WebQA_data_first_release/WebQA_test.json')

    # domain set overlap 
    train_color_ratio = domainset_overlap(train_dev_dataset, 'train', 'color')
    train_shape_ratio = domainset_overlap(train_dev_dataset, 'train', 'shape')
    train_YesNo_ratio = domainset_overlap(train_dev_dataset, 'train', 'YesNo')
    val_color_ratio = domainset_overlap(train_dev_dataset, 'val', 'color')
    val_shape_ratio = domainset_overlap(train_dev_dataset, 'val', 'shape')
    val_YesNo_ratio = domainset_overlap(train_dev_dataset, 'val', 'YesNo')

    # print(train_color_ratio, train_shape_ratio, train_YesNo_ratio, val_color_ratio, val_shape_ratio, val_YesNo_ratio)

    # # qa overlap based on noun 
    # q_noun_tuple = noun_overlap(train_dev_dataset, test_dataset, 'Q')
    # a_noun_tuple = noun_overlap(train_dev_dataset, test_dataset, 'A')

    # # validation approximation accuracy
    # yesno_accuracy = webqa_metrics_calc(train_dev_dataset, 'YesNo')
    # color_accuracy = webqa_metrics_calc(train_dev_dataset, 'color')
    # shape_accuracy = webqa_metrics_calc(train_dev_dataset, 'shape')
    # number_accuracy = webqa_metrics_calc(train_dev_dataset, 'number')
    # text_accuracy = webqa_metrics_calc(train_dev_dataset, 'text')
    # Others_accuracy = webqa_metrics_calc(train_dev_dataset, 'Others')
    # choose_accuracy = webqa_metrics_calc(train_dev_dataset, 'choose')