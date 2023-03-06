""" Sourced from https://github.com/duorc/duorc/blob/master/evaluate.py """
"""          and http://nlp.cs.washington.edu/zeroshot/evaluate.py """

""" Since we also expect the answers in the SQuAD format, we reuse its code """
""" Official evaluation script for v1.1 of the SQuAD dataset. """

from collections import Counter
import string
import re
import argparse
import json
import sys
import codecs
import re
from itertools import groupby
import string
import sys
import numpy as np
from docopt import docopt


""" DUORC EVALUATION CODE """
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_duorc(dataset, predictions):
    f1 = exact_match = total = 0
    for dp in dataset:
        for qa in dp['qa']:
            total += 1
            if qa['id'] not in predictions:
                message = 'Question id ' + qa['id'] + \
                            ' not present. Will receive score 0.'
                print(message, file=sys.stderr)
                continue
            ground_truths = ['NA'] if len(qa['answers']) == 0 else qa['answers']
            prediction = predictions[qa['id']]
            exact_match += metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths)
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

""" RACE EVALUATION CODE """
PUNCTUATION = set(string.punctuation)

def eval_race(test_set, answer_file):
    with codecs.open(test_set, 'r', 'utf-8') as fin:
        data = [line.strip().split('\t') for line in fin]
    metadata = [x[:4] for x in data]
    gold = [set(x[4:]) for x in data]
    with codecs.open(answer_file, 'r', 'utf-8') as fin:
        answers = [line.strip() for line in fin]

    telemetry = []
    for m, g, a in zip(metadata, gold, answers):
        stats = score(g, a)
        telemetry.append([m[0], m[1], str(len(g) > 0), stats])
    return aprf(telemetry)


def parse_no_answers(results):
    p_answer = [a for i, a in sorted([(int(i), a) for i, a in results[0]['scores'].items()])]
    p_no_answer = [a for i, a in sorted([(int(i), a) for i, a in results[0]['na'].items()])]

    import numpy as np
    return [answer > no_answer for answer, no_answer in zip(p_answer, p_no_answer)]


def gb(collection, keyfunc):
    return [(k, list(g)) for k, g in groupby(sorted(collection, key=keyfunc), keyfunc)]


def aprf(g):
    tp, tn, sys_pos, real_pos = sum(map(lambda x: x[-1], g))
    total = len(g)
    # a = float(tp + tn) / total
    # nr = tn / float(total - real_pos)
    # npr = tn / float(total - sys_pos)
    if tp == 0:
        p = r = f = 0.0
    else:
        p = tp / float(sys_pos)
        r = tp / float(real_pos)
        f = 2 * p * r / (p + r)
    # return np.array((a, p, r, f, npr, nr))
    return np.array((p, r, f))


def score(gold, answer):
    if len(gold) > 0:
        gold = set.union(*[simplify(g) for g in gold])
    answer = simplify(answer)
    result = np.zeros(4)
    if answer == gold:
        if len(gold) > 0:
            result[0] += 1
        else:
            result[1] += 1
    if len(answer) > 0:
        result[2] += 1
    if len(gold) > 0:
        result[3] += 1
    return result


def simplify(answer):
    return set(''.join(c for c in t if c not in PUNCTUATION) for t in answer.strip().lower().split()) - {'the', 'a', 'an', 'and', ''}


def pretify(results):
    return ' \t '.join([': '.join((k, v)) for k, v in zip(['Precision', 'Recall', 'F1'], map(lambda r: '{0:.2f}%'.format(r*100), results))])


""" GENERAL CODE """
def load_eval(a, p, d):
    with open(a) as dataset_file:
        dataset = json.load(dataset_file)
    with open(p) as prediction_file:
        predictions = json.load(prediction_file)
        
    res = eval_duorc(dataset, predictions) if d == "duorc" else eval_race(dataset, predictions)

    return json.dumps(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for DuoRC and RACE')
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    load_eval(args.dataset_file, args.prediction_file)
