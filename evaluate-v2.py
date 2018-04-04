""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
import pandas as pd


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
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    precision_scores_for_ground_truths = []
    recall_scores_for_ground_truths = []
    for ground_truth in ground_truths:
        if str(metric_fn).lower().__contains__('f1'):
            score, precision, recall = metric_fn(prediction, ground_truth)
        else:
            score = metric_fn(prediction, ground_truth)
            precision = 0
            recall = 0
        scores_for_ground_truths.append(score)
        precision_scores_for_ground_truths.append(precision)
        recall_scores_for_ground_truths.append(recall)
    return max(scores_for_ground_truths), max(precision_scores_for_ground_truths), max(recall_scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                _em, _temp1, temp2 = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                exact_match += _em
                _f1, _precision, _recall = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
                f1 += _f1

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'avg_exact_match': exact_match, 'avg_f1': f1, 'total_records': total}

def measure_quality(dataset, predictions):
    f1 = precision = recall = exact_match = total_records = filtered_records = 0
    results = []
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total_records += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                _em, _temp1, temp2= metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                exact_match += _em
                _f1, _precision, _recall  = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                if _f1 != 0:
                    filtered_records += 1
                    f1 += _f1
                    precision += _precision
                    recall += _recall
                    results.append((qa['id'], prediction, ground_truths, _precision, _recall, _f1))
    exact_match = 100.0 * exact_match / filtered_records
    f1 = 100.0 * f1 / filtered_records
    recall = 100.0 * recall / filtered_records
    precision = 100.0 * precision / filtered_records

    return {'filtered_records': filtered_records, 'total_records': total_records, 'exact_match': exact_match, 'avg_f1': f1, 'avg_precision': precision, 'avg_recall': recall}, results

if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', nargs='?', help='Dataset file')
    parser.add_argument('prediction_file', nargs= '?', help='Prediction File')
    parser.add_argument('result_path', nargs='?', help='Path the results will be written to')
    args = parser.parse_args()
    if args.dataset_file is None:
        args.dataset_file = '/home/jackalhan/data/squad/test-v1.1.json'
    if args.prediction_file is None:
        args.prediction_file = 'log/answer/answer.json'
    if args.result_path is None:
        args.result_path = 'data'
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print(json.dumps(evaluate(dataset, predictions)))
    summary, results = measure_quality(dataset, predictions)
    print(50* '-')
    print(json.dumps(summary))
    df_all = pd.DataFrame(data=results, columns=['qid', 'prediction', 'ground_truths', 'precision', 'recall', 'f1'])
    df_all.to_csv(os.path.join(args.result_path, 'detailed_results.csv'))
    del df_all['prediction']
    del df_all['ground_truths']
    df_all.to_csv(os.path.join(args.result_path, 'simple_results.csv'))

