""" Official evaluation script for DATASET_NAME dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import datasets
import numpy as np
from typing import List, Dict
import pandas as pd
import os
from os.path import join

bleurt_metric = datasets.load_metric('bleurt', **{'config_name': 'bleurt-base-128'})
rouge_metric = datasets.load_metric('rouge')
sacrebleu_metric = datasets.load_metric('sacrebleu')


def bleurt(prediction: str, ground_truth: str):
    score = bleurt_metric.compute(
        predictions=[prediction],
        references=[ground_truth]
    )
    return np.mean(score['scores'])


def rouge(prediction: str, ground_truth: str):
    score = rouge_metric.compute(
        predictions=[prediction],
        references=[ground_truth],
        **{'use_agregator': False, 'use_stemmer': True, 'rouge_types': ['rougeL']}
    )
    return score['rougeL'][0].fmeasure


def sacrebleu(prediction: str, ground_truth_list: List[str]):
    score = sacrebleu_metric.compute(
        predictions=[prediction],
        references=[ground_truth_list]  # scarebleu expects several golds per
    )
    return score['score'] / 100


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


def f1_score(prediction: str, ground_truth: str):
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


def exact_match_score(prediction: str, ground_truth: str):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset: List[str], predictions: List[str], evaluation_types: List[str], dataset_name) -> Dict:
    '''
    :param dataset: json file containing the gold labels
    :param predictions: list of strings, as the predictions
    :param evaluation_types: TODO
    :return:
    '''

    # TODO: update this so that the script accepts partial predictions
    assert len(predictions) == len(dataset), \
        f"The pred file does not have the same length as the gold data: {len(dataset)} vs {len(predictions)}"

    metrics = {}
    max_eval = 10
    for idx, (gold, pred) in datasets.tqdm(enumerate(zip(dataset, predictions))):

        # hack, to make it easier faster to test this code; drop it later
        # if idx > max_eval:
        #     break

        gold_outputs = str(gold).split(', ')

        # long-range text generation metrics
        if "long_generation" in evaluation_types:
            if 'rouge' not in metrics:
                metrics['sacrebleu'] = metrics['bleurt'] = metrics['rouge'] = 0
            metrics['rouge'] += metric_max_over_ground_truths(rouge, pred, gold_outputs)
            metrics['bleurt'] += metric_max_over_ground_truths(bleurt, pred, gold_outputs)
            metrics['sacrebleu'] += sacrebleu(pred, gold_outputs)

        # squad-like f1/em metrics
        if "short_answer" in evaluation_types:
            if 'exact_match' not in metrics:
                metrics['f1'] = metrics['exact_match'] = 0
            metrics['exact_match'] += metric_max_over_ground_truths(exact_match_score, pred, gold_outputs)
            metrics['f1'] += metric_max_over_ground_truths(f1_score, pred, gold_outputs)

        # e.g., selecting A, B, C, etc.
        if "classification" in evaluation_types:
            pass

        # TODO: task-specific constraints:
        if "winogrande_question_generation_object" in dataset_name:
            #   e.g., for Winogrande, check if PerxonX comes before PersonY: metrics['personx-before-persony_score'] = ...
            pass

    # normalize tne metrics
    for key in metrics.keys():
        metrics[key] /= len(predictions)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for DATASET-NAME ')
    parser.add_argument('--dataset_file', help='Dataset file')
    parser.add_argument('--prediction_file',
                        help='Prediction File; expected to be a json file of the following format: { "predictions": ["pred1", "pred2", ...] } ')
    parser.add_argument('--save_results',
                        help='Give a path where you want to save result json')
    args = parser.parse_args()
    print(args.dataset_file)
    data_df= pd.read_csv(args.dataset_file)
    with open(args.prediction_file) as prediction_file:
        predictions = prediction_file.read()
    
    predictions_list= predictions.split('\n')

    # TODO: read this from the gold data
    # evaluation_types = ['short_answer']
    evaluation_types = ['long_generation', 'short_answer']

    if not os.path.exists(args.save_results):
        os.makedirs(args.save_results)

    result= evaluate(data_df.Output.tolist(), predictions_list, evaluation_types, args.dataset_file)
    with open(join(args.save_results, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=4)
    
    print(result)
