from collections import defaultdict
import heapq
import os
from scipy.integrate import simps
import os.path

import numpy as np


class F1Evaluator:
    def __init__(self, model):
        self.model = model
        self.max_predicted_identifier_size = 6

    def compute_names_f1(self, features, real_targets, token_dictionary):
        """
        Compute the top X predictions for each paragraph vector.
        :param features:
        :param token_dictionary: contains all the non-unk words
        :rtype: PointSuggestionEvaluator
        """
        result_accumulator = PointSuggestionEvaluator()
        for i in xrange(features.shape[0]):
            result = self.model.predict_name(np.atleast_2d(features[i]))
            #print real_targets[i], result

            confidences = [suggestion[1] for suggestion in result]
            is_correct = [','.join(suggestion[0]) == real_targets[i] for suggestion in result]
            is_unkd = [is_unk(''.join(suggestion[0])) for suggestion in result]
            unk_word_accuracy = [self.unk_acc(suggestion[0], real_targets[i].split(','), token_dictionary) for suggestion in result]
            precision_recall = [token_precision_recall(suggestion[0], real_targets[i].split(',')) for suggestion in result]
            result_accumulator.add_result(confidences, is_correct, is_unkd, precision_recall, unk_word_accuracy)

        return result_accumulator

    def unk_acc(self, suggested_subtokens, real_subtokens, token_dictionary):
        real_unk_subtokens = set(t for t in real_subtokens if t not in token_dictionary)
        if len(real_unk_subtokens) == 0:
            return None
        return float(len([t for t in suggested_subtokens if t in real_unk_subtokens])) / len(real_unk_subtokens)


def is_unk(joined_tokens):
    return ["*"] * len(joined_tokens) == joined_tokens

class PointSuggestionEvaluator:
    def __init__(self):
        self.confidence_threshold = [0, 0.001, 0.005, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75,
                                     0.8, 0.85, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
        self.rank_to_eval = [1, 5]
        self.num_points = 0
        self.num_made_suggestions = np.array([[0] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.num_correct_suggestions = np.array([[0] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_precisions_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_recalls_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_f1_suggestions = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_unk_word_accuracy = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))
        self.sum_unk_word_locations = np.array([[0.] * len(self.confidence_threshold)] * len(self.rank_to_eval))


    def get_f1_at_all_ranks(self):
        """
        Get the F1 score, when all tokens are suggested at the self.rank_to_eval ranks
        :rtype: list
        :return: a list of the f1 scores
        """
        return self.sum_f1_suggestions[:, 0] / self.num_points

    def add_result(self, confidence, is_correct, is_unk, precision_recall, unk_word_accuracy):
        """
        Add a single point suggestion as a result.
        """
        confidence = np.array(confidence)
        is_correct = np.array(is_correct, dtype=np.bool)
        is_unk = np.array(is_unk, dtype=np.bool)
        self.num_points += 1
        if len(is_unk) == 0 or is_unk[0]:
            return  # No suggestions
        for i in xrange(len(self.confidence_threshold)):
            num_confident_suggestions = confidence[confidence >= self.confidence_threshold[i]].shape[0]
            for j in xrange(len(self.rank_to_eval)):
                rank = self.rank_to_eval[j]
                n_suggestions = min(rank, num_confident_suggestions)

                unk_at_rank = np.where(is_unk[:n_suggestions])[0]
                if unk_at_rank.shape[0] == 0:
                    unk_at_rank = n_suggestions + 1  # Beyond our current number of suggestions
                else:
                    unk_at_rank = unk_at_rank[0]

                if min(n_suggestions, unk_at_rank) > 0:
                    self.num_made_suggestions[j][i] += 1
                    if np.any(is_correct[:min(n_suggestions, unk_at_rank)]):
                        self.num_correct_suggestions[j][i] += 1

                    pr, re, f1 = self.get_best_f1(precision_recall[:min(n_suggestions, unk_at_rank)])
                    self.sum_precisions_suggestions[j][i] += pr
                    self.sum_recalls_suggestions[j][i] += re
                    self.sum_f1_suggestions[j][i] += f1

                unk_accuracies = [s for s in unk_word_accuracy[:min(n_suggestions, unk_at_rank)] if s is not None]
                if len(unk_accuracies) > 0:
                    # There is at least one UNK here
                    self.sum_unk_word_locations[j][i] += 1
                    self.sum_unk_word_accuracy[j][i] += max(unk_accuracies)

    def get_best_f1(self, suggestions_pr_re_f1):
        """
        Get the "best" precision, recall and f1 score from a list of tuples,
        picking the ones with the best f1
        """
        max_f1 = 0
        max_pr = 0
        max_re = 0
        for suggestion in suggestions_pr_re_f1:
            if suggestion[2] > max_f1:
                max_pr, max_re, max_f1 = suggestion
        return max_pr, max_re, max_f1

    def __str__(self):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        n_correct_suggestions = np.array(self.num_correct_suggestions, dtype=float)
        result_string = ""
        for i in xrange(len(self.rank_to_eval)):
            result_string += "At Rank " + str(self.rank_to_eval[i]) + os.linesep
            result_string += "Suggestion Frequency " + str(
                n_made_suggestions[i] / self.num_points) + os.linesep
            result_string += "Suggestion Accuracy " + str(
                np.divide(n_correct_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "UNK Accuracy " + str(
                np.divide(self.sum_unk_word_accuracy[i], self.sum_unk_word_locations[i])) + os.linesep

            result_string += "Suggestion Precision " + str(
                np.divide(self.sum_precisions_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Suggestion Recall " + str(
                np.divide(self.sum_recalls_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Suggestion F1 " + str(
                np.divide(self.sum_f1_suggestions[i], n_made_suggestions[i])) + os.linesep
            result_string += "Num Points: " + str(self.num_points) + os.linesep
        return result_string

    def get_f1_auc(self, rank_idx=0):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        f1_at_rank = np.divide(self.sum_f1_suggestions[rank_idx], n_made_suggestions[rank_idx])
        suggestion_freq = n_made_suggestions[rank_idx] / self.num_points

        mask = np.bitwise_not(np.isnan(f1_at_rank))
        unique_freq, unique_idx = np.unique(suggestion_freq[mask][::-1], return_index=True)
        unique_freq = unique_freq[::-1]
        f1_at_rank = f1_at_rank[mask][::-1][unique_idx][::-1]

        if len(unique_freq) > 0:
            return -simps(f1_at_rank, unique_freq)
        return 0

    def get_acc_auc(self, rank_idx=0):
        n_made_suggestions = np.array(self.num_made_suggestions, dtype=float)
        acc_at_rank = np.divide(self.num_correct_suggestions[rank_idx], n_made_suggestions[rank_idx])
        suggestion_freq = n_made_suggestions[rank_idx] / self.num_points
        mask = np.bitwise_not(np.isnan(acc_at_rank))
        unique_freq, unique_idx = np.unique(suggestion_freq[mask][::-1], return_index=True)
        unique_freq = unique_freq[::-1]

        acc_at_rank = acc_at_rank[mask][::-1][unique_idx][::-1]
        if len(unique_freq) > 0:
            return -simps(acc_at_rank, unique_freq)
        return 0


def token_precision_recall(predicted_parts, gold_set_parts):
    """
    Get the precision/recall for the given token.

    :param predicted_parts: a list of predicted parts
    :param gold_set_parts: a list of the golden parts
    :return: precision, recall, f1 as floats
    """
    ground = [tok.lower() for tok in gold_set_parts]

    tp = 0
    for subtoken in set(predicted_parts):
        if subtoken == "***" or subtoken is None:
            continue  # Ignore UNKs
        if subtoken.lower() in ground:
            ground.remove(subtoken.lower())
            tp += 1

    assert tp <= len(predicted_parts), (tp, len(predicted_parts))
    if len(predicted_parts) > 0:
       precision = float(tp) / len(predicted_parts)
    else:
       precision = 0

    assert tp <= len(gold_set_parts), (tp, gold_set_parts)
    if len(gold_set_parts) > 0:
       recall = float(tp) / len(gold_set_parts)
    else:
       recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.
    return precision, recall, f1
