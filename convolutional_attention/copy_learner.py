import cPickle
import heapq
from collections import defaultdict
from math import ceil
import logging
import sys
import os
import time
import re

import numpy as np
from experimenter import ExperimentLogger
from convolutional_attention.copy_model import CopyConvolutionalAttentionalModel
from convolutional_attention.f1_evaluator import F1Evaluator
from convolutional_attention.token_naming_data import TokenCodeNamingData


class CopyAttentionalLearner:

    def __init__(self, hyperparameters):
        self.name_cx_size = hyperparameters["name_cx_size"]
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def train(self, input_file, patience=5, max_epochs=1000, minibatch_size=500):
        assert self.parameters is None, "Model is already trained"
        print "Extracting data..."
        train_data, validation_data, self.naming_data = TokenCodeNamingData.get_data_in_copy_convolution_format_with_validation(input_file, self.name_cx_size, .92, self.padding_size)
        train_name_targets, train_original_targets, train_name_contexts, train_code_sentences, train_code, train_copy_vectors, train_target_is_unk, train_original_name_ids = train_data
        val_name_targets, val_original_targets, val_name_contexts, val_code_sentences, val_code, val_copy_vectors, val_target_is_unk, val_original_name_ids = validation_data

        model = CopyConvolutionalAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary),
                                                  len(self.naming_data.name_dictionary), self.naming_data.name_empirical_dist)

        def compute_validation_logprob():
            return model.log_prob_no_predict(val_name_contexts, val_code_sentences, val_copy_vectors, val_target_is_unk, val_name_targets)

        best_params = [p.get_value() for p in model.train_parameters]
        best_name_score = float('-inf')
        ratios = np.zeros(len(model.train_parameters))
        n_batches = 0
        epochs_not_improved = 0
        print "[%s] Starting training..." % time.asctime()
        for i in xrange(max_epochs):
            start_time = time.time()
            name_ordering = np.arange(len(train_name_targets), dtype=np.int32)
            np.random.shuffle(name_ordering)
            sys.stdout.write(str(i))
            num_minibatches = min(int(ceil(float(len(train_name_targets)) / minibatch_size))-1, 25)  # Clump minibatches

            for j in xrange(num_minibatches):
                name_batch_ids = name_ordering[j * minibatch_size:(j + 1) * minibatch_size]
                batch_code_sentences = train_code_sentences[name_batch_ids]
                for k in xrange(len(name_batch_ids)):
                        idx = name_batch_ids[k]
                        if train_target_is_unk[idx] == 1 and np.sum(train_copy_vectors[idx]) == 0:
                            continue
                        model.grad_accumulate(train_name_contexts[idx], batch_code_sentences[k],
                                              train_copy_vectors[idx], train_target_is_unk[idx],
                                              train_name_targets[idx])
                assert len(name_batch_ids) > 0
                ratios += model.grad_step()
                n_batches += 1
            sys.stdout.write("|")
            if i % 1 == 0:
                name_ll = compute_validation_logprob()
                if name_ll > best_name_score:
                    best_name_score = name_ll
                    best_params = [p.get_value() for p in model.train_parameters]
                    print "At %s validation: name_ll=%s [best so far]" % (i, name_ll)
                    epochs_not_improved = 0
                else:
                    print "At %s validation: name_ll=%s" % (i, name_ll)
                    epochs_not_improved += 1
                for k in xrange(len(model.train_parameters)):
                    print "%s: %.0e" % (model.train_parameters[k].name, ratios[k] / n_batches)
                n_batches = 0
                ratios = np.zeros(len(model.train_parameters))

            if epochs_not_improved >= patience:
                print "Not improved for %s epochs. Stopping..." % patience
                break
            elapsed = int(time.time() - start_time)
            print "Epoch elapsed %sh%sm%ss" % ((elapsed / 60 / 60) % 60, (elapsed / 60) % 60, elapsed % 60)
        print "[%s] Training Finished..." % time.asctime()
        self.parameters = best_params
        model.restore_parameters(best_params)
        self.model = model

    def save(self, filename):
        model_tmp = self.model
        del self.model
        with open(filename, 'wb') as f:
            cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
        self.model = model_tmp

    @staticmethod
    def load(filename):
        """
        :type filename: str
        :rtype: CopyAttentionalLearner
        """
        with open(filename, 'rb') as f:
            learner = cPickle.load(f)
        learner.model = CopyConvolutionalAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary),
                             len(learner.naming_data.name_dictionary), learner.naming_data.name_empirical_dist)
        learner.model.restore_parameters(learner.parameters)
        return learner

    def evaluate_decision_accuracy(self, input_file):
        test_data, original_names = self.naming_data.get_data_in_copy_convolution_format(input_file, self.name_cx_size, self.padding_size)
        name_targets, original_targets, name_contexts, code_sentences, code, copy_vectors, target_is_unk, original_name_ids = test_data

        num_suggestion_points = 0
        num_correct_suggestions_r1 = 0
        num_correct_suggestions_r5 = 0

        for i in xrange(len(name_targets)):
            current_code = code[i]
            current_code_sentence = code_sentences[i]
            current_name_contexts = name_contexts[i]
            copy_prob, copy_word, suggestions, subtoken_target_logprob = self.get_suggestions_for_next_subtoken(current_code,
                                                                                       current_code_sentence,
                                                                                       current_name_contexts)

            if original_targets[i] == suggestions[0]:
                num_correct_suggestions_r1 += 1
            if original_targets[i] in suggestions[:5]:
                num_correct_suggestions_r5 += 1
            # DEBUG:
            print "%s:%s--%s(%.2f, isUNK=%s)" % (original_targets[i], suggestions[:5], copy_word, copy_prob, target_is_unk[i])
            num_suggestion_points += 1

        print "Rank 1 Accuracy: %s" % (float(num_correct_suggestions_r1) / num_suggestion_points)
        print "Rank 5 Accuracy: %s" % (float(num_correct_suggestions_r5) / num_suggestion_points)

    def get_suggestions_for_next_subtoken(self, current_code, current_code_sentence, current_name_contexts):
        copy_weights, copy_prob, name_logprobs = self.model.copy_probs(current_name_contexts, current_code_sentence)
        copy_weights /= np.sum(copy_weights) # convert to probabilities
        copy_dist = self.get_copy_distribution(copy_weights, current_code)

        if len(copy_dist) > 0:
            copy_word = copy_dist.keys()[0]
            top_score = copy_dist[copy_word]
        else:
            copy_word = None
            top_score = None
        for word, score in copy_dist.iteritems():
            if score > top_score:
                top_score = score
                copy_word = word

        subtoken_target_logprob = defaultdict(lambda: float('-inf')) # log prob of each subtoken
        for j in xrange(len(self.naming_data.all_tokens_dictionary) - 1):
            subtoken_target_logprob[self.naming_data.all_tokens_dictionary.get_name_for_id(j)] = np.log(1. - copy_prob) + name_logprobs[j]

        copy_logprob = np.log(copy_prob)
        for word, word_copied_log_prob in copy_dist.iteritems():
            subtoken_target_logprob[word] = np.logaddexp(subtoken_target_logprob[word], copy_logprob + word_copied_log_prob)

        suggestions = sorted(subtoken_target_logprob.keys(), key=lambda x: subtoken_target_logprob[x], reverse=True)
        return copy_prob, copy_word, suggestions, subtoken_target_logprob

    identifier_matcher = re.compile('[a-zA-Z0-9]+')

    def get_copy_distribution(self, copy_weights, code):
        """
        Return a distribution over the copied tokens. Some tokens may be invalid (ie. non alphanumeric), there are
         excluded, but the distribution is not re-normalized. This is probabilistically weird, but it possibly lets the
         non-copy mechanism to recover.
        """
        token_probs = defaultdict(lambda: float('-inf')) # log prob of each token
        for code_token, weight in zip(code, copy_weights):
            if self.identifier_matcher.match(code_token) is not None:
                token_probs[code_token] = np.logaddexp(token_probs[code_token], np.log(weight))
        return token_probs


    def get_copy_pos(self, copy_weights, code):
        sorted_idxs = np.argsort(-copy_weights)
        for i in xrange(len(sorted_idxs)):
            token = code[sorted_idxs[i]]
            if self.identifier_matcher.match(token) is None:
                continue
            return sorted_idxs[i], i
        return np.argmax(copy_weights), i # There is not a single identifier, just use something...

    def evaluate_copy_decisions(self, input_file):
        test_data, original_names = self.naming_data.get_data_in_copy_convolution_format(input_file, self.name_cx_size, self.padding_size)
        name_targets, orignal_targets, name_contexts, code_sentences, code, copy_vectors, target_is_unk, original_name_ids = test_data

        # Pct of correct copy decisions (assuming a .5 threshold)
        copy = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

        # Given that we want to copy, pct of correct positions to copy
        copy_pos = {"correct": 0, "all": 0}

        for i in xrange(len(name_targets)):
            copy_weights, copy_prob, name_logprobs = self.model.copy_probs(name_contexts[i], code_sentences[i])
            should_we_copy = len(copy_vectors[i][copy_vectors[i] > 0]) > 0
            we_copied = copy_prob > .5
            if we_copied and should_we_copy:
                copy["tp"] += 1
            elif we_copied and not should_we_copy:
                copy["fp"] += 1
            elif not we_copied and should_we_copy:
                copy["fn"] += 1
            else:
                copy["tn"] += 1

            if should_we_copy:
                copied_pos, times_backoff = self.get_copy_pos(copy_weights, code[i])
                copied_correct_pos = copy_vectors[i][copied_pos] == 1
                copy_pos["all"] += 1
                if copied_correct_pos:
                    copy_pos["correct"] += 1

        print "Statistics"
        print "Copy Decisions: %s" % copy
        print "Copy Position Decisions: %s" % copy_pos

    def evaluate_suggestion_decisions(self, input_file):
        test_data, original_names = self.naming_data.get_data_in_copy_convolution_format(input_file, self.name_cx_size, self.padding_size)
        name_targets, orignal_targets, name_contexts, code_sentences, code, copy_vectors, target_is_unk, original_name_ids = test_data
        code = np.array(code, dtype=np.object)

        ids, unique_idx = np.unique(original_name_ids, return_index=True)
        eval = F1Evaluator(self)
        point_suggestion_eval = eval.compute_names_f1(code[unique_idx], original_names,
                                                      self.naming_data.all_tokens_dictionary.get_all_names())
        print point_suggestion_eval
        return point_suggestion_eval.get_f1_at_all_ranks()

    def predict_name(self, code, max_predicted_identifier_size=4, max_steps=100):
        assert self.parameters is not None, "Model is not trained"
        code = code[0]
        code_features = [self.naming_data.all_tokens_dictionary.get_id_or_unk(tok) for tok in code]
        padding = [self.naming_data.all_tokens_dictionary.get_id_or_unk(self.naming_data.NONE)]
        if self.padding_size % 2 == 0:
            code_sentence = padding * (self.padding_size / 2) + code_features + padding * (self.padding_size / 2)
        else:
            code_sentence = padding * (self.padding_size / 2 + 1) + code_features + padding * (self.padding_size / 2)

        code_features = np.array(code_sentence, dtype=np.int32)

        ## Predict all possible names
        suggestions = defaultdict(lambda: float('-inf'))  # A list of tuple of full suggestions (token, prob)
        # A stack of partial suggestion in the form ([subword1, subword2, ...], logprob)
        possible_suggestions_stack = [
            ([self.naming_data.NONE] * (self.name_cx_size - 1) + [self.naming_data.SUBTOKEN_START], [], 0)]
        # Keep the max_size_to_keep suggestion scores (sorted in the heap). Prune further exploration if something has already
        # lower score
        predictions_probs_heap = [float('-inf')]
        max_size_to_keep = 15
        nsteps = 0
        while True:
            scored_list = []
            while len(possible_suggestions_stack) > 0:
                subword_tokens = possible_suggestions_stack.pop()

                # If we're done, append to full suggestions
                if subword_tokens[0][-1] == self.naming_data.SUBTOKEN_END:
                    final_prediction = tuple(subword_tokens[1][:-1])
                    if len(final_prediction) == 0:
                        continue
                    log_prob_of_suggestion = np.logaddexp(suggestions[final_prediction], subword_tokens[2])
                    if log_prob_of_suggestion > predictions_probs_heap[0] and not log_prob_of_suggestion == float('-inf'):
                        # Push only if the score is better than the current minimum and > 0 and remove extraneous entries
                        suggestions[final_prediction] = log_prob_of_suggestion
                        heapq.heappush(predictions_probs_heap, log_prob_of_suggestion)
                        if len(predictions_probs_heap) > max_size_to_keep:
                            heapq.heappop(predictions_probs_heap)
                    continue
                elif len(subword_tokens[1]) > max_predicted_identifier_size:  # Stop recursion here
                    continue

                # Convert subword context
                context = [self.naming_data.name_dictionary.get_id_or_unk(k) for k in
                           subword_tokens[0][-self.name_cx_size:]]
                assert len(context) == self.name_cx_size
                context = np.array(context, dtype=np.int32)

                # Predict next subwords
                copy_prob, copy_word, next_subtoken_suggestions, subtoken_target_logprob \
                    = self.get_suggestions_for_next_subtoken(code, code_features, context)

                subtoken_target_logprob["***"] = subtoken_target_logprob[self.naming_data.all_tokens_dictionary.get_unk()]

                def get_possible_options(subword_name):
                    if subword_name == self.naming_data.all_tokens_dictionary.get_unk():
                        subword_name = "***"
                    name = subword_tokens[1] + [subword_name]
                    return subword_tokens[0][1:] + [subword_name], name, subtoken_target_logprob[subword_name] + \
                           subword_tokens[2]

                possible_options = [get_possible_options(next_subtoken_suggestions[i]) for i in xrange(max_size_to_keep)]
                # Disallow suggestions that contain duplicated subtokens.
                scored_list.extend(filter(lambda x: len(x[1]) == 1 or x[1][-1] != x[1][-2], possible_options))

            # Prune
            scored_list = filter(lambda suggestion: suggestion[2] >= predictions_probs_heap[0] and suggestion[2] >= float('-inf'), scored_list)
            scored_list.sort(key=lambda entry: entry[2], reverse=True)

            # Update
            possible_suggestions_stack = scored_list[:max_size_to_keep]
            nsteps += 1
            if nsteps >= max_steps:
                break

        # Sort and append to predictions
        suggestions = [(identifier, np.exp(logprob)) for identifier, logprob in suggestions.items()]
        suggestions.sort(key=lambda entry: entry[1], reverse=True)
        #print suggestions
        return suggestions


def run_from_config(params, *args):
    if len(args) < 2:
        print "No input file or test file given: %s:%s" % (args, len(args))
        sys.exit(-1)
    input_file = args[0]
    test_file = args[1]
    if len(args) > 2:
        num_epochs = int(args[2])
    else:
        num_epochs = 80

    # Transform params
    params["D"] = 2 ** params["logD"]
    params["conv_layer1_nfilters"] = 2 ** params["log_conv_layer1_nfilters"]
    params["conv_layer2_nfilters"] = 2 ** params["log_conv_layer2_nfilters"]

    model = CopyAttentionalLearner(params)
    model.train(input_file, max_epochs=num_epochs)
    f1_values = model.evaluate_suggestion_decisions(test_file)
    return -f1_values[1]

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print 'Usage <input_file> <max_num_epochs> d <test_file>'
        sys.exit(-1)
    logging.basicConfig(level=logging.INFO)
    input_file = sys.argv[1]
    max_num_epochs = int(sys.argv[2])
    params = {
        "D": int(sys.argv[3]),
        "name_cx_size": 1,
        "conv_layer1_nfilters": 32,
        "conv_layer2_nfilters": 16,
        "layer1_window_size": 18,
        "layer2_window_size": 19,
        "layer3_window_size": 2,
        "log_code_rep_init_scale": -3.1,
        "log_name_rep_init_scale": -1,
        "log_layer1_init_scale": -3.68,
        "log_layer2_init_scale": -4,
        "log_layer3_init_scale": -4,
        "log_name_cx_init_scale": -1.06,
        "log_copy_init_scale":-0.5,
        "log_learning_rate": -3.05,
        "rmsprop_rho": .99,
        "momentum": 0.87,
        "dropout_rate": 0.4,
        "grad_clip":.75
    }

    params["train_file"] = input_file
    params["test_file"] = sys.argv[4]
    with ExperimentLogger("CopyAttentionalLearner", params) as experiment_log:
        model = CopyAttentionalLearner(params)
        model.train(input_file, max_epochs=max_num_epochs)
        model.save("copy_convolutional_att_model" + os.path.basename(params["train_file"]) + ".pkl")

        model2 = CopyAttentionalLearner.load("copy_convolutional_att_model" + os.path.basename(params["train_file"]) + ".pkl")
        model2.evaluate_copy_decisions(sys.argv[4])
        model2.evaluate_decision_accuracy(sys.argv[4])
        f1_values = model2.evaluate_suggestion_decisions(sys.argv[4])
        print f1_values
        experiment_log.record_results({"f1_at_rank1":f1_values[0], "f1_at_rank5":f1_values[1]})
