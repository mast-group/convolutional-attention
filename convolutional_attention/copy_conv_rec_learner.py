import cPickle
import heapq
import os
from collections import defaultdict
from math import ceil
import sys
import time

import numpy as np
import re
from experimenter import ExperimentLogger

from convolutional_attention.copy_conv_rec_model import CopyConvolutionalRecurrentAttentionalModel
from convolutional_attention.f1_evaluator import F1Evaluator
from convolutional_attention.token_naming_data import TokenCodeNamingData

class ConvolutionalCopyAttentionalRecurrentLearner:

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def train(self, input_file, patience=5, max_epochs=1000, minibatch_size=500):
        assert self.parameters is None, "Model is already trained"
        print "saving best result so far to %s"%(
            "copy_convolutional_att_rec_model" +
            os.path.basename(self.hyperparameters["train_file"]) +
            ".pkl")
        print "Extracting data..."
        # Get data (train, validation)
        train_data, validation_data, self.naming_data = TokenCodeNamingData.get_data_in_recurrent_copy_convolution_format_with_validation(input_file, .92, self.padding_size)
        train_name_targets, train_code_sentences, train_code, train_target_is_unk, train_copy_vectors = train_data
        val_name_targets, val_code_sentences, val_code, val_target_is_unk, val_copy_vectors = validation_data

        # Create theano model and train
        model = CopyConvolutionalRecurrentAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary),
                                   self.naming_data.name_empirical_dist)
        self.model = model

        def compute_validation_score_names():
            return model.log_prob_with_targets(val_code_sentences, val_copy_vectors, val_target_is_unk, val_name_targets)

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
                if (j + 1) * minibatch_size > len(name_ordering):
                    j = 0
                name_batch_ids = name_ordering[j * minibatch_size:(j + 1) * minibatch_size]
                batch_code_sentences = train_code_sentences[name_batch_ids]
                for k in xrange(len(name_batch_ids)):
                    pos = name_batch_ids[k]
                    model.grad_accumulate(batch_code_sentences[k], train_copy_vectors[pos],
                                          train_target_is_unk[pos], train_name_targets[pos])
                assert len(name_batch_ids) > 0
                ratios += model.grad_step()
                sys.stdout.write("\r%d %d"%(i, n_batches))
                n_batches += 1
            sys.stdout.write("|")
            if i % 1 == 0:
                name_ll = compute_validation_score_names()
                if name_ll > best_name_score:
                    best_name_score = name_ll
                    best_params = [p.get_value() for p in model.train_parameters]
                    self.parameters = best_params
                    print "At %s validation: name_ll=%s [best so far]" % (i, name_ll)
                    epochs_not_improved = 0
                    self.save(
                        "copy_convolutional_att_rec_model" +
                        os.path.basename(self.hyperparameters["train_file"]) +
                        ".pkl")
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

    def get_suggestions_for_next_subtoken(self, current_code, current_code_sentence, predicted_target_tokens_so_far):
        copy_weights, copy_prob, name_logprobs = self.model.copy_probs(predicted_target_tokens_so_far, current_code_sentence)
        copy_weights, copy_prob, name_logprobs = copy_weights[-1], copy_prob[-1], name_logprobs[-1]  # Get values for the last prediction
        copy_weights /= np.sum(copy_weights) # convert to probabilities
        copy_dist = self.get_copy_distribution(copy_weights, current_code)

        subtoken_target_logprob = defaultdict(lambda: float('-inf'))  # log prob of each subtoken
        for j in xrange(len(self.naming_data.all_tokens_dictionary) - 1):
            subtoken_target_logprob[self.naming_data.all_tokens_dictionary.get_name_for_id(j)] = np.log(1. - copy_prob) + name_logprobs[j]

        copy_logprob = np.log(copy_prob)
        for word, word_copied_log_prob in copy_dist.iteritems():
            subtoken_target_logprob[word] = np.logaddexp(subtoken_target_logprob[word], copy_logprob + word_copied_log_prob)

        suggestions = sorted(subtoken_target_logprob.keys(), key=lambda x: subtoken_target_logprob[x], reverse=True)
        return copy_prob, suggestions, subtoken_target_logprob

    def predict_name(self, code, max_predicted_identifier_size=7, max_steps=100):
        assert self.parameters is not None, "Model is not trained"
        code = code[0]

        code_sentence = [self.naming_data.all_tokens_dictionary.get_id_or_unk(tok) for tok in code]
        padding = [self.naming_data.all_tokens_dictionary.get_id_or_unk(self.naming_data.NONE)]
        if self.padding_size % 2 == 0:
            code_sentence = padding * (self.padding_size / 2) + code_sentence + padding * (self.padding_size / 2)
        else:
            code_sentence = padding * (self.padding_size / 2 + 1) + code_sentence + padding * (self.padding_size / 2)

        code_sentence = np.array(code_sentence, dtype=np.int32)

        suggestions = defaultdict(lambda: float('-inf'))  # A list of tuple of full suggestions (token, prob)
        # A stack of partial suggestion in the form ([subword1, subword2, ...], logprob)
        possible_suggestions_stack = [
            ([self.naming_data.SUBTOKEN_START], [], 0)]
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
                previous_subtokens = [self.naming_data.all_tokens_dictionary.get_id_or_unk(k) for k in subword_tokens[0]]
                previous_subtokens = np.array(previous_subtokens, dtype=np.int32)

                # Predict next subwords
                copy_prob, next_subtoken_suggestions, subtoken_target_logprob \
                    = self.get_suggestions_for_next_subtoken(code, code_sentence, previous_subtokens)

                subtoken_target_logprob["***"] = subtoken_target_logprob[self.naming_data.all_tokens_dictionary.get_unk()]

                def get_possible_options(subword_name):
                    # TODO: Handle UNK differently?
                    if subword_name == self.naming_data.all_tokens_dictionary.get_unk():
                        subword_name = "***"
                    name = subword_tokens[1] + [subword_name]
                    return subword_tokens[0] + [subword_name], name, subtoken_target_logprob[subword_name] + \
                           subword_tokens[2]

                possible_options = [get_possible_options(next_subtoken_suggestions[i]) for i in xrange(max_size_to_keep)]
                # Disallow suggestions that contain duplicated subtokens.
                scored_list.extend(filter(lambda x: len(x[1])==1 or x[1][-1] != x[1][-2], possible_options))

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
        # print suggestions
        return suggestions

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
        :rtype: ConvolutionalAttentionalLearner
        """
        with open(filename, 'rb') as f:
            learner = cPickle.load(f)
        learner.model = CopyConvolutionalRecurrentAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary),
                             learner.naming_data.name_empirical_dist)
        learner.model.restore_parameters(learner.parameters)
        return learner


def run_from_config(params, *args):
    if len(args) < 2:
        print "No input file or test file given: %s:%s" % (args, len(args))
        sys.exit(-1)
    input_file = args[0]
    test_file = args[1]
    if len(args) > 2:
        num_epochs = int(args[2])
    else:
        num_epochs = 1000

    params["D"] = 2 ** params["logD"]
    params["conv_layer1_nfilters"] = 2 ** params["log_conv_layer1_nfilters"]
    params["conv_layer2_nfilters"] = 2 ** params["log_conv_layer2_nfilters"]

    model = ConvolutionalCopyAttentionalRecurrentLearner(params)
    model.train(input_file, max_epochs=num_epochs)

    test_data, original_names = model.naming_data.get_data_in_recurrent_copy_convolution_format(test_file, model.padding_size)
    test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data
    eval = F1Evaluator(model)
    point_suggestion_eval = eval.compute_names_f1(test_code, original_names,
                                                  model.naming_data.all_tokens_dictionary.get_all_names())
    return -point_suggestion_eval.get_f1_at_all_ranks()[1]


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print 'Usage <input_file> <max_num_epochs> d <test_file>'
        sys.exit(-1)

    input_file = sys.argv[1]
    max_num_epochs = int(sys.argv[2])
    params = {
        "D": int(sys.argv[3]),
        "conv_layer1_nfilters": 32,
        "conv_layer2_nfilters": 16,
        "layer1_window_size": 18,
        "layer2_window_size": 19,
        "layer3_window_size": 2,
        "log_name_rep_init_scale": -1,
        "log_layer1_init_scale": -3.68,
        "log_layer2_init_scale": -4,
        "log_layer3_init_scale": -4,
        "log_hidden_init_scale": -1,
        "log_copy_init_scale":-0.5,
        "log_learning_rate": -3.05,
        "rmsprop_rho": .99,
        "momentum": 0.87,
        "dropout_rate": 0.4,
        "grad_clip":.75
    }


    params["train_file"] = input_file
    if len(sys.argv) > 4:
        params["test_file"] = sys.argv[4]
    with ExperimentLogger("ConvolutionalCopyAttentionalRecurrentLearner", params) as experiment_log:
        if max_num_epochs:
            model = ConvolutionalCopyAttentionalRecurrentLearner(params)
            model.train(input_file, max_epochs=max_num_epochs)
            model.save("copy_convolutional_att_rec_model" + os.path.basename(params["train_file"]) + ".pkl")

        if params.get("test_file") is None:
            exit()

        model2 = ConvolutionalCopyAttentionalRecurrentLearner.load("copy_convolutional_att_rec_model" + os.path.basename(params["train_file"]) + ".pkl")

        test_data, original_names = model2.naming_data.get_data_in_recurrent_copy_convolution_format(params["test_file"], model2.padding_size)
        test_name_targets, test_code_sentences, test_code, test_target_is_unk, test_copy_vectors = test_data
        #name_ll = model2.model.log_prob_with_targets(test_code_sentences, test_name_targets)
        #print "Test name_ll=%s" % name_ll

        eval = F1Evaluator(model2)
        point_suggestion_eval = eval.compute_names_f1(test_code, original_names,
                                                      model2.naming_data.all_tokens_dictionary.get_all_names())
        print point_suggestion_eval
        results = point_suggestion_eval.get_f1_at_all_ranks()
        print results
        experiment_log.record_results({"f1_at_rank1": results[0], "f1_at_rank5":results[1]})
