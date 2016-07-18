import cPickle
import logging
from math import ceil
import sys
import os
import time

import numpy as np
from experimenter import ExperimentLogger

from convolutional_attention.conv_attentional_model import ConvolutionalAttentionalModel
from convolutional_attention.f1_evaluator import F1Evaluator
from convolutional_attention.token_naming_data import TokenCodeNamingData


class ConvolutionalAttentionalLearner:

    def __init__(self, hyperparameters):
        self.name_cx_size = hyperparameters["name_cx_size"]
        self.hyperparameters = hyperparameters
        self.naming_data = None
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3
        self.parameters = None

    def train(self, input_file, patience=5, max_epochs=1000, minibatch_size=500):
        assert self.parameters is None, "Model is already trained"
        print "Extracting data..."
        # Get data (train, validation)
        train_data, validation_data, self.naming_data = TokenCodeNamingData.get_data_in_convolution_format_with_validation(input_file, self.name_cx_size, .92, self.padding_size)
        train_name_targets, train_name_contexts, train_code_sentences, train_original_name_ids = train_data
        val_name_targets, val_name_contexts, val_code_sentences, val_original_name_ids = validation_data

        # Create theano model and train
        model = ConvolutionalAttentionalModel(self.hyperparameters, len(self.naming_data.all_tokens_dictionary), len(self.naming_data.name_dictionary),
                                   self.naming_data.name_empirical_dist)

        def compute_validation_score_names():
            return model.log_prob_with_targets(val_name_contexts, val_code_sentences, val_name_targets)

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
                        out = model.grad_accumulate(train_name_contexts[name_batch_ids[k]], batch_code_sentences[k], train_name_targets[name_batch_ids[k]])
                assert len(name_batch_ids) > 0
                ratios += model.grad_step()
                n_batches += 1
            sys.stdout.write("|")
            if i % 1 == 0:
                name_ll = compute_validation_score_names()
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

    def predict_name(self, code_features):
        assert self.parameters is not None, "Model is not trained"
        next_name_log_probs = lambda cx: self.model.log_prob(cx, code_features)
        return self.naming_data.get_suggestions_given_name_prefix(next_name_log_probs, self.name_cx_size)

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
        learner.model = ConvolutionalAttentionalModel(learner.hyperparameters, len(learner.naming_data.all_tokens_dictionary),
                             len(learner.naming_data.name_dictionary), learner.naming_data.name_empirical_dist)
        learner.model.restore_parameters(learner.parameters)
        return learner

    def get_attention_vector(self, name_cx, code_toks):
        attention_vector = self.model.attention_weights(name_cx,
                                                                 code_toks.astype(np.int32))
        return attention_vector

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

    model = ConvolutionalAttentionalLearner(params)
    model.train(input_file, max_epochs=num_epochs)

    test_data, original_names = model.naming_data.get_data_in_convolution_format(test_file, model.name_cx_size, model.padding_size)
    test_name_targets, test_name_contexts, test_code_sentences, test_original_name_ids = test_data
    ids, unique_idx = np.unique(test_original_name_ids, return_index=True)
    eval = F1Evaluator(model)
    point_suggestion_eval = eval.compute_names_f1(test_code_sentences[unique_idx], original_names,
                                                  model2.naming_data.all_tokens_dictionary.get_all_names())
    return -point_suggestion_eval.get_f1_at_all_ranks()[1]


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
        "conv_layer1_nfilters": 64,
        "conv_layer2_nfilters": 16,
        "layer1_window_size": 6,
        "layer2_window_size": 15,
        "layer3_window_size": 14,
        "log_code_rep_init_scale": -1.34,
        "log_name_rep_init_scale": -4.9,
        "log_layer1_init_scale": -1,
        "log_layer2_init_scale": -3.4,
        "log_layer3_init_scale": -1.8,
        "log_name_cx_init_scale": -1.3,
        "log_learning_rate": -2.95,
        "rmsprop_rho": .98,
        "momentum": 0.9,
        "dropout_rate": 0.25,
        "grad_clip":1
    }


    params["train_file"] = input_file
    params["test_file"] = sys.argv[4]
    with ExperimentLogger("ConvolutionalAttentionalLearner", params) as experiment_log:
        model = ConvolutionalAttentionalLearner(params)

        model.train(input_file, max_epochs=max_num_epochs)

        model.save("convolutional_att_model" + os.path.basename(params["train_file"]) + ".pkl")

        model2 = ConvolutionalAttentionalLearner.load("convolutional_att_model" + os.path.basename(params["train_file"]) + ".pkl")

        test_data, original_names = model2.naming_data.get_data_in_convolution_format(sys.argv[4], model2.name_cx_size, model2.padding_size)
        test_name_targets, test_name_contexts, test_code_sentences, test_original_name_ids = test_data
        name_ll = model2.model.log_prob_with_targets(test_name_contexts, test_code_sentences, test_name_targets)
        print "Test name_ll=%s" % name_ll

        ids, unique_idx = np.unique(test_original_name_ids, return_index=True)
        eval = F1Evaluator(model2)
        point_suggestion_eval = eval.compute_names_f1(test_code_sentences[unique_idx], original_names,
                                                      model2.naming_data.all_tokens_dictionary.get_all_names())
        print point_suggestion_eval
        results = point_suggestion_eval.get_f1_at_all_ranks()
        print results
        experiment_log.record_results({"f1_at_rank1": results[0], "f1_at_rank5":results[1]})
