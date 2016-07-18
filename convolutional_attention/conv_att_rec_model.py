import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from theanoutils.optimization import nesterov_rmsprop_multiple, dropout_multiple
floatX = theano.config.floatX


class ConvolutionalAttentionalRecurrentModel(object):
    def __init__(self, hyperparameters, all_voc_size, empirical_name_dist):
        self.D = hyperparameters["D"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size

        self.__init_parameter(empirical_name_dist)

    def __init_parameter(self, empirical_name_dist):
        all_name_rep = np.random.randn(self.all_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.all_name_reps = theano.shared(all_name_rep.astype(floatX), name="all_name_reps")

        conv_layer1_code = np.random.randn(self.hyperparameters["conv_layer1_nfilters"], 1,
                                     self.hyperparameters["layer1_window_size"], self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_code = theano.shared(conv_layer1_code.astype(floatX), name="conv_layer1_code")
        conv_layer1_bias = np.random.randn(self.hyperparameters["conv_layer1_nfilters"]) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_bias = theano.shared(conv_layer1_bias.astype(floatX), name="conv_layer1_bias")

        conv_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"],
                                     self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_code = theano.shared(conv_layer2_code.astype(floatX), name="conv_layer2_code")

        # GRU parameters
        gru_prev_hidden_to_next = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_next = theano.shared(gru_prev_hidden_to_next.astype(floatX), name="gru_prev_hidden_to_next")
        gru_prev_hidden_to_reset = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_reset = theano.shared(gru_prev_hidden_to_reset.astype(floatX), name="gru_prev_hidden_to_reset")
        gru_prev_hidden_to_update = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer2_nfilters"])\
                                * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prev_hidden_to_update = theano.shared(gru_prev_hidden_to_update.astype(floatX), name="gru_prev_hidden_to_update")

        gru_prediction_to_reset = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_reset = theano.shared(gru_prediction_to_reset.astype(floatX), name="gru_prediction_to_reset")

        gru_prediction_to_update = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_update = theano.shared(gru_prediction_to_update.astype(floatX), name="gru_prediction_to_update")

        gru_prediction_to_hidden = np.random.randn(self.D, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_hidden_init_scale"]
        self.gru_prediction_to_hidden = theano.shared(gru_prediction_to_hidden.astype(floatX), name="gru_prediction_to_hidden")

        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias.astype(floatX), name="conv_layer2_bias")

        gru_hidden_update_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_hidden_update_bias = theano.shared(gru_hidden_update_bias.astype(floatX), name="gru_hidden_update_bias")
        gru_update_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_update_bias = theano.shared(gru_update_bias.astype(floatX), name="gru_update_bias")
        gru_reset_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gru_reset_bias = theano.shared(gru_reset_bias.astype(floatX), name="gru_reset_bias")

        h0 = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.h0 = theano.shared(h0.astype(floatX), name="h0")

        # Currently conflate all to one dimension
        conv_layer3_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_code = theano.shared(conv_layer3_code.astype(floatX), name="conv_layer3_code")
        conv_layer3_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_bias = theano.shared(conv_layer3_bias[0].astype(floatX), name="conv_layer3_bias")

        # Names
        self.name_bias = theano.shared(np.log(empirical_name_dist).astype(floatX)[:-1], name="name_bias")

        self.rng = RandomStreams()

        self.train_parameters = [self.all_name_reps,
                                 self.h0,
                                 self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update,
                                 self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update,
                                 self.gru_hidden_update_bias, self.gru_update_bias, self.gru_reset_bias,
                                 self.conv_layer1_code, self.conv_layer1_bias,
                                 self.conv_layer2_code, self.conv_layer2_bias, self.conv_layer3_code, self.conv_layer3_bias,
                                 self.name_bias,]

        self.__compile_model_functions()

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D",
                      "log_name_rep_init_scale",
                      "conv_layer1_nfilters", "layer1_window_size", "log_layer1_init_scale",
                      "conv_layer2_nfilters", "layer2_window_size", "log_layer2_init_scale",
                      "layer3_window_size", "log_layer3_init_scale",
                      "log_hidden_init_scale",
                      "log_learning_rate", "momentum", "rmsprop_rho", "dropout_rate", "grad_clip"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def restore_parameters(self, values):
        for value, param in zip(values, self.train_parameters):
            param.set_value(value)
        self.__compile_model_functions()

    def __get_model_likelihood_for_sentence(self, sentence, name_targets, do_dropout=False, dropout_rate=0.5):
        code_embeddings = self.all_name_reps[sentence]  # SentSize x D

        if do_dropout:
            # Convolutional weights
            conv_weights_code_l1, conv_weights_code_l2, conv_weights_code_l3  \
                = dropout_multiple(dropout_rate, self.rng, self.conv_layer1_code, self.conv_layer2_code, self.conv_layer3_code)

            # GRU
            gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update, \
                gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update = \
                dropout_multiple(dropout_rate, self.rng,
                             self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update,
                             self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update)

        else:
            conv_weights_code_l1 = self.conv_layer1_code
            conv_weights_code_l2 = self.conv_layer2_code
            conv_weights_code_l3 = self.conv_layer3_code

            gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update,\
                gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update = \
                    self.gru_prediction_to_reset, self.gru_prediction_to_hidden, self.gru_prediction_to_update, \
                             self.gru_prev_hidden_to_reset, self.gru_prev_hidden_to_next, self.gru_prev_hidden_to_update


        code_convolved_l1 = T.nnet.conv2d(code_embeddings.dimshuffle('x', 'x', 0, 1), conv_weights_code_l1, image_shape=(1, 1, None, self.D),
                                          filter_shape=self.conv_layer1_code.get_value().shape)
        l1_out = code_convolved_l1 + self.conv_layer1_bias.dimshuffle('x', 0, 'x', 'x')
        l1_out = T.switch(l1_out>0, l1_out, 0.1 * l1_out)

        code_convolved_l2 = T.nnet.conv2d(l1_out, conv_weights_code_l2, image_shape=(1, self.hyperparameters["conv_layer1_nfilters"], None, 1),
                                          filter_shape=self.conv_layer2_code.get_value().shape)
        l2_out = code_convolved_l2 + self.conv_layer2_bias.dimshuffle('x', 0, 'x', 'x')

        def step(target_token_id, hidden_state, conv_out,
                 gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update,
                 gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update,
                 gru_hidden_update_bias, gru_update_bias, gru_reset_bias,
                 conv_weights_code_l3, conv_layer3_bias, code_embeddings, all_name_reps, use_prev_stat):

            gated_l2 = conv_out * T.switch(hidden_state>0, hidden_state, 0.01 * hidden_state).dimshuffle(0, 1, 'x', 'x')
            gated_l2 = gated_l2 / gated_l2.norm(2)

            code_convolved_l3 = T.nnet.conv2d(gated_l2, conv_weights_code_l3,
                                              image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                              filter_shape=self.conv_layer3_code.get_value().shape)[:, 0, :, 0]

            l3_out = code_convolved_l3 + conv_layer3_bias
            code_toks_weights = T.nnet.softmax(l3_out)  # This should be one dimension (the size of the sentence)

            # the first/last tokens are padding
            padding_size = T.constant(self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3)

            predicted_embedding = T.tensordot(code_toks_weights, code_embeddings[padding_size/2 + 1:-padding_size/2 + 1], [[1], [0]])[0]

            # Get the next hidden!
            if do_dropout:
                # For regularization, we can use the context embeddings *some* of the time
                embedding_used = T.switch(use_prev_stat, all_name_reps[target_token_id], predicted_embedding)
            else:
                embedding_used = all_name_reps[target_token_id]

            reset_gate = T.nnet.sigmoid(
                T.dot(embedding_used, gru_prediction_to_reset) + T.dot(hidden_state, gru_prev_hidden_to_reset) + gru_reset_bias
            )
            update_gate = T.nnet.sigmoid(
                T.dot(embedding_used, gru_prediction_to_update) + T.dot(hidden_state, gru_prev_hidden_to_update) + gru_update_bias
            )
            hidden_update = T.tanh(
                T.dot(embedding_used, gru_prediction_to_hidden) + reset_gate * T.dot(hidden_state, gru_prev_hidden_to_next) + gru_hidden_update_bias
            )

            next_hidden = (1. - update_gate) * hidden_state + update_gate * hidden_update
            return next_hidden, predicted_embedding, code_toks_weights


        use_prev_stat = self.rng.binomial(n=1, p=1.-dropout_rate)
        non_sequences = [l2_out,
                         gru_prediction_to_reset, gru_prediction_to_hidden, gru_prediction_to_update,
                         gru_prev_hidden_to_reset, gru_prev_hidden_to_next, gru_prev_hidden_to_update,
                         self.gru_hidden_update_bias, self.gru_update_bias, self.gru_reset_bias,
                         conv_weights_code_l3, self.conv_layer3_bias,
                         code_embeddings, self.all_name_reps, use_prev_stat]
        [h, predictions, token_weights], _ = theano.scan(step, sequences=name_targets, outputs_info=[self.h0, None, None],
                                          name="target_name_scan", non_sequences=non_sequences,
                                          strict=True)

        name_log_probs = T.log(T.nnet.softmax(T.dot(predictions, T.transpose(self.all_name_reps[:-1])) + self.name_bias)) # SxD, DxK -> SxK

        avg_target_name_log_prob = T.mean(name_log_probs[T.arange(name_targets.shape[0] - 1), name_targets[1:]]) # avg ll

        return sentence, name_log_probs, avg_target_name_log_prob, token_weights

    def __compile_model_functions(self):
            grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(floatX)) for param in self.train_parameters] \
                        + [theano.shared(np.zeros(1,dtype=floatX)[0], name="sentence_count")]

            sentence = T.ivector("sentence")
            name_targets = T.ivector("name_targets")
            _, _, targets_log_prob, _ \
                    = self.__get_model_likelihood_for_sentence(sentence, name_targets, do_dropout=True,
                                                           dropout_rate=self.hyperparameters["dropout_rate"])
            grad = T.grad(targets_log_prob, self.train_parameters)
            outs = grad + [targets_log_prob]
            self.grad_accumulate = theano.function(inputs=[sentence, name_targets],
                                                   updates=[(v, v+g) for v, g in zip(grad_acc, grad)] + [(grad_acc[-1], grad_acc[-1]+1)],
                                                   outputs=outs)



            normalized_grads = [T.switch(grad_acc[-1]>0 ,g / grad_acc[-1], g) for g in grad_acc[:-1]]
            step_updates, ratios = nesterov_rmsprop_multiple(self.train_parameters, normalized_grads,
                                                    learning_rate=10 ** self.hyperparameters["log_learning_rate"],
                                                    rho=self.hyperparameters["rmsprop_rho"],
                                                    momentum=self.hyperparameters["momentum"],
                                                    grad_clip=self.hyperparameters["grad_clip"],
                                                    output_ratios=True)
            step_updates.extend([(v, T.zeros(v.shape,dtype=floatX)) for v in grad_acc[:-1]])  # Set accumulators to 0
            step_updates.append((grad_acc[-1],  T.zeros(1,dtype=floatX)))

            self.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)



            test_name_targets = T.ivector("name_targets")
            test_sentence, test_name_log_probs, test_targets_log_probs, test_att_weights \
                    = self.__get_model_likelihood_for_sentence(T.ivector("sentence"), test_name_targets, do_dropout=False)

            self.__log_prob_with_targets = theano.function(inputs=[test_sentence, test_name_targets],
                                                         outputs=test_targets_log_probs)

            self.__log_prob_last = theano.function(inputs=[test_sentence, test_name_targets],
                                                         outputs=test_name_log_probs[-1])

            self.attention_weights = theano.function(inputs=[test_sentence, test_name_targets],
                                                     outputs=test_att_weights[-1])

    def log_prob_with_targets(self, sentence, name_targets):
        ll = 0
        for i in xrange(len(name_targets)):
            ll += self.__log_prob_with_targets(sentence[i], name_targets[i])
        return (ll / len(name_targets))

    def log_prob(self, name_contexts, sentence):
        ll = []
        for i in xrange(len(sentence)):
            log_probs = self.__log_prob_last(sentence[i], name_contexts[i])
            ll.append(log_probs)
        return np.array(ll)
