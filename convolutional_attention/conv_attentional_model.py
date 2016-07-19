import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from theanoutils.optimization import nesterov_rmsprop_multiple
floatX = theano.config.floatX


class ConvolutionalAttentionalModel(object):
    def __init__(self, hyperparameters, all_voc_size, name_voc_size,
                 empirical_name_dist):
        self.D = hyperparameters["D"]
        self.name_cx_size = hyperparameters["name_cx_size"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size
        self.name_voc_size = name_voc_size

        self.__init_parameter(empirical_name_dist)

    def __init_parameter(self, empirical_name_dist):
        all_name_rep = np.random.randn(self.all_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.all_name_reps = theano.shared(all_name_rep.astype(floatX), name="all_name_reps")

        name_cx_rep = np.random.randn(self.name_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.name_cx_reps = theano.shared(name_cx_rep.astype(floatX), name="name_cx_rep")

        conv_layer1_code = np.random.randn(self.hyperparameters["conv_layer1_nfilters"], 1,
                                     self.hyperparameters["layer1_window_size"], self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_code = theano.shared(conv_layer1_code.astype(floatX), name="conv_layer1_code")
        conv_layer1_bias = np.random.randn(self.hyperparameters["conv_layer1_nfilters"]) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_bias = theano.shared(conv_layer1_bias.astype(floatX), name="conv_layer1_bias")

        conv_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"],
                                     self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_code = theano.shared(conv_layer2_code.astype(floatX), name="conv_layer2_code")
        gate_layer2_code = np.random.randn(self.hyperparameters["conv_layer2_nfilters"], self.hyperparameters["conv_layer1_nfilters"],
                                     self.hyperparameters["layer2_window_size"], 1) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gate_weights_code_l2 = theano.shared(gate_layer2_code.astype(floatX), name="gate_weights_code_l2")
        conv_layer2_name_cx = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"], self.name_cx_size, self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer2_name_cx = theano.shared(conv_layer2_name_cx.astype(floatX), name="conv_layer2_name_cx")

        conv_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.conv_layer2_bias = theano.shared(conv_layer2_bias.astype(floatX), name="conv_layer2_bias")
        gate_layer2_bias = np.random.randn(self.hyperparameters["conv_layer2_nfilters"]) * 10 ** self.hyperparameters["log_layer2_init_scale"]
        self.gate_layer2_bias = theano.shared(gate_layer2_bias.astype(floatX), name="gate_layer2_bias")

        # Currently conflate all to one dimension
        conv_layer3_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_code = theano.shared(conv_layer3_code.astype(floatX), name="conv_layer3_code")
        conv_layer3_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_bias = theano.shared(conv_layer3_bias[0].astype(floatX), name="conv_layer3_bias")

        # Names
        self.name_bias = theano.shared(np.log(empirical_name_dist).astype(floatX)[:-1], name="name_bias")

        self.rng = RandomStreams()

        self.train_parameters = [self.all_name_reps, self.name_cx_reps,
                                 self.conv_layer1_code, self.conv_layer2_name_cx, self.conv_layer1_bias,
                                 self.conv_layer2_code, self.conv_layer2_bias, self.conv_layer3_code, self.conv_layer3_bias,
                                 self.name_bias, self.gate_layer2_bias, self.gate_weights_code_l2]

        self.__compile_model_functions()

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D", "name_cx_size",
                      "log_code_rep_init_scale", "log_name_rep_init_scale",
                      "conv_layer1_nfilters", "layer1_window_size", "log_layer1_init_scale",
                      "conv_layer2_nfilters", "layer2_window_size", "log_layer2_init_scale",
                      "layer3_window_size", "log_layer3_init_scale",
                      "log_name_cx_init_scale",
                      "log_learning_rate", "momentum", "rmsprop_rho", "dropout_rate", "grad_clip"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def restore_parameters(self, values):
        for value, param in zip(values, self.train_parameters):
            param.set_value(value)
        self.__compile_model_functions()

    def __get_model_likelihood_for_sentence(self, sentence, name_contexts, name_target, do_dropout=False, dropout_rate=0.5):
        #theano.config.compute_test_value = 'warn'
        #sentence.tag.test_value = np.arange(100).astype(np.int32)
        #name_contexts.tag.test_value = np.arange(self.name_cx_size).astype(np.int32)
        #name_target.tag.test_value = 5

        name_reps = self.name_cx_reps[name_contexts] # CxSize x D
        code_embeddings = self.all_name_reps[sentence] # SentSize x D

        if do_dropout:
            mask = self.rng.binomial(code_embeddings.shape, p=1.-dropout_rate, dtype=code_embeddings.dtype)
            code_embeddings *= mask / (1. - dropout_rate)

            conv_mask_code_l1 = self.rng.binomial(self.conv_layer1_code.shape, p=1.-dropout_rate,
                                          dtype=self.conv_layer1_code.dtype)
            conv_weights_code_l1 = self.conv_layer1_code * conv_mask_code_l1 / (1. - dropout_rate)
            conv_mask_name_l2 = self.rng.binomial(self.conv_layer2_name_cx.shape, p=1.-dropout_rate,
                                          dtype=self.conv_layer2_name_cx.dtype)
            conv_weights_name_l2 = self.conv_layer2_name_cx * conv_mask_name_l2 / (1. - dropout_rate)


            conv_mask_code_l2 = self.rng.binomial(self.conv_layer2_code.shape, p=1.-dropout_rate,
                                          dtype=self.conv_layer2_code.dtype)
            conv_weights_code_l2 = self.conv_layer2_code * conv_mask_code_l2 / (1. - dropout_rate)

            gate_mask_code_l2 = self.rng.binomial(self.gate_weights_code_l2.shape, p=1.-dropout_rate,
                                          dtype=self.gate_weights_code_l2.dtype)
            gate_weights_code_l2 = self.gate_weights_code_l2 * gate_mask_code_l2 / (1. - dropout_rate)

            conv_mask_code_l3 = self.rng.binomial(self.conv_layer3_code.shape, p=1.-dropout_rate,
                                          dtype=self.conv_layer3_code.dtype)
            conv_weights_code_l3 = self.conv_layer3_code * conv_mask_code_l3 / (1. - dropout_rate)

        else:
            conv_weights_code_l1 = self.conv_layer1_code
            conv_weights_name_l2 = self.conv_layer2_name_cx
            gate_weights_code_l2 = self.gate_weights_code_l2
            conv_weights_code_l2 = self.conv_layer2_code
            conv_weights_code_l3 = self.conv_layer3_code



        code_convolved_l1 = T.nnet.conv2d(code_embeddings.dimshuffle('x', 'x', 0, 1), conv_weights_code_l1, image_shape=(1, 1, None, self.D),
                                          filter_shape=self.conv_layer1_code.get_value().shape)
        l1_out = code_convolved_l1 + self.conv_layer1_bias.dimshuffle('x', 0, 'x', 'x')
        l1_out = T.switch(l1_out>0, l1_out, 0.1 * l1_out)

        code_convolved_l2 = T.nnet.conv2d(l1_out, conv_weights_code_l2, image_shape=(1, self.hyperparameters["conv_layer1_nfilters"], None, 1),
                                          filter_shape=self.conv_layer2_code.get_value().shape)
        code_gate_l2 = T.nnet.conv2d(l1_out, gate_weights_code_l2, image_shape=(1, self.hyperparameters["conv_layer1_nfilters"], None, 1),
                                          filter_shape=self.conv_layer2_code.get_value().shape)

        name_gate_l2 = T.tensordot(name_reps, conv_weights_name_l2, [[0, 1], [2, 3]]).dimshuffle(0, 1, 'x', 'x')

        l2_out = code_convolved_l2 + self.conv_layer2_bias.dimshuffle('x', 0, 'x', 'x')
        gate_val = name_gate_l2 + code_gate_l2 + self.gate_layer2_bias.dimshuffle('x', 0, 'x', 'x')
        l2_out *= T.switch(gate_val>0, gate_val, 0.01 * gate_val)
        l2_out = l2_out / l2_out.norm(2)


        code_convolved_l3 = T.nnet.conv2d(l2_out, conv_weights_code_l3,
                                          image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                          filter_shape=self.conv_layer3_code.get_value().shape)[:, 0, :, 0]

        l3_out = code_convolved_l3 + self.conv_layer3_bias
        code_toks_weights = T.nnet.softmax(l3_out)  # This should be one dimension (the size of the sentence)

        # the first/last tokens are padding
        padding_size = T.constant(self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3)

        name_context_with_code_data = T.tensordot(code_toks_weights, code_embeddings[padding_size/2 + 1:-padding_size/2 + 1], [[1], [0]])
        name_log_prob = T.log(T.nnet.softmax(T.dot(name_context_with_code_data, T.transpose(self.all_name_reps[:-1])) + self.name_bias))

        target_name_log_probs = name_log_prob[0, name_target]
        return sentence, name_contexts, name_target, name_log_prob, target_name_log_probs, name_context_with_code_data, code_toks_weights[0]

    def __compile_model_functions(self):
            grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(floatX)) for param in self.train_parameters] \
                        + [theano.shared(np.zeros(1,dtype=floatX)[0], name="sentence_count")]

            sentence = T.ivector("sentence")
            name_context = T.ivector("name_context")
            name_target = T.iscalar("name_target")
            _, _, _, _, targets_log_prob, _, _ \
                    = self.__get_model_likelihood_for_sentence(sentence, name_context, name_target, do_dropout=True,
                                                           dropout_rate=self.hyperparameters["dropout_rate"])
            grad = T.grad(targets_log_prob, self.train_parameters)
            outs = grad + [targets_log_prob]
            self.grad_accumulate = theano.function(inputs=[name_context, sentence, name_target],
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
            step_updates.append((grad_acc[-1], T.zeros(1,dtype=floatX)))

            self.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)




            test_sentence, test_name_contexts, test_name_targets, test_name_log_prob, test_targets_log_probs, \
            test_code_representation, test_code_weights = self.__get_model_likelihood_for_sentence(
                T.ivector("sentence"),  T.ivector("name_context"), T.iscalar("name_target"), do_dropout=False)

            self.__log_prob_with_targets = theano.function(inputs=[test_name_contexts, test_sentence, test_name_targets],
                                                         outputs=test_targets_log_probs)

            self.__log_prob = theano.function(inputs=[test_name_contexts, test_sentence],
                                                         outputs=test_name_log_prob)

            self.attention_weights = theano.function(inputs=[test_name_contexts, test_sentence],
                                                     outputs=test_code_weights)

            self.get_representation = theano.function(inputs=[test_name_contexts, test_sentence], outputs=test_code_representation)

    def log_prob_with_targets(self, name_contexts, sentence, name_targets):
        ll = 0
        for i in xrange(len(name_targets)):
            ll += self.__log_prob_with_targets(name_contexts[i], sentence[i].astype(np.int32),
                                               name_targets[i])
        return (ll / len(name_targets))

    def log_prob(self, name_contexts, sentence):
        ll = []
        for i in xrange(len(sentence)):
            log_probs = self.__log_prob(name_contexts[i], sentence[i])[0]
            #assert len(log_probs)==self.name_voc_size-1, (log_probs.shape, self.name_voc_size)
            ll.append(log_probs)
        return np.array(ll)
