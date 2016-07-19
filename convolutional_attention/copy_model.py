import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

from theanoutils.optimization import nesterov_rmsprop_multiple, logsumexp
floatX = theano.config.floatX


class CopyConvolutionalAttentionalModel(object):
    def __init__(self, hyperparameters, all_voc_size, name_voc_size, empirical_name_dist):
        self.D = hyperparameters["D"]
        self.name_cx_size = hyperparameters["name_cx_size"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size
        self.name_voc_size = name_voc_size

        self.__init_parameter(empirical_name_dist)

    def __init_parameter(self, empirical_name_dist):
        all_name_rep = np.random.randn(self.all_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.all_name_reps = theano.shared(all_name_rep.astype(floatX), name="code_name_reps")

        name_cx_rep = np.random.randn(self.name_voc_size, self.D) * 10 ** self.hyperparameters["log_name_rep_init_scale"]
        self.name_cx_reps = theano.shared(name_cx_rep.astype(floatX), name="name_cx_rep")

        # By convention, the last one is NONE, which is never predicted.
        self.name_bias = theano.shared(np.log(empirical_name_dist).astype(floatX)[:-1], name="name_bias")

        conv_layer1_code = np.random.randn(self.hyperparameters["conv_layer1_nfilters"], 1,
                                     self.hyperparameters["layer1_window_size"], self.D) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_code = theano.shared(conv_layer1_code.astype(floatX), name="conv_layer1_code")
        conv_layer1_bias = np.random.randn(self.hyperparameters["conv_layer1_nfilters"]) * 10 ** self.hyperparameters["log_layer1_init_scale"]
        self.conv_layer1_bias = theano.shared(conv_layer1_bias.astype(floatX), name="conv_layer1_bias")

        # Currently conflate all to one dimension
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

        # Probability that each token will be copied
        conv_layer3_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_copy_code = theano.shared(conv_layer3_code.astype(floatX), name="conv_layer3_copy_code")
        conv_layer3_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_copy_bias = theano.shared(conv_layer3_bias[0].astype(floatX), name="conv_layer3_copy_bias")


        # Probability that we do a copy
        conv_copy_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.conv_copy_code = theano.shared(conv_copy_code.astype(floatX), name="conv_copy_code")

        conv_copy_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.conv_copy_bias = theano.shared(conv_copy_bias[0].astype(floatX), name="conv_copy_bias")

        conv_copy_name_cx = np.random.randn(self.name_cx_size, self.D) * 10 ** self.hyperparameters["log_copy_init_scale"]
        self.copy_name_cx = theano.shared(conv_copy_name_cx.astype(floatX), name="conv_copy_name_cx")

        # Attention vectors
        conv_layer3_att_code = np.random.randn(1, self.hyperparameters["conv_layer2_nfilters"],
                                     self.hyperparameters["layer3_window_size"], 1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_att_code = theano.shared(conv_layer3_att_code.astype(floatX), name="conv_layer3_att_code")
        conv_att_name_cx = np.random.randn(self.name_cx_size, self.D) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.att_name_cx = theano.shared(conv_att_name_cx.astype(floatX), name="conv_att_name_cx")
        conv_layer3_att_bias = np.random.randn(1) * 10 ** self.hyperparameters["log_layer3_init_scale"]
        self.conv_layer3_att_bias = theano.shared(conv_layer3_att_bias[0].astype(floatX), name="conv_layer3_att_bias")


        self.rng = RandomStreams()
        self.padding_size = self.hyperparameters["layer1_window_size"] + self.hyperparameters["layer2_window_size"] + self.hyperparameters["layer3_window_size"] - 3

        self.train_parameters = [self.all_name_reps, self.name_cx_reps,
                                 self.conv_layer1_code, self.conv_layer2_name_cx, self.conv_layer1_bias,
                                 self.conv_layer2_code, self.conv_layer2_bias,
                                 self.conv_layer3_copy_code, self.conv_layer3_copy_bias,
                                 self.gate_layer2_bias, self.gate_weights_code_l2,
                                 self.conv_copy_code, self.conv_copy_bias, self.copy_name_cx,
                                 self.conv_layer3_att_code, self.att_name_cx, self.conv_layer3_att_bias,
                                 self.name_bias]

        self.__compile_model_functions()

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D", "name_cx_size",
                      "log_code_rep_init_scale", "log_name_rep_init_scale",
                      "conv_layer1_nfilters", "layer1_window_size", "log_layer1_init_scale",
                      "conv_layer2_nfilters", "layer2_window_size", "log_layer2_init_scale",
                      "layer3_window_size", "log_layer3_init_scale",
                      "log_copy_init_scale", "log_name_cx_init_scale",
                      "log_learning_rate", "momentum", "rmsprop_rho", "dropout_rate", "grad_clip"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def restore_parameters(self, values):
        for value, param in zip(values, self.train_parameters):
            param.set_value(value)
        self.__compile_model_functions()

    def __get_model_likelihood_for_sentence(self, sentence, name_contexts, do_dropout=False, dropout_rate=0.5):
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

            conv_mask_code_l3 = self.rng.binomial(self.conv_layer3_copy_code.shape, p=1.-dropout_rate,
                                          dtype=self.conv_layer3_copy_code.dtype)
            conv_weights_code_copy_l3 = self.conv_layer3_copy_code * conv_mask_code_l3 / (1. - dropout_rate)

            conv_weights_code_copy_mask = self.rng.binomial(self.conv_copy_code.shape, p=1.-dropout_rate,
                                          dtype=self.conv_copy_code.dtype)
            conv_weights_code_do_copy = self.conv_copy_code * conv_weights_code_copy_mask / (1. - dropout_rate)

            conv_weights_code_att_mask = self.rng.binomial(self.conv_layer3_att_code.shape, p=1.-dropout_rate,
                                          dtype=self.conv_layer3_att_code.dtype)
            conv_weights_code_att_l3 = self.conv_layer3_att_code * conv_weights_code_att_mask / (1. - dropout_rate)

            copy_name_cx_mask = self.rng.binomial(self.copy_name_cx.shape, p=1.-dropout_rate,
                                          dtype=self.copy_name_cx.dtype)
            copy_name_cx = self.copy_name_cx * copy_name_cx_mask / (1. - dropout_rate)

            att_name_cx_mask = self.rng.binomial(self.att_name_cx.shape, p=1.-dropout_rate,
                                          dtype=self.att_name_cx.dtype)
            att_name_cx = self.att_name_cx * att_name_cx_mask / (1. - dropout_rate)

        else:
            conv_weights_code_l1 = self.conv_layer1_code
            conv_weights_name_l2 = self.conv_layer2_name_cx
            gate_weights_code_l2 = self.gate_weights_code_l2
            conv_weights_code_l2 = self.conv_layer2_code
            conv_weights_code_copy_l3 = self.conv_layer3_copy_code
            conv_weights_code_do_copy = self.conv_copy_code
            conv_weights_code_att_l3 = self.conv_layer3_att_code
            copy_name_cx = self.copy_name_cx
            att_name_cx = self.att_name_cx


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
        l2_out *= T.switch(gate_val > 0, gate_val, 0.01 * gate_val)
        l2_out = l2_out / l2_out.norm(2)

        # Do we copy?
        do_copy_code = T.max(T.nnet.conv2d(l2_out, conv_weights_code_do_copy,
                                          image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                          filter_shape=self.conv_copy_code.get_value().shape)[:, 0, :, 0])
        name_do_copy = T.tensordot(name_reps, copy_name_cx, [[0, 1], [0, 1]])

        copy_prob = T.nnet.sigmoid(do_copy_code + name_do_copy + self.conv_copy_bias)

        # Where do we copy?
        code_copy_convolved_l3 = T.nnet.conv2d(l2_out, conv_weights_code_copy_l3,
                                          image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                          filter_shape=self.conv_layer3_copy_code.get_value().shape)[:, 0, :, 0]

        copy_l3_out = code_copy_convolved_l3 + self.conv_layer3_copy_bias
        copy_pos_probs = T.nnet.softmax(copy_l3_out)  # This should be one dimension (the size of the sentence)

        # Attention
        code_att_convolved_l3 = T.nnet.conv2d(l2_out, conv_weights_code_att_l3,
                                          image_shape=(1, self.hyperparameters["conv_layer2_nfilters"], None, 1),
                                          filter_shape=self.conv_layer3_att_code.get_value().shape)[:, 0, :, 0]
        name_att_l3 = T.tensordot(name_reps, att_name_cx, [[0, 1], [0, 1]])

        att_l3_out = code_att_convolved_l3 + name_att_l3 + self.conv_layer3_att_bias
        attention_weights = T.nnet.softmax(att_l3_out)  # This should be one dimension (the size of the sentence)

        # the first/last tokens are padding
        name_context_with_code_data = T.tensordot(attention_weights, code_embeddings[self.padding_size/2 + 1:-self.padding_size/2 + 1], [[1], [0]])
        # By convention, the last one in all_name_reps is NONE, which is never predicted.
        name_log_probs = T.log(T.nnet.softmax(T.dot(name_context_with_code_data, T.transpose(self.all_name_reps[:-1])) + self.name_bias))

        return sentence, name_contexts, copy_pos_probs[0], copy_prob, name_log_probs[0], attention_weights

    def model_objective(self, copy_prob, copy_weights, is_copy_vector, name_log_probs, name_target, target_is_unk):
        # if there is at least one position to copy from, then we should
        use_copy_prob = T.switch(T.sum(is_copy_vector) > 0, T.log(copy_prob) + T.log(T.sum(is_copy_vector * copy_weights)+10e-8), -1000)
        use_model_prob = T.switch(target_is_unk, -10, 0) + T.log(1. - copy_prob) + name_log_probs[name_target]
        correct_answer_log_prob = logsumexp(use_copy_prob, use_model_prob)
        return correct_answer_log_prob

    def __compile_model_functions(self):
            grad_acc = [theano.shared(np.zeros(param.get_value().shape).astype(floatX)) for param in self.train_parameters] \
                        + [theano.shared(np.float32(0), name="sentence_count")]

            sentence = T.ivector("sentence")
            is_copy_vector = T.ivector("is_copy_vector")
            name_context = T.ivector("name_context")
            name_target = T.iscalar("name_target")
            target_is_unk = T.iscalar("target_is_unk")

            #theano.config.compute_test_value = 'warn'
            sentence.tag.test_value = np.arange(100).astype(np.int32)
            is_copy_test_value = [i % 7 == 2 for i in xrange(100 - self.padding_size)]
            is_copy_vector.tag.test_value = np.array(is_copy_test_value, dtype=np.int32)
            name_context.tag.test_value = np.arange(self.name_cx_size).astype(np.int32)
            name_target.tag.test_value = 5
            target_is_unk.tag.test_value = 0

            _, _, copy_weights, copy_prob, name_log_probs, _ \
                    = self.__get_model_likelihood_for_sentence(sentence, name_context, do_dropout=True,
                                                           dropout_rate=self.hyperparameters["dropout_rate"])

            correct_answer_log_prob = self.model_objective(copy_prob, copy_weights, is_copy_vector, name_log_probs,
                                                           name_target, target_is_unk)

            grad = T.grad(correct_answer_log_prob, self.train_parameters)
            self.grad_accumulate = theano.function(inputs=[name_context, sentence, is_copy_vector, target_is_unk, name_target],
                                                   updates=[(v, v+g) for v, g in zip(grad_acc, grad)] + [(grad_acc[-1], grad_acc[-1]+1)],
                                                   #mode='NanGuardMode'
                                                   )



            normalized_grads = [T.switch(grad_acc[-1] >0, g / grad_acc[-1], g) for g in grad_acc[:-1]]
            step_updates, ratios = nesterov_rmsprop_multiple(self.train_parameters, normalized_grads,
                                                    learning_rate=10 ** self.hyperparameters["log_learning_rate"],
                                                    rho=self.hyperparameters["rmsprop_rho"],
                                                    momentum=self.hyperparameters["momentum"],
                                                    grad_clip=self.hyperparameters["grad_clip"],
                                                    output_ratios=True)
            step_updates.extend([(v, T.zeros(v.shape,dtype=floatX)) for v in grad_acc[:-1]])  # Set accumulators to 0
            step_updates.append((grad_acc[-1], T.zeros(1,dtype=floatX)))

            self.grad_step = theano.function(inputs=[], updates=step_updates, outputs=ratios)


            test_sentence, test_name_contexts, test_copy_weights, test_copy_prob, test_name_log_probs, test_attention_weights \
                = self.__get_model_likelihood_for_sentence(T.ivector("sentence"),  T.ivector("name_context"),
                                                          do_dropout=False)

            self.copy_probs = theano.function(inputs=[test_name_contexts, test_sentence],
                                                      outputs=[test_copy_weights, test_copy_prob, test_name_log_probs])
            test_copy_vector = T.ivector("test_copy_vector")
            test_name_target = T.iscalar("test_name_target")
            test_target_is_unk = T.iscalar("test_target_is_unk")
            ll = self.model_objective(test_copy_prob, test_copy_weights, test_copy_vector, test_name_log_probs,
                                                           test_name_target, test_target_is_unk)
            self.copy_logprob = theano.function(inputs=[test_name_contexts, test_sentence, test_copy_vector, test_target_is_unk, test_name_target],
                                                outputs=ll)

            self.attention_weights = theano.function(inputs=[test_name_contexts, test_sentence],
                                                     outputs=test_attention_weights)


    def log_prob_no_predict(self, name_contexts, sentences, copy_vectors, target_is_unk, name_target):
        ll = 0
        for i in xrange(len(sentences)):
            ll += self.copy_logprob(name_contexts[i], sentences[i], copy_vectors[i], target_is_unk[i], name_target[i])
        return (ll / len(sentences))

