from convolutional_attention.conv_att_rec_learner import ConvolutionalAttentionalRecurrentLearner
import json
import numpy as np
import sys

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage <model.pkl> <dataset.json> <output.json>"
        sys.exit(-1)

    learner = ConvolutionalAttentionalRecurrentLearner.load(sys.argv[1])
    dataset = sys.argv[2]

    data, original_names = learner.naming_data.get_data_in_recurrent_convolution_format(dataset, learner.padding_size)
    name_targets, code_sentences = data

    all_data = []
    for i in xrange(len(name_targets)):
        for j in xrange(1, len(name_targets[i])):
            suggestion_data = {}

            current_prefix = name_targets[i][:j]
            suggestion_data["original_name"] = original_names[i]

            if j == len(name_targets[i]) -1:  # END token
                target_subtoken = learner.naming_data.all_tokens_dictionary.get_name_for_id(name_targets[i][j])
            else:
                target_subtoken = original_names[i].split(",")[j-1]
            suggestion_data["target subtoken"] = target_subtoken

            subtoken_probs = learner.model.log_prob(np.atleast_2d(current_prefix), np.atleast_2d(code_sentences[i]))[0]

            suggestion_data["suggestions"] = {learner.naming_data.all_tokens_dictionary.get_name_for_id(j): np.exp(subtoken_probs[j]) for j in np.argsort(subtoken_probs)[-20:][::-1]}

            suggestion_data["tokens"] = [learner.naming_data.all_tokens_dictionary.get_name_for_id(c) for c in code_sentences[i]][learner.padding_size/2+1: -learner.padding_size/2+1]
            suggestion_data["att_vector"] = [p for p in learner.model.attention_weights(code_sentences[i], current_prefix)[-1]]
            all_data.append(suggestion_data)


    with open(sys.argv[3], 'w') as f:
        json.dump(all_data, f)
