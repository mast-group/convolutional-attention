from convolutional_attention.conv_attentional_learner import ConvolutionalAttentionalLearner
import numpy as np
import sys
import json

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage <model.pkl> <dataset.json> <output.json>"
        sys.exit(-1)

    learner = ConvolutionalAttentionalLearner.load(sys.argv[1])
    dataset = sys.argv[2]

    data, original_names = learner.naming_data.get_data_in_convolution_format(dataset, learner.name_cx_size, learner.padding_size)
    name_targets, name_contexts, code_sentences, original_name_ids = data

    assert len(name_targets) == len(original_name_ids) == len(code_sentences)

    all_data = []
    for i in xrange(len(name_targets)):
        suggestion_data = {}
        suggestion_data["original_name"] = original_names[original_name_ids[i]]
        suggestion_data["target subtoken"] = learner.naming_data.all_tokens_dictionary.get_name_for_id(name_targets[i])

        subtoken_probs = learner.model.log_prob(name_contexts[[i]], [code_sentences[i]])[0]
        suggestion_data["suggestions"] = {learner.naming_data.all_tokens_dictionary.get_name_for_id(j): np.exp(subtoken_probs[j]) for j in np.argsort(subtoken_probs)[-20:][::-1]}

        suggestion_data["att_vector"] = [p for p in learner.get_attention_vector(name_contexts[i], code_sentences[i])]
        suggestion_data["tokens"] = [learner.naming_data.all_tokens_dictionary.get_name_for_id(c) for c in code_sentences[i]][learner.padding_size/2: -learner.padding_size/2]
        all_data.append(suggestion_data)

    with open(sys.argv[3], 'w') as f:
        json.dump(all_data, f)

