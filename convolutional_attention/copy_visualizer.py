from convolutional_attention.copy_learner import CopyAttentionalLearner
import json
import sys

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "Usage <model.pkl> <dataset.json> <output.json>"
        sys.exit(-1)

    learner = CopyAttentionalLearner.load(sys.argv[1])
    dataset = sys.argv[2]

    data, original_names = learner.naming_data.get_data_in_copy_convolution_format(dataset, learner.name_cx_size, learner.padding_size)
    name_targets, original_targets, name_contexts, code_sentences, code, copy_vectors, target_is_unk, original_name_ids = data

    assert len(name_targets) == len(original_name_ids) == len(code_sentences)

    all_data = []
    for i in xrange(len(name_targets)):
        suggestion_data = {}
        suggestion_data["original_name"] = original_names[original_name_ids[i]]
        suggestion_data["target subtoken"] = original_targets[i]

        copy_weights, copy_prob, subtoken_probs = learner.model.copy_probs(name_contexts[i], code_sentences[i])

        _, _, sorted_subtokens, suggestions = learner.get_suggestions_for_next_subtoken(code[i], code_sentences[i], name_contexts[i])
        suggestion_data["suggestions"] = {k: suggestions[k] for k in sorted_subtokens[:20]}

        suggestion_data["att_vector"] = [p for p in learner.model.attention_weights(name_contexts[i], code_sentences[i])[0]]
        suggestion_data["copy_vector"] = [p for p in copy_weights]
        suggestion_data["copy_prob"] = float(copy_prob)
        suggestion_data["tokens"] = code[i]
        suggestion_data["is_unk"] = [learner.naming_data.all_tokens_dictionary.is_unk(t) for t in code[i]]
        all_data.append(suggestion_data)

    with open(sys.argv[3], 'w') as f:
        json.dump(all_data, f)

