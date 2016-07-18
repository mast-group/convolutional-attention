from collections import defaultdict
import heapq
from itertools import chain, repeat
from feature_dict import FeatureDictionary
import json
import numpy as np
import scipy.sparse as sp

class TokenCodeNamingData:

    SUBTOKEN_START = "%START%"
    SUBTOKEN_END = "%END%"
    NONE = "%NONE%"

    @staticmethod
    def __get_file_data(input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)
        # data=[{"tokens":"hello world I am OK".split(),"name":"hello world you".split()}]*4
        # data+=[{"tokens":"just another test of a silly program".split(),"name":"who knows".split()}]*4
        names = []
        original_names = []
        code = []
        for entry in data:
            # skip entries with no relevant data (this will crash the code)
            if len(entry["tokens"]) == 0 or len(entry["name"]) == 0:
                continue
            code.append(TokenCodeNamingData.remove_identifiers_markers(entry["tokens"]))
            original_names.append(",".join(entry["name"]))
            subtokens = entry["name"]
            names.append([TokenCodeNamingData.SUBTOKEN_START] + subtokens + [TokenCodeNamingData.SUBTOKEN_END])

        return names, code, original_names

    def __init__(self, names, code):
        self.name_dictionary = FeatureDictionary.get_feature_dictionary_for(chain.from_iterable(names), 2)
        self.name_dictionary.add_or_get_id(self.NONE)

        self.all_tokens_dictionary = FeatureDictionary.get_feature_dictionary_for(chain.from_iterable(
            [chain.from_iterable(code), chain.from_iterable(names)]), 5)
        self.all_tokens_dictionary.add_or_get_id(self.NONE)
        self.name_empirical_dist = self.__get_empirical_distribution(self.all_tokens_dictionary, chain.from_iterable(names))

    @staticmethod
    def __get_empirical_distribution(element_dict, elements, dirichlet_alpha=10.):
        """
        Retrive te empirical distribution of tokens
        :param element_dict: a dictionary that can convert the elements to their respective ids.
        :param elements: an iterable of all the elements
        :return:
        """
        targets = np.array([element_dict.get_id_or_unk(t) for t in elements])
        empirical_distribution = np.bincount(targets, minlength=len(element_dict)).astype(float)
        empirical_distribution += dirichlet_alpha / len(empirical_distribution)
        return empirical_distribution / (np.sum(empirical_distribution) + dirichlet_alpha)

    def __get_in_lbl_format(self, data, dictionary, cx_size):
        targets = []
        contexts = []
        ids = []

        for i, sequence in enumerate(data):
            for j in xrange(1, len(sequence)): # First element should always be predictable (ie sentence start)
                ids.append(i)
                targets.append(dictionary.get_id_or_unk(sequence[j]))
                context = sequence[:j]
                if len(context) < cx_size:
                    context = [self.NONE] * (cx_size - len(context)) + context
                else:
                    context = context[-cx_size:]
                assert len(context) == cx_size, (len(context), cx_size,)
                contexts.append([dictionary.get_id_or_unk(t) for t in context])
        return np.array(targets, dtype=np.int32), np.array(contexts, dtype=np.int32), np.array(ids, np.int32)

    def get_data_in_lbl_format(self, input_file, code_cx_size, names_cx_size):
        names, code, original_names = self.__get_file_data(input_file)
        return self.__get_in_lbl_format(names, self.name_dictionary, names_cx_size), \
               self.__get_in_lbl_format(code, self.all_tokens_dictionary, code_cx_size), original_names

    @staticmethod
    def get_data_in_lbl_format_with_validation(input_file, code_cx_size, names_cx_size, pct_train):
        assert pct_train < 1
        assert pct_train > 0
        names, code, original_names = TokenCodeNamingData.__get_file_data(input_file)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        original_names = np.array(original_names, dtype=np.object)
        lim = int(pct_train * len(names))
        naming = TokenCodeNamingData(names[:lim], code[:lim])
        return naming.__get_in_lbl_format(names[:lim], naming.name_dictionary, names_cx_size), \
               naming.__get_in_lbl_format(code[:lim], naming.all_tokens_dictionary, code_cx_size), original_names[:lim], \
               naming.__get_in_lbl_format(names[lim:], naming.name_dictionary, names_cx_size), \
               naming.__get_in_lbl_format(code[lim:], naming.all_tokens_dictionary, code_cx_size), original_names[lim:], naming

    @staticmethod
    def get_data_in_forward_format_with_validation(input_file, names_cx_size, pct_train):
        assert pct_train < 1
        assert pct_train > 0
        names, code, original_names = TokenCodeNamingData.__get_file_data(input_file)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        original_names = np.array(original_names, dtype=np.object)
        lim = int(pct_train * len(names))
        naming = TokenCodeNamingData(names[:lim], code[:lim])
        return naming.__get_data_in_forward_format(names[:lim], code[:lim], names_cx_size),\
                naming.__get_data_in_forward_format(names[lim:], code[lim:], names_cx_size), naming

    def get_data_in_forward_format(self, input_file, name_cx_size):
        names, code, original_names = self.__get_file_data(input_file)
        return self.__get_data_in_forward_format(names, code, name_cx_size), original_names


    def __get_data_in_forward_format(self, names, code, name_cx_size):
        """
        Get the data in a "forward" model format.
        :param data:
        :param name_cx_size:
        :return:
        """
        assert len(names) == len(code), (len(names), len(code), code.shape)
        # Keep only identifiers in code
        #code = self.keep_identifiers_only(code)

        name_targets = []
        name_contexts = []
        original_names_ids = []
        id_xs = []
        id_ys = []

        k = 0
        for i, name in enumerate(names):
            for j in xrange(1, len(name)):  # First element should always be predictable (ie sentence start)
                name_targets.append(self.name_dictionary.get_id_or_unk(name[j]))
                original_names_ids.append(i)
                context = name[:j]
                if len(context) < name_cx_size:
                    context = [self.NONE] * (name_cx_size - len(context)) + context
                else:
                    context = context[-name_cx_size:]
                assert len(context) == name_cx_size, (len(context), name_cx_size,)
                name_contexts.append([self.name_dictionary.get_id_or_unk(t) for t in context])
                for code_token in set(code[i]):
                    token_id = self.all_tokens_dictionary.get_id_or_none(code_token)
                    if token_id is not None:
                        id_xs.append(k)
                        id_ys.append(token_id)
                k += 1

        code_features = sp.csr_matrix((np.ones(len(id_xs)), (id_xs, id_ys)), shape=(k, len(self.all_tokens_dictionary)), dtype=np.int32)
        name_targets = np.array(name_targets, dtype=np.int32)
        name_contexts = np.array(name_contexts, dtype=np.int32)
        original_names_ids = np.array(original_names_ids, dtype=np.int32)
        return name_targets, name_contexts, code_features, original_names_ids

    @staticmethod
    def keep_identifiers_only(self, code):
        filtered_code = []
        for tokens in code:
            identifier_tokens = []
            in_id = False
            for t in tokens:
                if t == "<id>":
                    in_id = True
                elif t == '</id>':
                    in_id = False
                elif in_id:
                    identifier_tokens.append(t)
            filtered_code.append(identifier_tokens)
        return filtered_code

    @staticmethod
    def remove_identifiers_markers(code):
        return filter(lambda t: t != "<id>" and t != "</id>", code)

    def get_data_in_convolution_format(self, input_file, name_cx_size, min_code_size):
        names, code, original_names = self.__get_file_data(input_file)
        return self.get_data_for_convolution(names, code, name_cx_size, min_code_size), original_names

    def get_data_in_copy_convolution_format(self, input_file, name_cx_size, min_code_size):
        names, code, original_names = self.__get_file_data(input_file)
        return self.get_data_for_copy_convolution(names, code, name_cx_size, min_code_size), original_names

    def get_data_in_recurrent_convolution_format(self, input_file, min_code_size):
        names, code, original_names = self.__get_file_data(input_file)
        return self.get_data_for_recurrent_convolution(names, code, min_code_size), original_names

    def get_data_in_recurrent_copy_convolution_format(self, input_file, min_code_size):
        names, code, original_names = self.__get_file_data(input_file)
        return self.get_data_for_recurrent_copy_convolution(names, code, min_code_size), original_names

    def get_data_for_convolution(self, names, code, name_cx_size, sentence_padding):
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        name_contexts = []
        original_names_ids = []
        code_sentences = []
        padding = [self.all_tokens_dictionary.get_id_or_unk(self.NONE)]

        for i, name in enumerate(names):
            code_sentence = [self.all_tokens_dictionary.get_id_or_unk(t) for t in code[i]]

            if sentence_padding % 2 == 0:
                code_sentence = padding * (sentence_padding / 2) + code_sentence + padding * (sentence_padding / 2)
            else:
                code_sentence = padding * (sentence_padding / 2 + 1) + code_sentence + padding * (sentence_padding / 2)
            for j in xrange(1, len(name)):  # First element should always be predictable (ie sentence start)
                name_targets.append(self.all_tokens_dictionary.get_id_or_unk(name[j]))
                original_names_ids.append(i)
                context = name[:j]
                if len(context) < name_cx_size:
                    context = [self.NONE] * (name_cx_size - len(context)) + context
                else:
                    context = context[-name_cx_size:]
                assert len(context) == name_cx_size, (len(context), name_cx_size,)
                name_contexts.append([self.name_dictionary.get_id_or_unk(t) for t in context])
                code_sentences.append(np.array(code_sentence, dtype=np.int32))

        name_targets = np.array(name_targets, dtype=np.int32)
        name_contexts = np.array(name_contexts, dtype=np.int32)
        code_sentences = np.array(code_sentences, dtype=np.object)
        original_names_ids = np.array(original_names_ids, dtype=np.int32)
        return name_targets, name_contexts, code_sentences, original_names_ids

    def get_data_for_recurrent_convolution(self, names, code, sentence_padding):
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        code_sentences = []
        padding = [self.all_tokens_dictionary.get_id_or_unk(self.NONE)]

        for i, name in enumerate(names):
            code_sentence = [self.all_tokens_dictionary.get_id_or_unk(t) for t in code[i]]

            if sentence_padding % 2 == 0:
                code_sentence = padding * (sentence_padding / 2) + code_sentence + padding * (sentence_padding / 2)
            else:
                code_sentence = padding * (sentence_padding / 2 + 1) + code_sentence + padding * (sentence_padding / 2)
            name_tokens = [self.all_tokens_dictionary.get_id_or_unk(t) for t in name]

            name_targets.append(np.array(name_tokens, dtype=np.int32))
            code_sentences.append(np.array(code_sentence, dtype=np.int32))

        name_targets = np.array(name_targets, dtype=np.object)
        code_sentences = np.array(code_sentences, dtype=np.object)
        return name_targets, code_sentences

    def get_data_for_recurrent_copy_convolution(self, names, code, sentence_padding):
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        target_is_unk = []
        copy_vectors = []
        code_sentences = []
        padding = [self.all_tokens_dictionary.get_id_or_unk(self.NONE)]

        for i, name in enumerate(names):
            code_sentence = [self.all_tokens_dictionary.get_id_or_unk(t) for t in code[i]]

            if sentence_padding % 2 == 0:
                code_sentence = padding * (sentence_padding / 2) + code_sentence + padding * (sentence_padding / 2)
            else:
                code_sentence = padding * (sentence_padding / 2 + 1) + code_sentence + padding * (sentence_padding / 2)
            name_tokens = [self.all_tokens_dictionary.get_id_or_unk(t) for t in name]
            unk_tokens = [self.all_tokens_dictionary.is_unk(t) for t in name]
            target_can_be_copied = [[t == subtok for t in code[i]] for subtok in name]

            name_targets.append(np.array(name_tokens, dtype=np.int32))
            target_is_unk.append(np.array(unk_tokens, dtype=np.int32))
            copy_vectors.append(np.array(target_can_be_copied, dtype=np.int32))
            code_sentences.append(np.array(code_sentence, dtype=np.int32))

        name_targets = np.array(name_targets, dtype=np.object)
        code_sentences = np.array(code_sentences, dtype=np.object)
        code = np.array(code, dtype=np.object)
        target_is_unk = np.array(target_is_unk, dtype=np.object)
        copy_vectors = np.array(copy_vectors, dtype=np.object)
        return name_targets, code_sentences, code, target_is_unk, copy_vectors

    @staticmethod
    def get_data_in_recurrent_convolution_format_with_validation(input_file, pct_train, min_code_size):
        assert pct_train < 1
        assert pct_train > 0
        names, code, original_names = TokenCodeNamingData.__get_file_data(input_file)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(pct_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = TokenCodeNamingData(names[idxs[:lim]], code[idxs[:lim]])
        return naming.get_data_for_recurrent_convolution(names[idxs[:lim]], code[idxs[:lim]], min_code_size),\
                naming.get_data_for_recurrent_convolution(names[idxs[lim:]], code[idxs[lim:]], min_code_size), naming

    @staticmethod
    def get_data_in_recurrent_copy_convolution_format_with_validation(input_file, pct_train, min_code_size):
        assert pct_train < 1
        assert pct_train > 0
        names, code, original_names = TokenCodeNamingData.__get_file_data(input_file)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(pct_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = TokenCodeNamingData(names[idxs[:lim]], code[idxs[:lim]])
        return naming.get_data_for_recurrent_copy_convolution(names[idxs[:lim]], code[idxs[:lim]], min_code_size),\
                naming.get_data_for_recurrent_copy_convolution(names[idxs[lim:]], code[idxs[lim:]], min_code_size), naming

    @staticmethod
    def get_data_in_convolution_format_with_validation(input_file, names_cx_size, pct_train, min_code_size):
        assert pct_train < 1
        assert pct_train > 0
        names, code, original_names = TokenCodeNamingData.__get_file_data(input_file)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(pct_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = TokenCodeNamingData(names[idxs[:lim]], code[idxs[:lim]])
        return naming.get_data_for_convolution(names[idxs[:lim]], code[idxs[:lim]], names_cx_size, min_code_size),\
                naming.get_data_for_convolution(names[idxs[lim:]], code[idxs[lim:]], names_cx_size, min_code_size), naming

    @staticmethod
    def get_data_in_copy_convolution_format_with_validation(input_file, names_cx_size, pct_train, min_code_size):
        assert pct_train < 1
        assert pct_train > 0
        names, code, original_names = TokenCodeNamingData.__get_file_data(input_file)

        names = np.array(names, dtype=np.object)
        code = np.array(code, dtype=np.object)
        lim = int(pct_train * len(names))
        idxs = np.arange(len(names))
        np.random.shuffle(idxs)
        naming = TokenCodeNamingData(names[idxs[:lim]], code[idxs[:lim]])
        return naming.get_data_for_copy_convolution(names[idxs[:lim]], code[idxs[:lim]], names_cx_size, min_code_size),\
                naming.get_data_for_copy_convolution(names[idxs[lim:]], code[idxs[lim:]], names_cx_size, min_code_size), naming

    def get_data_for_copy_convolution(self, names, code, name_cx_size, sentence_padding):
        assert len(names) == len(code), (len(names), len(code), code.shape)
        name_targets = []
        original_targets = []
        name_contexts = []
        original_names_ids = []
        code_sentences = []
        original_code = []
        copy_vector = []
        target_is_unk = []
        padding = [self.all_tokens_dictionary.get_id_or_unk(self.NONE)]

        for i, name in enumerate(names):
            code_sentence = [self.all_tokens_dictionary.get_id_or_unk(t) for t in code[i]]

            if sentence_padding % 2 == 0:
                code_sentence = padding * (sentence_padding / 2) + code_sentence + padding * (sentence_padding / 2)
            else:
                code_sentence = padding * (sentence_padding / 2 + 1) + code_sentence + padding * (sentence_padding / 2)
            for j in xrange(1, len(name)):  # First element should always be predictable (ie sentence start)
                name_targets.append(self.all_tokens_dictionary.get_id_or_unk(name[j]))
                original_targets.append(name[j])
                target_is_unk.append(self.all_tokens_dictionary.is_unk(name[j]))
                original_names_ids.append(i)
                context = name[:j]
                if len(context) < name_cx_size:
                    context = [self.NONE] * (name_cx_size - len(context)) + context
                else:
                    context = context[-name_cx_size:]
                assert len(context) == name_cx_size, (len(context), name_cx_size,)
                name_contexts.append([self.name_dictionary.get_id_or_unk(t) for t in context])
                code_sentences.append(np.array(code_sentence, dtype=np.int32))
                original_code.append(code[i])
                tokens_to_be_copied = [t == name[j] for t in code[i]]
                copy_vector.append(np.array(tokens_to_be_copied, dtype=np.int32))

        name_targets = np.array(name_targets, dtype=np.int32)
        name_contexts = np.array(name_contexts, dtype=np.int32)
        code_sentences = np.array(code_sentences, dtype=np.object)
        original_names_ids = np.array(original_names_ids, dtype=np.int32)
        copy_vector = np.array(copy_vector, dtype=np.object)
        target_is_unk = np.array(target_is_unk, dtype=np.int32)
        return name_targets, original_targets, name_contexts, code_sentences, original_code, copy_vector, target_is_unk, original_names_ids


    def get_suggestions_given_name_prefix(self, next_name_log_probs, name_cx_size, max_predicted_identifier_size=5, max_steps=100):
        suggestions = defaultdict(lambda: float('-inf'))  # A list of tuple of full suggestions (token, prob)
        # A stack of partial suggestion in the form ([subword1, subword2, ...], logprob)
        possible_suggestions_stack = [
            ([self.NONE] * (name_cx_size - 1) + [self.SUBTOKEN_START], [], 0)]
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
                if subword_tokens[0][-1] == self.SUBTOKEN_END:
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
                context = [self.name_dictionary.get_id_or_unk(k) for k in
                           subword_tokens[0][-name_cx_size:]]
                assert len(context) == name_cx_size
                context = np.array([context], dtype=np.int32)
    
                # Predict next subwords
                target_subword_logprobs = next_name_log_probs(context)
    
                def get_possible_options(name_id):
                    # TODO: Handle UNK differently?
                    subword_name = self.all_tokens_dictionary.get_name_for_id(name_id)
                    if subword_name == self.all_tokens_dictionary.get_unk():
                        subword_name = "***"
                    name = subword_tokens[1] + [subword_name]
                    return subword_tokens[0][1:] + [subword_name], name, target_subword_logprobs[0, name_id] + \
                           subword_tokens[2]
                
                top_indices = np.argsort(-target_subword_logprobs[0])
                possible_options = [get_possible_options(top_indices[i]) for i in xrange(max_size_to_keep)]
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

