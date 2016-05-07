from collections import Counter

class FeatureDictionary:
    """
    A simple feature dictionary that can convert features (ids) to
    their textual representation and vice-versa.
    """

    def __init__(self):
        self.next_id = 0
        self.token_to_id = {}
        self.id_to_token = {}
        self.add_or_get_id(self.get_unk())

    def add_or_get_id(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]

        this_id = self.next_id
        self.next_id += 1
        self.token_to_id[token] = this_id
        self.id_to_token[this_id] = token

        return this_id

    def is_unk(self, token):
        return token not in self.token_to_id

    def get_id_or_unk(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return self.token_to_id[self.get_unk()]

    def get_id_or_none(self, token):
        if token in self.token_to_id:
            return self.token_to_id[token]
        else:
            return None

    def get_name_for_id(self, token_id):
        return self.id_to_token[token_id]

    def __len__(self):
        return len(self.token_to_id)

    def __str__(self):
        return str(self.token_to_id)

    def get_all_names(self):
        return frozenset(self.token_to_id.keys())

    @staticmethod
    def get_unk():
        return "%UNK%"

    @staticmethod
    def get_feature_dictionary_for(tokens, count_threshold=10):
        token_counter = Counter(tokens)
        feature_dict = FeatureDictionary()
        for token, count in token_counter.iteritems():
            if count >= count_threshold:
                feature_dict.add_or_get_id(token)
        return feature_dict