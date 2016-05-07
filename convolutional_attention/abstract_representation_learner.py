

class RepresentationForNameLearner:

    def train(self, input_file):
        """
        Train the learner for the given input file.
        :param input_file: the file directory
        :return:
        """
        raise NotImplementedError()

    def get_representations(self, elements):
        """
        :param elements the elements in their json-like format
        Get the representations for each of the elements in the elements
        :return:
        """

    def predict_name(self, representation):
        """
        Predict the name, given the representation.
        :param context:
        :param representation:
        :return: a list of all possible suggestions
        """
        raise NotImplemented()



