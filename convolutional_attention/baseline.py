from collections import defaultdict, Counter
import math
import numpy as np

from f1_evaluator import PointSuggestionEvaluator, token_precision_recall

def compute_idfs(tokens_per_document):
    token_counts = defaultdict(int)
    document_term_count = defaultdict(int)
    document_count = len(tokens_per_document)

    for document in tokens_per_document:
        term_counts = Counter(document)
        for token, count in term_counts.iteritems():
            document_term_count[token] += 1
            token_counts[token] += count


    # Remove rare words
    to_remove = []
    for token, count in token_counts.iteritems():
        if count < 5:
            to_remove.append(token)

    for token in to_remove:
        del document_term_count[token]
        del token_counts[token]

    def idf(token):
        idf = math.log(1. + float(document_count) / document_term_count[token])
        return idf

    return {t: idf(t) for t in token_counts}

def compute_names_vector(data, vsm_idfs):
    names = []
    vsm = np.zeros((len(data), len(vsm_idfs)))
    for i, document in enumerate(data):
        document_terms = Counter(document["tokens"])
        total_terms = len(document["tokens"])
        name = document["name"]
        names.append(name)
        for j, term in enumerate(vsm_idfs.keys()):
            vsm[i, j] = float(document_terms[term]) / total_terms * vsm_idfs[term]
    return names, vsm


import sys
import json
from scipy.spatial.distance import cdist
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'Usage <train_file> <test_file>'
        sys.exit(-1)

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    print 'Training idfs'
    with open(train_file) as f:
        train_data = json.load(f)

    tokens = [document["tokens"] for document in train_data]
    vsm_idfs = compute_idfs(tokens)
    names, train_vectors = compute_names_vector(train_data, vsm_idfs)

    print "Retrieving test vectors..."
    with open(test_file) as f:
        test_data = json.load(f)
    test_names, test_vectors = compute_names_vector(test_data, vsm_idfs)

    print "Computing Suggestions..."
    # Suggest the nearest neighbor name for the test vector in minibatches
    eval = PointSuggestionEvaluator()
    num_ranks = 5
    minibatch_size = 100
    for batch_id in xrange(int(math.ceil(float(len(test_names)) / minibatch_size))):
        from_id = minibatch_size * batch_id
        to_id = from_id + minibatch_size
        distances = cdist(test_vectors[from_id:to_id], train_vectors, 'cosine')
        suggestions = np.argsort(distances, axis=1)
        sys.stdout.write(".")
        for i in xrange(suggestions.shape[0]):
            suggested_names = [names[suggestions[i][j]] for j in xrange(num_ranks)]
            real_name = test_names[from_id + i]
            res = [token_precision_recall(name, real_name) for name in suggested_names]
            is_correct = [len(real_name) == len(name) and all(k == l for k, l in zip(name, real_name)) for name in suggested_names]
            eval.add_result([1.] * num_ranks , is_correct, [False] * num_ranks, res)
    print ''
    print eval