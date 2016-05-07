import json
import sys
from collections import Counter
from itertools import chain

def tokens(input_json):
    with open(input_json) as f:
        data = json.load(f)

    return Counter(chain.from_iterable((e["name"] for e in data)))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage <trainJson> <testJson>"
        sys.exit(-1)

    train_toks = tokens(sys.argv[1])
    test_toks = tokens(sys.argv[2])

    known_tokens = set(t for t, c in train_toks.iteritems() if c>1)
    num_unk_tokens = sum(c for t, c in test_toks.iteritems() if t not in known_tokens)
    total_test_toks = sum(test_toks.values())

    print "UNK toks in test %s (%s/%s)" % (float(num_unk_tokens)/total_test_toks , num_unk_tokens, total_test_toks)

