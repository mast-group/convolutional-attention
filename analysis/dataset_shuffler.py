import json
import random
import sys


def shuffle_data(filename, outfile):
    with open(filename) as f:
        data = json.load(f)
    for method in data:
        code_tokens = method["tokens"][1:-1]
        random.shuffle(code_tokens)
        method["tokens"] = method["tokens"][0:1] + code_tokens + method["tokens"][-1:]

    with open(outfile, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print "Usage <inputJson> <outputJson>"
        sys.exit(-1)
    shuffle_data(sys.argv[1], sys.argv[2])