import json
import sys
from collections import Counter
from itertools import chain

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'Usage <input_file>'
        sys.exit(-1)

    input_file = sys.argv[1]
    with open(input_file, 'r') as f:
        data = json.load(f)

    name_counter = Counter(chain.from_iterable((e["name"] for e in data)))

    rare_limit = 10

    in_name_rare = {True: 0, False:0}
    also_found_in_body_rare = {True: 0, False: 0}

    for element in data:
        subtokens = set(element["name"])
        code_subtokens = set(element["tokens"])

        for t in subtokens:
            is_rare = name_counter[t] < rare_limit
            in_name_rare[is_rare]+=1
            if t in code_subtokens:
                also_found_in_body_rare[is_rare]+=1

    print "Rare: %s" % (float(also_found_in_body_rare[True]) / in_name_rare[True])
    print "Common: %s" % (float(also_found_in_body_rare[False]) / in_name_rare[False])