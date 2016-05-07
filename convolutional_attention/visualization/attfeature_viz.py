import cPickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import sys

if __name__ == "__main__":
    if len(sys.argv) !=2:
        print "Usage <inputPkl>"
        sys.exit(-1)

    with open(sys.argv[1], 'rb') as f:
        code_att_feats = cPickle.load(f)

    # Construct matrix
    feat_pos = []
    feature_vecs = []
    cnt = 0
    for i, sentence_data in enumerate(code_att_feats):
        if not sentence_data[0].startswith("is"):
            continue
        elif cnt > 200:
            break  # Just use the first X sentences for now
        cnt +=1
        sentence_features = sentence_data[2].T
        for j in xrange(1, sentence_features.shape[0]-1):  # Ignore START/END
            feat_pos.append((i, j))
            feature_vecs.append(sentence_features[j])

    feature_vecs = np.array(feature_vecs)
    print feature_vecs.shape
    distances = squareform(pdist(feature_vecs, 'cosine'))

    def highlight_location(code_tokens, position, context_size=6):
        return "..." + " ".join(code_tokens[max(position-context_size, 0):position]) + " ***" + code_tokens[position] \
               + "*** " + " ".join(code_tokens[position+1:position+context_size+1]) + "..."

    for i in xrange(distances.shape[0]):
        code_id, tok_id = feat_pos[i]
        print "Neighbors of " + highlight_location(code_att_feats[code_id][1], tok_id) + " in " + code_att_feats[code_id][0]
        nearest_neighbors = np.argsort(distances[i])[1:]  # Ignore self
        for j in xrange(4):
            neigh_id, neigh_tok_id = feat_pos[nearest_neighbors[j]]
            print str(j+1) + ". " + highlight_location(code_att_feats[neigh_id][1], neigh_tok_id) + \
                  " (distance " + str(distances[i][nearest_neighbors[j]]) + ")" + " in " + code_att_feats[neigh_id][0]

        print "---------------------------------------"
        print ""

