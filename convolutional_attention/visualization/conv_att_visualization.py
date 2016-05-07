import json
import pystache
import sys
import codecs

if len(sys.argv) < 2:
    print "Usage <data.json>"
    sys.exit(-1)

import numpy as np

with open("entry.mustache") as f:
    entry_template = pystache.parse(unicode(f.read()))
with open("token.mustache") as f:
    token_template = pystache.parse(unicode(f.read()))

with open(sys.argv[1]) as f:
    raw_data = json.load(f)

renderer = pystache.Renderer()


def colormap(value):
    red = 255
    green = 255
    blue = int(255 * (1. - value))
    return red, green, blue


def token_with_attention_to_string(tokens, attention_vector):
    token_html = []
    attention_vector = np.array(attention_vector)
    normalized_attention_vector = attention_vector / np.max(attention_vector)
    for token, attention_value, norm_attention_value in zip(tokens, attention_vector, normalized_attention_vector):
        r, g, b = colormap(norm_attention_value)
        html = renderer.render(token_template, {"weight": attention_value,
                                                "token": token,
                                                "red": r,
                                                "green": g,
                                                "blue": b
                                                })
        token_html.append(html)

    return ' '.join(token_html)

count = 0
break_points = []
data = []
is_first_subtoken = True
for i, suggestion in enumerate(raw_data):
    suggestion["attention_visualization"] = token_with_attention_to_string(suggestion["tokens"],
                                                                           suggestion["att_vector"])
    suggestion["is_first"] = is_first_subtoken

    sorted_suggestions = sorted(suggestion["suggestions"].items(), key=lambda x: x[1], reverse=True)
    suggestion["suggestions"] = [{"token": k,
                                  "probability": "%.2e" % v,
                                  "is_correct": k == suggestion["target subtoken"]}
                                 for k, v in sorted_suggestions[:10]]
    is_first_subtoken = False

    html = renderer.render(entry_template, suggestion)
    data.append({"html": html})
    count += 1
    if suggestion["target subtoken"] == "%END%":
        data.append({"html": "<hr/>"})
        is_first_subtoken = True
        if count > 1000:
           count = 0
           break_points.append(len(data))
break_points.append(len(data)) # last batch

# Generate per-page html
prev_idx = 0
for i, next_pos in enumerate(break_points):
    current_batch = data[prev_idx:next_pos]
    prev_idx = next_pos

    # Load full document
    with open("document.mustache") as f:
        output = pystache.render(f.read(), {"examples": current_batch, "model":"Attentional Convolutional"})

    with codecs.open("conv_att_visualization" + str(i) + ".html", 'w', 'utf-8') as f:
        f.write(output)
