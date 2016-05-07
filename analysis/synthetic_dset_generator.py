import random
import json
target_names_no_order = {
    frozenset(["A"]): ["get"],
    frozenset(["A", "B"]): ["get", "blue"],
    frozenset(["A", "B", "C"]): ["get", "black"],
    frozenset(["A", "C"]): ["get", "color"],
    frozenset(["B"]): ["set", "blue"],
    frozenset(["B", "C"]): ["initialize"],
    frozenset(["C"]): ["is", "nice"],
    frozenset(["C", "D"]): ["is", "test"],
    frozenset(["D"]): ["had", "test"],
    frozenset(["A", "D"]): ["get", "test"],
    frozenset(["B", "D"]): ["no", "color"]
}


def generate_synthetic_no_order(num_samples, p_noise=.8):
    """
    Generate a random synthetic dataset using the above mapping
    :param num_samples:
    :param p_noise:
    :return:
    """
    samples = []
    for i in xrange(num_samples):
        current_elements = target_names_no_order.keys()[random.randint(0, len(target_names_no_order) - 1)]
        name = target_names_no_order[current_elements]
        tokens = []
        included_elements = set()
        add_noise = random.random() < p_noise
        while add_noise or len(included_elements) != len(current_elements):
            if add_noise:
                element = str(random.randint(0, 100))
            else:
                element = [t for t in current_elements][random.randint(0, len(current_elements) - 1)]
                included_elements.update(element)
            tokens.append(element)
            add_noise = random.random() < p_noise
        samples.append({"tokens": tokens, "name": name})
    return samples

target_names_order = {
    ("A", "B"): ["get"],
    ("B", "A"): ["set"],
    ("A", "B", "C"): ["get", "color"],
    ("B", "A", "C"): ["set", "color"],
    ("B", "C"): ["is", "color"],
    ("C", "B"): ["no", "nice"],
    ("C", "A"): ["no", "color"],
    ("A", "C"): ["random"],
    ("C", "A", "B"): ["no", "test"],
    ("A", "D", "B"): ["get", "node"], # D is position invariant
    ("A", "B", "D"): ["get", "node"],
    ("D", "A", "B"): ["get", "node"],
    ("B", "D", "A"): ["set", "node"],
    ("B", "A", "D"): ["set", "node"],
    ("D", "B", "A"): ["set", "node"],
    ("D"): ["node"],
    ("B", "C", "D"): ["is", "node"],
    ("D", "B", "C"): ["is", "node"],
    ("B", "D", "C"): ["is", "node"],
    ("C", "B", "D"): ["no", "node"],
    ("D", "C", "B"): ["no", "node"],
    ("C", "D", "B"): ["no", "node"],
    ("D", "Q"): ["wow"],
    ("Q", "D"): ["lol"],
    ("Q", "A", "B"): ["get", "lol", "wow"],
    ("A", "Q", "B"): ["get", "lol", "wow"],
    ("Q", "B", "A"): ["set", "lol", "wow"],
    ("B", "A", "Q"): ["set", "lol", "wow"]
}

def generate_synthetic_with_order(num_samples, p_noise=.7):
    """
    Generate a random synthetic dataset using the above mapping
    :param num_samples:
    :param p_noise:
    :return:
    """
    samples = []
    for i in xrange(num_samples):
        current_elements = target_names_order.keys()[random.randint(0, len(target_names_order) - 1)]
        name = target_names_order[current_elements]
        tokens = []
        current_idx = 0
        add_noise = random.random() < p_noise
        while add_noise or current_idx < len(current_elements):
            if add_noise:
                element = str(random.randint(0, 100))
            else:
                element = current_elements[current_idx]
                if random.random() < p_noise:
                    current_idx += 1
            tokens.append(element)
            add_noise = random.random() < p_noise
        samples.append({"tokens": tokens, "name": name})
    return samples

if __name__ == "__main__":
    with open('synthetic_train.json', 'w') as f:
        json.dump(generate_synthetic_with_order(2000), f)
    with open('synthetic_test.json', 'w') as f:
        json.dump(generate_synthetic_with_order(2000), f)

