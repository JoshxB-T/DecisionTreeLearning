import math, collections, csv

class DecisionNode:
    def __init__(self, attribute, label, is_leaf):
        self.attribute = attribute
        self.label = label
        self.is_leaf = is_leaf
        self.branches = {}

    def add_branch(self, value, node):
        self.branches[value] = node

def print_tree(node, attrs_names, indent):
    # Base case to print leaf nodes 
    if node.is_leaf:
        print(indent + f"-> {node.label}")
        return
    
    if attrs_names:
        attr_label = attrs_names[node.attribute]
    else:
        attr_label = f"A{node.attribute}"

    for val, subtree in node.branches.items():
        print(indent + f" {attr_label} ({val})")
        print_tree(subtree, attrs_names, indent + "--") # Recursive call

def entropy(exs):
    if not exs:
        return 0.0

    counts = collections.Counter([row[-1] for row in exs])
    total = len(exs)
    ent = 0.0

    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)

    return ent

def plurality_value(exs):
    if not exs:
        return DecisionNode(None, None, True)

    counts = collections.Counter([row[-1] for row in exs])
    label, _ = counts.most_common(1)[0]

    return DecisionNode(None, label, True)

def all_same_class(exs):
    if not exs:
        return True

    first = exs[0][-1]

    return all(row[-1] == first for row in exs)

def get_attribute_values(attr_index, exs):
    return sorted({row[attr_index] for row in exs})

def get_subset(exs, attr_index, value):
    return [row for row in exs if row[attr_index] == value]

def remove_attribute(attrs, attr):
    return [a for a in attrs if a != attr]

def information_gain(exs, attr_index):
    base_entropy = entropy(exs)
    total = len(exs)
    vals = get_attribute_values(attr_index, exs)
    remainder = 0.0

    for v in vals:
        subset = get_subset(exs, attr_index, v)
        remainder += (len(subset) / total) * entropy(subset)

    return base_entropy - remainder

def choose_best_attribute(exs, attrs):
    best_attr = None
    best_gain = -float('inf')

    for a in attrs:
        gain = information_gain(exs, a)

        if gain > best_gain:
            best_gain = gain
            best_attr = a

    return best_attr

def classification(exs):
    return plurality_value(exs)

def Learn_Decision_Tree(exs, attrs, parent_exs):
    if not exs:
        return plurality_value(parent_exs)
    if all_same_class(exs):
        return classification(exs)
    if not attrs:
        return plurality_value(exs)

    best_attr = choose_best_attribute(exs, attrs)
    tree = DecisionNode(best_attr, None, False)

    for value in get_attribute_values(best_attr, exs):
        subset = get_subset(exs, best_attr, value)
        subtree = Learn_Decision_Tree(subset, remove_attribute(attrs, best_attr), exs)
        tree.add_branch(value, subtree)

    return tree

def parse_data(data):
    reader = csv.reader(data.splitlines(), skipinitialspace=True)
    parsed = [row for row in reader if row]

    return parsed

def open_file():
    with open("restaurant.csv", "r", encoding="utf-8") as file:
        data = file.read()
    parsed_data = parse_data(data)

    return parsed_data

def main():
    data = open_file()
    if not data:
        print("No data loaded from restaurant.csv")
        return

    # Last column is output
    num_attrs = len(data[0]) - 1
    attrs = list(range(num_attrs))
    attrs_names = ["Alt", "Bar", "Fri", "Hun", "Pat", "Price", "Rain", "Res", "Type", "Est"]
    tree = Learn_Decision_Tree(data, attrs, [])

    # Recursively print the tree
    print("Decision tree learned:")
    print_tree(tree, attrs_names, "|--")

if __name__ == "__main__":
    main()