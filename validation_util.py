import random

def split(dataset, labels, size):
    valid_spec = []
    valid_labels = []
    
    for i in range(0, size):
        ran_num = random.randint(0, len(labels))

        del dataset [ran_num]
        del labels [ran_num]

        valid_spec.append(dataset[ran_num])
        valid_labels.append(labels[ran_num])
    
    return valid_spec, valid_labels

def get_label_index(label : str, all_labels) -> int:

    for i in range(len(all_labels)):
        if all_labels[i] == label:
            return i
    return -1
    