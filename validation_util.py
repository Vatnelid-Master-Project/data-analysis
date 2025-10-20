import random

def split(dataset, size, labels = None):
    valid_spec = []
    valid_labels = []
    
    for i in range(0, size):
        ran_num = random.randint(0, len(dataset))

        del dataset [ran_num]
        
        if labels != None:
            del labels [ran_num]
            valid_labels.append(labels[ran_num])

        valid_spec.append(dataset[ran_num])
    
    if labels != None:
        return valid_spec, valid_labels
    
    return valid_spec

def get_label_index(label : str, all_labels) -> int:

    for i in range(len(all_labels)):
        if all_labels[i] == label:
            return i
    return -1
    