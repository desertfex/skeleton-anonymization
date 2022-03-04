import numpy as np

import pickle

with open("data/ntu/xview/train_label.pkl", 'rb') as f:
    filenames, _ = pickle.load(f, encoding='latin1')
    new_labels = [int(x[9:12]) - 1 for x in filenames]
with open('data/ntu/xview/train_label_reid.pkl', 'wb') as f:
    pickle.dump((filenames, new_labels), f)

with open("data/ntu/xview/val_label.pkl", 'rb') as f:
    filenames, _ = pickle.load(f, encoding='latin1')
    new_labels = [int(x[9:12]) - 1 for x in filenames]
with open('data/ntu/xview/val_label_reid.pkl', 'wb') as f:
    pickle.dump((filenames, new_labels), f)
