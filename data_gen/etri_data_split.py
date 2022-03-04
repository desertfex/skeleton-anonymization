import pickle
from tqdm import tqdm
import numpy as np
import sys
sys.path.extend(['../'])


with open('data/etri/genders.pkl', 'rb') as f:
    genders = pickle.load(f)
with open('data/etri/ages.pkl', 'rb') as f:
    ages = pickle.load(f)
with open('data/etri/data_info.pkl', 'rb') as f:
    data_info = pickle.load(f)

# print(len(ages[0]))

# 3:2:1

# data for training anonymization model
data_train = []
genders_train = [[], []]
ages_train = [[], []]
actions_train = [[], []]

# data for training evaluation model
data_test = []
genders_test = [[], []]
ages_test = [[], []]
actions_test = [[], []]

# data for testing evaluation model
data_test_b = []
genders_test_b = [[], []]
ages_test_b = [[], []]
actions_test_b = [[], []]

data = np.load("data/etri/data_normalized.npy", mmap_mode='r')

for i, skel in tqdm(enumerate(data)):
    if np.any(np.isnan(skel)):
        print("skip", i)
        continue
    filename = genders[0][i]
    person_id = int(filename[6:9])
    action_id = int(filename[1:4]) - 1
    if person_id % 3 == 0:
        data_test.append(skel)
        ages_test[0].append(filename)
        ages_test[1].append(ages[1][i])
        genders_test[0].append(filename)
        genders_test[1].append(genders[1][i])
        actions_test[0].append(filename)
        actions_test[1].append(action_id)
    else:
        data_train.append(skel)
        ages_train[0].append(filename)
        ages_train[1].append(ages[1][i])
        genders_train[0].append(filename)
        genders_train[1].append(genders[1][i])
        actions_train[0].append(filename)
        actions_train[1].append(action_id)

np.save('data/etri/data_train.npy', np.array(data_train))
np.save('data/etri/data_test.npy', np.array(data_test))

with open('data/etri/genders_train.pkl', 'wb') as f:
    pickle.dump(genders_train, f)
with open('data/etri/genders_test.pkl', 'wb') as f:
    pickle.dump(genders_test, f)

with open('data/etri/ages_train.pkl', 'wb') as f:
    pickle.dump(ages_train, f)
with open('data/etri/ages_test.pkl', 'wb') as f:
    pickle.dump(ages_test, f)

with open('data/etri/actions_train.pkl', 'wb') as f:
    pickle.dump(actions_train, f)
with open('data/etri/actions_test.pkl', 'wb') as f:
    pickle.dump(actions_test, f)
