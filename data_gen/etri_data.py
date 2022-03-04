import os
# import argparse
import numpy as np
# from numpy.lib.format import open_memmap
from tqdm import tqdm
import csv
# import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool

skeleton_bone_pairs = (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
)
training_subjects = [i for i in range(1, 101) if i % 3 != 0]
test_subjects = [i for i in range(1, 101) if i % 3 == 0]

joint_info_keys = ['3dX', '3dY', '3dZ', 'depthX', 'depthY',
                   'orientationX', 'orientationY', 'orientationZ',
                   'orientationW', 'trackingState']

max_frame = 664
num_joint = 25


def read_skeleton_filter(file):
    skeleton_sequence = {}
    skeleton_sequence['frameInfo'] = []
    with open(file, 'r') as f:
        rows = list(csv.DictReader(f, delimiter=','))
        bodies = list(set([row['bodyindexID'] for row in rows]))
        last_frame_num = -1
        for row in rows:
            if last_frame_num != row['frameNum']:
                frame_info = {
                    'numBody': len(bodies),
                    'bodyInfo': []
                }
                for _ in range(len(bodies)):
                    frame_info['bodyInfo'].append({
                        'numJoint': 25,
                        'jointInfo': [],
                    })
            else:
                frame_info = skeleton_sequence['frameInfo'][-1]
            body_index = bodies.index(row['bodyindexID'])
            body_info = frame_info['bodyInfo'][body_index]

            for joint_id in range(1, body_info['numJoint'] + 1):
                joint_info = {}
                for key in joint_info_keys:
                    joint_info['%s' % key] = row['joint%d_%s' %
                                                 (joint_id, key)]
                body_info['jointInfo'].append(joint_info)

            if last_frame_num != row['frameNum']:
                skeleton_sequence['frameInfo'].append(frame_info)
            last_frame_num = row['frameNum']
    skeleton_sequence['numFrame'] = len(skeleton_sequence['frameInfo'])
    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + \
            s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=1, num_joint=25):
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for frame_num, frame in enumerate(seq_info['frameInfo']):
        if frame['numBody'] > 1:
            return None  # Skip
        for body_num, body in enumerate(frame['bodyInfo']):
            for joint_num, joint in enumerate(body['jointInfo']):
                if body_num < max_body and joint_num < num_joint:
                    data[body_num, frame_num, joint_num, :] = [
                        joint['3dX'], joint['3dY'], joint['3dZ']]

    data = data.transpose(3, 1, 2, 0)  # (coordinate, frame, joint, body)
    return data


def gender_age(file):
    labels = []
    with open(file, 'r') as f:
        rows = csv.DictReader(f, delimiter='\t')
        for row in rows:
            labels.append([row['Gender'] == 'm', row['Age']])
    return labels


def func(info):
    try:
        return read_xyz(info['full_path'], 1, 25)
    except Exception as e:
        print(info)
        raise


def gendata(data_path_list):
    gender_age_list = gender_age('data/etri_raw/gender_age.txt')
    sample_infos = []
    for data_path in data_path_list:
        for filename in sorted(os.listdir(data_path)):
            action_class = int(
                filename[filename.find('A') + 1:filename.find('A') + 4])
            subject_id = int(
                filename[filename.find('P') + 1:filename.find('P') + 4])
            group_id = int(filename[filename.find(
                'G') + 1:filename.find('G') + 4])
            camera_id = int(filename[filename.find(
                'C') + 1:filename.find('C') + 4])
            if action_class in range(44, 49):  # skip human-human interactions
                continue
            is_training = subject_id in training_subjects
            sample_infos.append({
                'name': filename,
                'full_path': os.path.join(data_path, filename),
                'gender': gender_age_list[subject_id - 1][0],
                'age': gender_age_list[subject_id - 1][1],
                'is_training': is_training,
            })
    datas = []
    skip_cases = []
    skip_indices = []
    with Pool() as pool:
        with tqdm(total=len(sample_infos)) as pbar:
            for i, data in enumerate(pool.imap(func, sample_infos, 50)):
                pbar.update()
                if data is None:
                    skip_cases.append(sample_infos[i]['name'])
                    skip_indices.append(i)
                    # print("skip", sample_infos[i]['name'])
                else:
                    datas.append(data)

    print(skip_cases)
    with open('data/etri/skip_cases.pkl', 'wb') as f:
        pickle.dump(skip_cases, f)

    sample_infos = [info for i, info in enumerate(
        sample_infos) if i not in skip_indices]
    with open('data/etri/labels.pkl', 'wb') as f:
        pickle.dump(sample_infos, f)

    fp = np.zeros((len(datas), 3, max_frame, num_joint, 1),
                  dtype=np.float32)  # samples, xyz, frames, joints, body
    for i, data in enumerate(datas):
        fp[i, :, 0:data.shape[1], :, :] = data

    return fp


np.save("data/etri/data.npy",
        gendata(['data/etri_raw/P001-P050', 'data/etri_raw/P051-P100']))
