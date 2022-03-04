from multiprocessing import Pool
from rotation import *
import numpy as np
from tqdm import tqdm
import sys
sys.path.extend(['../'])


def pre_normalization_step_1(skeleton):
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        if person[0].sum() == 0:
            index = (person.sum(-1).sum(-1) != 0)
            tmp = person[index].copy()
            person *= 0
            person[:len(tmp)] = tmp
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:
                    rest = len(person) - i_f
                    num = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[0:i_f]
                                         for _ in range(num)], 0)[:rest]
                    skeleton[i_p, i_f:] = pad
                    break
    return skeleton


def pre_normalization_step_2(skeleton):
    T = skeleton.shape[1]
    V = skeleton.shape[2]
    if skeleton.sum() == 0:
        return skeleton
    main_body_center = skeleton[0][:, 1:2, :].copy()
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        mask = (person.sum(-1) != 0).reshape(T, V, 1)
        skeleton[i_p] = (skeleton[i_p] - main_body_center) * mask
    return skeleton


def pre_normalization_step_3(skeleton):
    if skeleton.sum() == 0:
        return skeleton
    joint_bottom = skeleton[0, 0, 0]  # HIP
    joint_top = skeleton[0, 0, 1]  # SPINE
    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
    matrix_z = rotation_matrix(axis, angle)
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                continue
            for i_j, joint in enumerate(frame):
                skeleton[i_p, i_f, i_j] = np.dot(matrix_z, joint)
    return skeleton


def pre_normalization_step_4(skeleton):
    if skeleton.sum() == 0:
        return skeleton
    joint_rshoulder = skeleton[0, 0, 8]
    joint_lshoulder = skeleton[0, 0, 4]
    axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    matrix_x = rotation_matrix(axis, angle)
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                continue
            for i_j, joint in enumerate(frame):
                skeleton[i_p, i_f, i_j] = np.dot(matrix_x, joint)
    return skeleton


def pre_normalization(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('pad the null frames with the previous frames')
    new_data = []

    with Pool() as pool:
        with tqdm(total=len(data)) as pbar:
            for i, res in enumerate(pool.imap(pre_normalization_step_1, s, 10)):
                new_data.append(res)
                pbar.update()

    new_data_2 = []

    print('sub the center joint #1 (spine joint in ntu and neck joint in kinetics)')
    with Pool() as pool:
        with tqdm(total=len(data)) as pbar:
            for i, res in enumerate(pool.imap(pre_normalization_step_2, new_data, 10)):
                new_data_2.append(res)
                pbar.update()

    new_data_3 = []

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    with Pool() as pool:
        with tqdm(total=len(data)) as pbar:
            for i, res in enumerate(pool.imap(pre_normalization_step_3, new_data_2, 10)):
                new_data_3.append(res)
                pbar.update()

    new_data_4 = []

    print('parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    with Pool() as pool:
        with tqdm(total=len(data)) as pbar:
            for i, res in enumerate(pool.imap(pre_normalization_step_4, new_data_3, 10)):
                new_data_4.append(res)
                pbar.update()

    return np.transpose(new_data_4, [0, 4, 2, 3, 1])
