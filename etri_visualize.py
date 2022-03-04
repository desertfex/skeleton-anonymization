import multiprocessing
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os
import sys
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from model.anonymization_res import Anonymizer
import torch
from multiprocessing import Pool
from multiprocessing.pool import MapResult

# NTU RGB+D 60/120 Action Classes
actions = {
    1: "eating food with a fork",
    2: "pouring water into a cup",
    3: "taking medicine",
    4: "drinking water",
    5: "putting food in the fridge/taking food from the fridge",
    6: "trimming vegetables",
    7: "peeling fruit",
    8: "using a gas stove",
    9: "cutting vegetable on the cutting board",
    10: "brushing teeth",
    11: "washing hands",
    12: "washing face",
    13: "wiping face with a towel",
    14: "putting on cosmetics",
    15: "putting on lipstick",
    16: "brushing hair",
    17: "blow drying hair",
    18: "putting on a jacket",
    19: "taking off a jacket",
    20: "putting on/taking off shoes",
    21: "putting on/taking off glasses",
    22: "washing the dishes",
    23: "vacuumming the floor",
    24: "scrubbing the floor with a rag",
    25: "wipping off the dinning table",
    26: "rubbing up furniture",
    27: "spreading bedding/folding bedding",
    28: "washing a towel by hands",
    29: "hanging out laundry",
    30: "looking around for something",
    31: "using a remote control",
    32: "reading a book",
    33: "reading a newspaper",
    34: "handwriting",
    35: "talking on the phone",
    36: "playing with a mobile phone",
    37: "using a computer",
    38: "smoking",
    39: "clapping",
    40: "rubbing face with hands",
    41: "doing freehand exercise",
    42: "doing neck roll exercise",
    43: "massaging a shoulder oneself",
    44: "taking a bow",
    45: "talking to each other",
    46: "handshaking",
    47: "hugging each other",
    48: "fighting each other",
    49: "waving a hand",
    50: "flapping a hand up and down (beckoning)",
    51: "pointing with a finger",
    52: "opening the door and walking in",
    53: "fallen on the floor",
    54: "sitting up/standing up",
    55: "lying down",
}

ntu_skeleton_bone_pairs = tuple((i-1, j-1) for (i, j) in (
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5),
    (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
    (13, 1), (14, 13), (15, 14), (16, 15), (17, 1), (18, 17),
    (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25), (25, 12)
))


def visualize_one(option):
    skel, action_class, action_name, index, filename, gender_class, output_path, anonymized = option
    bones = ntu_skeleton_bone_pairs
    mpl.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    skeleton1 = skel[..., 0]   # out (C,T,V)

    skeleton_index = [0]
    skeleton_frames = skeleton1.transpose(1, 0, 2)

    lines = {}
    for i, j in bones:
        lines[(i, j)], = ax.plot([0, 0], [0, 0], [0, 0], color='blue')

    def animate(skeleton):
        for i, j in bones:
            joint_locs = skeleton[:, [i, j]]
            lines[(i, j)].set_xdata(joint_locs[0])
            lines[(i, j)].set_ydata(joint_locs[1])
            lines[(i, j)].set_3d_properties(joint_locs[2])

        if anonymized:
            plt.title('Anonymized Skeleton {} Frame #{} of {} from {}\n (Action {}: {})\n{} ({})'.format(
                index, skeleton_index[0], skeleton_frames.shape[0], 'etri', action_class, action_name, filename, 'male' if gender_class == True else 'female'),
                fontsize=8)
        else:
            plt.title('Skeleton {} Frame #{} of {} from {}\n (Action {}: {})\n{} ({})'.format(
                index, skeleton_index[0], skeleton_frames.shape[0], 'etri', action_class, action_name, filename, 'male' if gender_class == True else 'female'),
                fontsize=8)
        skeleton_index[0] += 1
        skeleton_index[0] %= skeleton_frames.shape[0]
        return lines.values()

    ani = FuncAnimation(fig, animate, skeleton_frames,
                        interval=1000 / 24, blit=True)

    if anonymized:
        ani.save(os.path.join(
            output_path, 'anonymized_{}.webp'.format(index)), writer=FFMpegWriter(fps=24, codec='libwebp'))
    else:
        ani.save(os.path.join(
            output_path, 'original_{}.webp'.format(index)), writer=FFMpegWriter(fps=24, codec='libwebp'))


def visualize(data_path, action_label_path, gender_label_path, output_path, indices=[], filenames=[], anonymizer=None, weight=None, device=0, pool: Pool = None) -> MapResult:
    data = np.load(data_path, mmap_mode='r')
    with open(action_label_path, 'rb') as f:
        labels = pickle.load(f)
    with open(gender_label_path, 'rb') as f:
        gender_labels = pickle.load(f)

    if len(filenames):
        indices = indices + \
            [find_by_filename(name, action_label_path) for name in filenames]
    targets = np.copy(data[indices])

    options = []
    if anonymizer is None and weight:
        anonymizer = Anonymizer(
            num_point=25, num_person=1, num_gcn_scales=13, num_g3d_scales=6,
            graph='graph.ntu_rgb_d.AdjMatrixGraph').cuda(device)
        anonymizer.load_state_dict(torch.load(weight))

    if anonymizer is None:
        for k, index in enumerate(indices):
            original = targets[k]
            data_filename = labels[0][index]

            action_class = labels[1][index] + 1
            action_name = actions[action_class]
            gender_class = gender_labels[1][index]
            options.append([original, action_class, action_name,
                            index, data_filename, gender_class, output_path, False])
    else:
        with torch.no_grad():
            anonymized = anonymizer(
                torch.Tensor(targets).cuda(device)).cpu().numpy()
        for k, index in enumerate(indices):
            anonymized_one = anonymized[k]
            data_filename = labels[0][index]
            action_class = labels[1][index] + 1
            action_name = actions[action_class]
            gender_class = gender_labels[1][index]
            options.append([anonymized_one, action_class,
                            action_name, index, data_filename, gender_class, output_path, True])

    res = pool.map_async(visualize_one, options)
    print("visualization started")
    return res


def find_by_filename(filename, action_label_path):
    with open(action_label_path, 'rb') as f:
        labels = pickle.load(f)

    return labels[0].index(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ETRI Skeleton Visualizer')

    parser.add_argument('-o', '--outputpath',
                        help='output path')
    parser.add_argument('-p', '--datapath',
                        help='location of dataset numpy file')
    parser.add_argument('-w', '--weight',
                        help='location of model weight')
    parser.add_argument('-g', '--genderlabelpath',
                        help='location of gender label pickle file')
    parser.add_argument('-l', '--labelpath',
                        help='location of action label pickle file')
    parser.add_argument('-i', '--indices',
                        type=int,
                        nargs='+',
                        required=True,
                        help='the indices of the samples to visualize')
    parser.add_argument('-d', '--device',
                        type=int)

    args = parser.parse_args()
    pool = Pool()
    res_1 = visualize(args.datapath, args.labelpath,
                      args.genderlabelpath, args.outputpath,
                      indices=args.indices, weight=args.weight, device=args.device, pool=pool)
    res_2 = visualize(args.datapath, args.labelpath,
                      args.genderlabelpath, args.outputpath,
                      indices=args.indices, weight=None, device=args.device, pool=pool)
    res_1.wait()
    res_2.wait()
    pool.close()
    pool.join()
