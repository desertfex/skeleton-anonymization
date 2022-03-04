from feeders import tools
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import pickle
import torch
import sys
sys.path.extend(['../'])


class Feeder(Dataset):
    def __init__(self, data_path, gender_label_path, action_label_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, anonymizer=None, output_device=0, batch_size=100):
        """
        :param data_path:
        :param gender_label_path:
        :param action_label_path:
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_path = data_path
        self.gender_label_path = gender_label_path
        self.action_label_path = action_label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.anonymizer = anonymizer
        self.batch_size = batch_size
        self.output_device = output_device
        self.load_data()
        self.anonymize()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.gender_label_path) as f:
                self.sample_name, self.gender_label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.gender_label_path, 'rb') as f:
                self.sample_name, self.gender_label = pickle.load(
                    f, encoding='latin1')

        try:
            with open(self.action_label_path) as f:
                self.sample_name, self.action_label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.action_label_path, 'rb') as f:
                self.sample_name, self.action_label = pickle.load(
                    f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            perm = np.random.choice(
                len(self.gender_label), 1000, replace=False)
            self.gender_label = np.array(self.gender_label)[perm]
            self.action_label = np.array(self.action_label)[perm]
            self.data = self.data[perm]
            self.sample_name = np.array(self.sample_name)[perm]

    def anonymize(self):
        print("Anonymize test data")

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield torch.from_numpy(iterable[ndx:min(ndx + n, l)].copy()).cuda(self.output_device)
        self.anonymizer.eval()
        process = tqdm(batch(self.data, n=self.batch_size), dynamic_ncols=True)
        results = []
        with torch.no_grad():
            for batch_idx, samples in enumerate(process):
                anonymized = self.anonymizer(samples)
                results.append(anonymized.cpu().numpy())
        self.anonymized = np.concatenate(results)

    def get_reconstruction_loss(self):
        return np.square(self.anonymized - self.data).mean()

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(
            axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape(
            (N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.gender_label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        original = torch.from_numpy(np.copy(self.data[index]))
        anonymized = self.anonymized[index]
        gender_label = self.gender_label[index]
        action_label = self.action_label[index]

        if self.normalization:
            anonymized = (anonymized - self.mean_map) / self.std_map
        if self.random_shift:
            anonymized = tools.random_shift(anonymized)
        if self.random_choose:
            anonymized = tools.random_choose(anonymized, self.window_size)
        elif self.window_size > 0:
            anonymized = tools.auto_pading(anonymized, self.window_size)
        if self.random_move:
            anonymized = tools.random_move(anonymized)

        return original, anonymized, gender_label, action_label, index

    def top_k_action(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:]
                     for i, l in enumerate(self.action_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_gender(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:]
                     for i, l in enumerate(self.gender_label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
