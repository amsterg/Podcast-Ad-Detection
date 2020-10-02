import matplotlib
from random import randint, sample
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import h5py
import warnings
from tqdm import tqdm
from colorama import Fore
import pathlib
from scipy.ndimage import binary_dilation
import soundfile
import webrtcvad
import numpy as np
from librosa.display import specshow
import librosa
from collections import Counter
import pandas as pd
from yaml import safe_load
import sys
import os
from sklearn.metrics import confusion_matrix
from itertools import product

# nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())


warnings.filterwarnings("ignore")

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

RAW_DATA_DIR = config['RAW_DATA_DIR']
PROC_DATA_DIR = config['PROC_DATA_DIR']
INTERIM_DATA_DIR = config['INTERIM_DATA_DIR']
MODEL_SAVE_DIR = config['MODEL_SAVE_DIR']
PODS_TRAIN_DIR: config['PODS_TRAIN_DIR']
PODS_VAL_DIR: config['PODS_VAL_DIR']
PODS_TEST_DIR: config['PODS_TEST_DIR']

SAMPLING_RATE = config['SAMPLING_RATE']
WINDOW_SIZE = config['WINDOW_SIZE']
WINDOW_STEP = config['WINDOW_STEP']
N_FFT = int(WINDOW_SIZE * SAMPLING_RATE / 1000)
H_L = int(WINDOW_STEP * SAMPLING_RATE / 1000)
STEP_SIZE_EM = int((SAMPLING_RATE/16)/H_L)
MEL_CHANNELS = config['MEL_CHANNELS']
SMOOTHING_LENGTH = config['SMOOTHING_LENGTH']
SMOOTHING_WSIZE = config['SMOOTHING_WSIZE']
DBFS = config['DBFS']
SMOOTHING_WSIZE = int(SMOOTHING_WSIZE * SAMPLING_RATE / 1000)

dirs_ = set([globals()[d] for d in globals() if d.__contains__('DIR')] +
            [config[d] for d in config if d.__contains__('DIR')])

VAD = webrtcvad.Vad(mode=config['VAD_MODE'])


def structure(dirs=[]):
    """
    Summary:

    Args:

    Returns:

    """
    dirs_reqd = set(list(dirs_) + list(dirs))
    for data_dir in dirs_reqd:
        if not pathlib.Path.exists(pathlib.Path(data_dir)):
            os.makedirs(data_dir)


def normalization(aud, norm_type='peak'):
    """
    Summary:

    Args:

    Returns:

    """
    try:
        assert len(aud) > 0
        if norm_type == "peak":
            aud = aud / np.max(aud)

        elif norm_type == "rms":
            dbfs_diff = DBFS - (20 *
                                np.log10(np.sqrt(np.mean(np.square(aud)))))
            if DBFS > 0:
                aud = aud * np.power(10, dbfs_diff / 20)

        return aud
    except AssertionError as e:
        raise AssertionError("Empty audio sig")


def preprocess_aud(aud_input, sr=44100):
    """
    Summary:

    Args:

    Returns:

    """
    if isinstance(aud_input, list) or isinstance(aud_input, np.ndarray):
        aud = np.array(aud_input)
    else:
        fname = aud_input
        aud, sr = librosa.load(fname, sr=None)
    if sr != SAMPLING_RATE:
        aud = librosa.resample(aud, sr, SAMPLING_RATE)
    try:
        aud = normalization(aud, norm_type='peak')

    except AssertionError as e:
        print(AssertionError("Empty audio sig"))

    trim_len = len(aud) % SMOOTHING_WSIZE
    aud = np.append(aud, np.zeros(SMOOTHING_WSIZE - trim_len))

    assert len(aud) % SMOOTHING_WSIZE == 0, print(len(aud) % trim_len, aud)

    pcm_16 = np.round(
        (np.iinfo(np.int16).max * aud)).astype(np.int16).tobytes()
    voices = [
        VAD.is_speech(pcm_16[2 * ix:2 * (ix + SMOOTHING_WSIZE)],
                      sample_rate=SAMPLING_RATE)
        for ix in range(0, len(aud), SMOOTHING_WSIZE)
    ]
    # for i,v in enumerate(voices):
    #     print(v)
    #     exit()
    #     if v:
    #         continue
    #     else:
    #         voice_segments.append(i)
    # voice_segments = np.where(np.diff(voices)!=0)[0]
    # print(voice_segments)

    smoothing_mask = np.repeat(
        binary_dilation(voices, np.ones(SMOOTHING_LENGTH)), SMOOTHING_WSIZE)
    print(len(smoothing_mask))
    aud = aud[smoothing_mask]

    try:
        aud = normalization(aud, norm_type='peak')
        return aud, SAMPLING_RATE

    except AssertionError as e:
        print(AssertionError("Empty audio sig"))
        return aud, SAMPLING_RATE
        # exit()


def mel_spectogram(aud):
    """
    Summary:

    Args:

    Returns:

    """
    mel = librosa.feature.melspectrogram(aud,
                                         sr=SAMPLING_RATE,
                                         n_fft=N_FFT,
                                         hop_length=H_L,
                                         n_mels=MEL_CHANNELS)
    # mel = np.log(mel + 1e-5)
    return mel


def split_audio_ixs(n_samples: int, rate=STEP_SIZE_EM, min_coverage=0.75):
    """
    Summary:

    Args:

    Returns:

    """
    assert 0 < min_coverage <= 1

    # Compute how many frames separate two partial utterances
    samples_per_frame = int((SAMPLING_RATE * WINDOW_STEP / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((SAMPLING_RATE / rate) / samples_per_frame))
    assert 0 < frame_step, "The rate is too high"
    assert frame_step <= H_L, "The rate is too low, it should be %f at least" % \
        (SAMPLING_RATE / (samples_per_frame * H_L))

    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - H_L + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + H_L])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / \
        (last_wav_range.stop - last_wav_range.start)
    if coverage < min_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    return wav_slices, mel_slices


def plot_confusion_matrix(preds, labels, label_names=None, normalize='true'):
    """
    Summary:

    Args:

    Returns:

    """
    cm = confusion_matrix(preds, labels, normalize=normalize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(label_names))
    cmap_min, cmap_max = plt.cm.Blues(0), plt.cm.Blues(256)

    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.title("Ad-Detection Confusion Matrix")
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if bool(normalize.capitalize()):
        fmt = '.2f'
    else:
        fmt = 'd'
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(len(label_names)), range(len(label_names))):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        plt.text(i, j, format(cm[i, j], fmt),
                 horizontalalignment="center", color=color)
    plt.show()


class AdPodTorchDataset(data.Dataset):
    def __init__(self, pod_data, device=torch.device('cpu')):
        self.pod_data = pod_data
        self.categories = sorted(os.listdir(pod_data))
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())
        self.device = device
        self.raw_dir = config['RAW_DATA_DIR']

    def __len__(self):
        return len(self.categories) + int(4e4)

    def _get_wav_parts(self):
        while True:
            rand_cat = randint(0, 1)
            category = self.categories[rand_cat]

            wav_files = os.listdir(os.path.join(self.pod_data, category))
            rand_wav_file = os.path.join(
                self.pod_data, category, sample(wav_files, 1)[0])
            rand_wav_file = rand_wav_file.replace(self.pod_data, self.raw_dir)

            wav, sr = librosa.load(rand_wav_file, sr=None)
            # wav, sr = preprocess_aud(wav, SAMPLING_RATE)
            if len(wav) > 4*SAMPLING_RATE:
                break
        rand_wav_ix = randint(0, len(wav)-(3*SAMPLING_RATE))
        rand_wav = wav[rand_wav_ix:rand_wav_ix+(3*SAMPLING_RATE)]
        mel = mel_spectogram(rand_wav)
        mel = torch.Tensor(mel)
        label = torch.LongTensor([rand_cat]).squeeze()
        return mel, label

    def __getitem__(self, ix=0):
        mel, label = self._get_wav_parts()
        mel = mel.to(device=self.device)
        label = label.to(device=self.device)
        return mel, label


class HDF5TorchDataset(data.Dataset):
    def __init__(self, accent_data, device=torch.device('cpu')):
        hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(accent_data))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.accents = self.hdf5_file.keys()
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())
        self.device = device

    def __len__(self):
        return len(self.accents) + int(1e4)

    def _get_acc_uttrs(self):
        while True:
            rand_accent = sample(list(self.accents), 1)[0]
            if self.hdf5_file[rand_accent].__len__() > 0:
                break
        wavs = list(self.hdf5_file[rand_accent])
        while len(wavs) < self.config['UTTR_COUNT']:
            wavs.extend(sample(wavs, 1))

        rand_wavs = sample(wavs, self.config['UTTR_COUNT'])
        rand_accent_ix = list(self.accents).index(rand_accent)
        rand_uttrs = []
        labels = []
        for wav in rand_wavs:
            wav_ = self.hdf5_file[rand_accent][wav]

            rix = randint(0, wav_.shape[1] - self.config['SLIDING_WIN_SIZE'])

            ruttr = wav_[:, rix:rix + self.config['SLIDING_WIN_SIZE']]

            ruttr = torch.Tensor(ruttr)
            rand_uttrs.append(ruttr)
            labels.append(rand_accent_ix)
        return rand_uttrs, labels

    def __getitem__(self, ix=0):
        rand_uttrs, labels = self._get_acc_uttrs()
        rand_uttrs = torch.stack(rand_uttrs).to(device=self.device)
        labels = torch.LongTensor(labels).to(device=self.device)
        return rand_uttrs, labels

    def collate(self, data):
        pass


def write_hdf5(out_file, data):
    """
    Summary:

    Args:

    Returns:

    """
    gmu_proc_file = h5py.File(out_file, 'w')
    for g in data:
        group = gmu_proc_file.create_group(g)
        for datum in data[g]:
            group.create_dataset("mel_spects_{}".format(datum[0]),
                                 data=datum[1])
    gmu_proc_file.close()


if __name__ == "__main__":
    adpd = AdPodTorchDataset(
        config['PODS_TRAIN_DIR']
    )
    # adpd.__getitem__()
    loader = data.DataLoader(adpd, 4)
    for y in loader:
        print(y[0].shape)
        print(y[1].shape)
