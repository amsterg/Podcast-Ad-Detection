import argparse
import json
import os
import sys
from collections import Counter
from math import ceil, floor

import hdbscan
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import wget
from mutagen.mp3 import MP3 as MP3_META
from mutagen.wave import WAVE as WAVE_META
from scipy.ndimage import binary_dilation
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, confusion_matrix
from tqdm import tqdm
from yaml import safe_load
from umap.umap_ import UMAP
import umap
import matplotlib.pyplot as plt
from time import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # noqa
sys.path.append(os.getcwd())  # noqa

from src.data.data_utils import (H_L, STEP_SIZE_EM,
                        mel_spectogram, preprocess_aud,
                        split_audio_ixs, plot_confusion_matrix,
                        load_audio, AdPodFileTorchDataset, structure, detect_voices)  # noqa
from encoder import DIARIZE_ENCODER  # noqa
np.random.seed(42)


with open('src/config.yaml', 'r') as f:
    config_yml = safe_load(f.read())


def greedy_outliers(labels, percent=0.2):
    """
    Greedily select least occuring speakers

    Args:

    Returns:

    """
    greedy_outliers_ = []

    for i, count in Counter(labels).most_common()[::-1]:
        greedy_outliers_.append((i, count))
        if sum([g[1] for g in greedy_outliers_]) / len(labels) >= percent:
            greedy_outliers_ = greedy_outliers_[:-1]
            break

    greedy_outlier_labels = [g[0] for g in greedy_outliers_]
    return greedy_outlier_labels


def get_cont_segs(labels):
    """
    Create contiguous segments of audio from labels

    Args:

    Returns:

    """
    start = 0
    cont_segs = []
    cont_segs_labels = []

    for i, c in enumerate(labels):
        if i > 0 and labels[i] != labels[i-1]:
            end = i
            cont_seg = (start, end-1)
            cont_segs.append(cont_seg)
            cont_segs_labels.append(labels[i-1])

            start = i
    cont_segs.append((start, len(labels)-1))
    cont_segs_labels.append(labels[-1])

    return cont_segs, cont_segs_labels


def segment_ads(aud, aud_splits, fname, labels):
    """
    Extract ads from the audio with the provided labels

    Args:

    Returns:

    """
    fname_rel = fname.split('/')[-1].split('.')[0]
    ads = []
    labels = [mode(labels[i:i+10], axis=None).mode[0]
              for i in range(0, len(labels), 1)]

    greedy_outlier_labels = greedy_outliers(labels, 0.8)

    cont_segs, cont_segs_labels = get_cont_segs(labels)

    assert len(cont_segs) == len(cont_segs_labels)

    with open(os.path.join(config_yml['ADS_VIS_DIR'], '{}_{}.json'.format(fname_rel, "cont_segs")), 'w') as f:
        json.dump(cont_segs, f)

    for i, seg in enumerate(cont_segs):
        aud_slice = aud[aud_splits[seg[0]].start:aud_splits[seg[1]].stop]
        # and np.mean(detect_voices(aud_slice)) > 0.7:
        if cont_segs_labels[i] in greedy_outlier_labels and 12 < len(aud_slice)/config_yml['SAMPLING_RATE'] < 120:
            ads.append(
                ('{}_{}.wav'.format(fname_rel, i).format(i), aud_slice)
            )
            # soundfile.write(
            #     '{}/{}_{}.wav'.format(config_yml['ADS_VIS_DIR'], fname_rel, i), aud_slice, config_yml['SAMPLING_RATE'], format='WAV')
    return ads


def metrics(model, data_iterator):
    """
    Summary:

    Args:

    Returns:

    """
    umap_proj = UMAP(metric='euclidean', n_neighbors=200, low_memory=True)
    hdb_clusterer = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=2,
    )
    ads_pred = []
    ads_actual = []
    total_duration = []
    pred_ads_duration = []
    for i, (data, labels) in tqdm(enumerate(data_iterator)):
        aud_len = MP3_META(data).info.length
        total_duration.append(aud_len)
        aud_data = load_audio(data)
        embeds, (aud_splits, _ ) = encoder.embed(aud_data, group=False)
        print(data, "Embed done")
        try:
            projs = umap_proj.fit_transform(embeds)
            print(data, "Created Projections")
        except Exception as e:
            print(e)
            continue
        clusters = hdb_clusterer.fit_predict(projs)
        print(data, "Created Clusters")

        ads = segment_ads(aud_data, aud_splits, data, clusters)
        pred_ads_duration.append(len(ads)*10)
        ads_pred.append(len(ads))
        ads_actual.append(labels)
        print(data, "Done segmenting ads")

        plt.scatter(projs[:, 0], projs[:, 1], cmap='Spectral')
        plt.title(str(Counter(clusters)))
        plt.savefig(
            '{}/{}_umap.jpg'.format(config_yml['ADS_VIS_DIR'], data.split('/')[-1]))
        plt.close()
        plt.plot(clusters)
        plt.savefig(
            '{}/{}_hdb_labels.jpg'.format(config_yml['ADS_VIS_DIR'], data.split('/')[-1]))
        plt.close()
        print("ads_pred ", ads_pred)
        print("ads_actual ", ads_actual)
        print("total_duration ", np.sum(total_duration))
        print("pred_ads_duration ", np.sum(pred_ads_duration))
        print("time diff ", (np.sum(total_duration)-np.sum(pred_ads_duration)
                             )/np.sum(total_duration))
        continue
    with open(os.path.join(config_yml['ADS_VIS_DIR'], 'ad_count.json'), 'w') as f:
        json.dump({"ads_pred": ads_pred, "ads_actual": ads_actual}, f)

    



if __name__ == "__main__":
    structure()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpt",
        default=23,
        help="# of the save model cpt to load"

    )
    parser.add_argument("--device",
                        help="cpu or cuda",
                        default='cpu',
                        choices=['cpu', 'cuda'])

    parser.add_argument("--dataset_train",
                        help="path to train_dataset",
                        required=False, default='')
    parser.add_argument("--dataset_val",
                        help="path to val_dataset",
                        required=False, default='')
    parser.add_argument("--mode",
                        help="train or eval",
                        default=eval,
                        choices=['train', 'eval'])
    parser.add_argument(
        "--filedir",
        help="dir with fnames to run similiarity eval,atleast 2, separted by a comma",
        type=str)
    parser.add_argument("--data_dir",
                        default=config_yml['PODS_TRAIN_DIR'],
                        help="directory to load data from ")

    args = parser.parse_args()

    encoder = DIARIZE_ENCODER(device=args.device,
                              load_model=True, mode=eval).to(device=args.device)

    encoder.load_model_cpt(cpt=args.cpt, device=args.device)
    data_iterator = AdPodFileTorchDataset(
        args.data_dir
    )
    metrics(encoder, data_iterator)
