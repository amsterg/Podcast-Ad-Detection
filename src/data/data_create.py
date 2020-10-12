import json
import os
import pathlib
import random
import sys
import warnings
from collections import Counter
from math import ceil, floor
from multiprocessing import Pool, cpu_count
from random import sample, shuffle

import h5py
import librosa
import numpy as np
import pandas as pd
import soundfile
import wget
from colorama import Fore
from mutagen.mp3 import MP3 as MP3_META
from mutagen.wave import WAVE as WAVE_META
from mutagen.wavpack import WavPack
from scipy.ndimage import binary_dilation
from tqdm import tqdm
from yaml import safe_load

from data_utils import mel_spectogram, preprocess_aud, structure, write_hdf5

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
AUDIO_WRITE_FORMAT = config['AUDIO_WRITE_FORMAT']
AUDIO_READ_FORMAT_PODS = config['AUDIO_READ_FORMAT_PODS']
PODS_READ_FORMAT_PODS = config['PODS_READ_FORMAT_PODS']

PODS_DIR = config['PODS_DIR']
ADS_DIR = config['ADS_DIR']
NON_ADS_DIR = config['NON_ADS_DIR']
PODS_TRAIN_DIR: config['PODS_TRAIN_DIR']
PODS_VAL_DIR: config['PODS_VAL_DIR']
PODS_TEST_DIR: config['PODS_TEST_DIR']


ADS_OUT_FILE_TRAIN = os.path.join(
    PROC_DATA_DIR, 'ads_train.hdf5')


dirs_ = set([globals()[d] for d in globals() if d.__contains__('DIR')])



def preprocess_TIMIT(data_root, out_file):
    """
    Summary:

    Args:

    Returns:

    """

    categories = os.listdir(data_root)
    speakers = [[
        os.path.join(data_root, c, s)
        for s in os.listdir(os.path.join(data_root, c))
    ] for c in categories]

    speakers = [s for c in speakers for s in c]

    wavs = [[
        os.path.join(s, w) for w in os.listdir(s)
        if w.__contains__(config['AUDIO_READ_FORMAT_TIMIT'])
    ] for s in speakers]

    wavs = [s for c in wavs for s in c]

    count = 0
    shuffle_ixs = list(range(len(wavs)))
    shuffle(shuffle_ixs)
    wavs = np.array(wavs)[shuffle_ixs][:].tolist()

    mels = {c: [] for c in categories}

    for wav_fname in tqdm(wavs,
                          bar_format="{l_bar}%s{bar}%s{r_bar}" %
                          (Fore.GREEN, Fore.RESET)):
        try:
            aud = preprocess(wav_fname)
        except AssertionError as e:
            print(e, "Couldn't process ", len(aud), wav_fname)
            continue

        # file_out_ = wav_fname.split('.')[0].replace(
        #     RAW_DATA_DIR, INTERIM_DATA_DIR) + '_' + AUDIO_WRITE_FORMAT
        # os.makedirs('/'.join(file_out_.split('/')[:-1]), exist_ok=True)
        # soundfile.write(file_out_, aud, config['SAMPLING_RATE'])

        mel = mel_spectogram(aud)
        if mel.shape[1] <= config['SLIDING_WIN_SIZE']:
            print("Couldn't process ", mel.shape, wav_fname)
            continue
        c = wav_fname.split('/')[-3]
        s = '_'.join(wav_fname.split('/')[-2:]).split('.')[0]
        mels[c].append((s, mel))

    write_hdf5(out_file, mels)


def extract_ad(pod_info, create_ads=False, create_non_ads=False):
    """
    Summary:

    Args:

    Returns:

    """
    pod_info['fname'] = wget.filename_from_url(pod_info['content_url'])
    if os.path.exists(os.path.join(PODS_DIR, pod_info['fname'])):
        pod_meta = MP3_META(os.path.join(PODS_DIR, pod_info['fname']))
        pod_file_length = float(pod_meta.info.length)
        try:
            pod_length_json = pod_info['content_duration']
            if pod_length_json.count(':') == 1:
                pod_length = [float(x) for x in pod_length_json.split(':')]
                pod_length_json = pod_length[0]*60 + pod_length[1]
            elif pod_length_json.count(':') == 2:
                pod_length = [float(x) for x in pod_length_json.split(':')]
                pod_length_json = pod_length[0]*3600 + \
                    pod_length[1]*60 + pod_length[2]
            else:
                pod_length_json = float(pod_length_json)
        except ValueError as e:
            print(e, pod_info['fname'])
            pod_length_json = pod_file_length
        # if abs(pod_file_length-pod_length_json) < 5:
        if pod_file_length <= pod_length_json:
            # if True:
            # print('Extracting ad from {}'.format(pod_info['fname']))
            if create_non_ads or create_ads:
                pod_aud, pod_sr = librosa.load(os.path.join(
                    PODS_DIR, pod_info['fname']), sr=None)
                pod_aud, pod_sr = preprocess_aud(pod_aud, pod_sr)
                # return 1
                aud_len = len(pod_aud)
                ad_slices = []
                non_ad_aud = np.array([])
                ad_stop = 0
                for i, ad in enumerate(pod_info['ads']):
                    ad_slice = slice(
                        floor(int(ad['ad_start'])*pod_sr), ceil((int(ad['ad_end'])+1)*pod_sr))
                    ad_aud = pod_aud[ad_slice.start:ad_slice.stop]
                    ad_slices.append(ad_slice)
                    non_ad_aud = np.append(
                        non_ad_aud, pod_aud[ad_stop:ad_slice.start])
                    ad_stop = ad_slice.stop
                    ad_fname = os.path.join(ADS_DIR, "{}_{}.wav".format(
                        pod_info['fname'].split('.')[0], i))

                    if not os.path.exists(ad_fname) and create_ads:
                        soundfile.write(ad_fname, ad_aud, pod_sr, format='WAV')
                if ad_slice.stop < aud_len:
                    non_ad_aud = np.append(non_ad_aud, pod_aud[ad_slice.stop:])
                # print(ad_slices)
                ad_ranges = [x for y in [list(range(ad_slice.start, ad_slice.stop))
                                         for ad_slice in ad_slices] for x in y]

                try:
                    assert len(ad_ranges)+len(non_ad_aud) <= aud_len
                    non_ad_fname = os.path.join(NON_ADS_DIR, "{}_content.wav".format(
                        pod_info['fname'].split('.')[0]))
                    if not os.path.exists(non_ad_fname) and create_non_ads:
                        soundfile.write(non_ad_fname, non_ad_aud,
                                        pod_sr, format='WAV')
                except AssertionError as ae:
                    print("{} Aud,ad length mismatch".format(pod_info['fname']),
                          len(ad_ranges), len(non_ad_aud), aud_len, aud_len-len(ad_ranges)+len(non_ad_aud))
        else:
            print('Skipping {} length mismatch'.format(pod_info['fname']))


def extract_ads(pod_file, create_ads=False, create_non_ads=False):
    """
    Summary:

    Args:

    Returns:

    """
    pool = Pool(int(cpu_count()/2))

    with open(pod_file, 'r') as f:
        podcasts_json = json.load(f)


    proc_args = {
        'pod_file': podcasts_json,
        'create_ads': create_ads,
        'create_non_ads': create_non_ads
    }
    from functools import partial
    extract_ad_p = partial(extract_ad, create_ads=create_ads,
                           create_non_ads=create_non_ads)
    pool.map(extract_ad_p, podcasts_json[:])


def download_pod(pod_url):
    """
    Summary:

    Args:

    Returns:

    """
    fname = wget.filename_from_url(pod_url)
    if not os.path.exists(os.path.join(PODS_DIR, fname)):
        try:
            print("Downloading ", fname)
            wget.download(pod_url, PODS_DIR, bar=None)
        # break
        except Exception as e:
            print(e, fname)


def download_pods(pod_file):
    """
    Summary:

    Args:

    Returns:

    """
    pool = Pool(int(cpu_count()/2))

    errored_files = []
    with open(pod_file, 'r') as f:
        podcasts_json = json.load(f)
    podcast_urls = [podcast['content_url'] for podcast in podcasts_json]
    pool.map(download_pod, podcast_urls[:])

    # errored_files.append(fname)
    # print(errored_files)


def greedy_file_select(aud_len, aud_files):
    """
    Summary:

    Args:

    Returns:

    """
    len_sofar = 0
    files_sofar = []
    total_aud_ixs = list(range(len(aud_files)))

    while True:
        rand_ix = random.sample(total_aud_ixs, 1)[0]
        total_aud_ixs.remove(rand_ix)
        aud_file = aud_files[rand_ix]
        aud_file_len = WAVE_META(aud_file).info.length
        if len_sofar+aud_file_len >= aud_len:
            break
        else:
            files_sofar.append(aud_file)
            len_sofar += aud_file_len
    return files_sofar, len_sofar


def link_files_raw_interim(src_files, data_cat, split):
    """
    Summary:

    Args:

    Returns:

    """
    data_split_category_dir = os.path.join(
        config['PODS_{}_DIR'.format(split.upper())], data_cat)
    os.makedirs(data_split_category_dir, exist_ok=True)
    for src_file in src_files:
        os.symlink(src_file, os.path.join(
            data_split_category_dir, src_file.split('/')[-1]))


def split_pods_train_test(data_root):
    """
    Summary:

    Args:

    Returns:

    """
    aud_files = [
        os.path.join(data_root, s)
        for s in os.listdir(data_root)
        if s.__contains__(config['AUDIO_READ_FORMAT_PODS'])
    ]

    aud_files_lens = {i: WAVE_META(
        fname).info.length for i, fname in enumerate(aud_files)}
    total_aud_len = sum(aud_files_lens.values())
    train_aud_len = total_aud_len*0.8

    random.seed(13)

    train_files, train_aud_len = greedy_file_select(train_aud_len, aud_files)

    non_train_files = list(set(aud_files)-set(train_files))

    val_aud_len = (total_aud_len-train_aud_len)*0.5
    val_files, val_aud_len = greedy_file_select(val_aud_len, non_train_files)

    test_files = list(set(aud_files)-set(train_files)-set(val_files))
    test_aud_len = (total_aud_len-train_aud_len)*0.5

    test_files, test_aud_len = greedy_file_select(test_aud_len, test_files)

    rem_files = list(set(aud_files)-set(train_files) -
                     set(val_files)-set(test_files))
    rem_aud_len = total_aud_len-train_aud_len-val_aud_len-test_aud_len

    val_files += rem_files
    val_aud_len += rem_aud_len

    assert train_aud_len+val_aud_len+test_aud_len == total_aud_len

    shuffle_ixs = list(range(len(aud_files)))
    random.shuffle(shuffle_ixs)
    train_ixs = shuffle_ixs[:int(len(shuffle_ixs)*0.8)]
    val_ixs = list(set(shuffle_ixs)-set(train_ixs))
    val_ixs = val_ixs[:int(len(val_ixs)*0.5)]
    test_ixs = list(set(shuffle_ixs)-set(train_ixs)-set(val_ixs))

    assert len(test_ixs)+len(val_ixs)+len(train_ixs) == len(shuffle_ixs)

    train_aud_files = np.array(aud_files)[train_ixs][:].tolist()
    val_aud_files = np.array(aud_files)[val_ixs][:].tolist()
    test_aud_files = np.array(aud_files)[test_ixs][:].tolist()

    data_category = data_root.split('/')[-1]

    link_files_raw_interim(train_aud_files, data_category, 'train')
    link_files_raw_interim(val_aud_files, data_category, 'val')
    link_files_raw_interim(test_aud_files, data_category, 'test')


def preprocess_pods(data_root, out_file):
    """
    Summary:

    Args:

    Returns:

    """

    aud_files = [
        os.path.join(data_root, s)
        for s in os.listdir(data_root)
        if s.__contains__(config['AUDIO_READ_FORMAT_PODS'])
    ]
    

if __name__ == "__main__":
    structure(dirs_)

    download_pods(config['PODS_DATA_INFO'])
    
    extract_ads(config['PODS_DATA_INFO'], create_ads=True, create_non_ads=True)

    split_pods_train_test(ADS_DIR)
    split_pods_train_test(NON_ADS_DIR)

