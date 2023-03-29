'''
Datasets
'''

import os
import random
import torch
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from mutagen.flac import FLAC
from torch.utils.data import Dataset, DataLoader


class CNCeleb(Dataset):
    def __init__(self, train_list, train_path, num_frames, **kwargs):
        self.train_path = train_path
        self.num_frames = num_frames
        if os.path.exists(train_list):
            print('load {}'.format(train_list))
            df = pd.read_csv(train_list)
            speaker_int_labels = []
            utt_paths = []
            for (utt_path, label) in zip(df["utt_path"].values, df["speaker_int_label"].values):
                if utt_path[-4:] == 'flac':
                    utt_paths.append(utt_path)
                    speaker_int_labels.append(label)
        else:
            utt_tuples, speakers = findAllUtt(train_path, extension='flac', speaker_level=1)
            utt_tuples = np.array(utt_tuples, dtype=str)
            utt_paths = utt_tuples.T[0]
            speaker_int_labels = utt_tuples.T[1].astype(int)
            speaker_str_labels = []
            for i in speaker_int_labels:
                speaker_str_labels.append(speakers[i])

            csv_dict = {"speaker_str_label": speaker_str_labels,
                        "utt_path": utt_paths,
                        "speaker_int_label": speaker_int_labels
                        }
            df = pd.DataFrame(data=csv_dict)
            try:
                df.to_csv(train_list)
                print(f'Saved data list file at {train_list}')
            except OSError as err:
                print(f'Ran in an error while saving {train_list}: {err}')

        # Load data & labels
        self.data_list = utt_paths
        self.data_label = speaker_int_labels
        self.n_class = len(np.unique(self.data_label))
        print("find {} speakers".format(self.n_class))
        print("find {} utterance".format(len(self.data_list)))

    def __getitem__(self, index):
        audio, sr = sf.read(self.data_list[index])
        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = np.pad(audio, (0, shortage), 'wrap')
        start_frame = np.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = np.stack([audio], axis=0)
        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


def findAllUtt(dirName, extension='wav', speaker_level=1):
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)

    # speaker_dict:{speaker_str_label:speaker_int_label}
    # utt_tuple:(utt_path,speaker_int_label)
    speaker_dict = {}
    utt_tuples = []
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speaker_str_label = root[prefixSize:].split(os.sep)[0]
            if speaker_str_label not in speaker_dict.keys():
                speaker_dict[speaker_str_label] = len(speaker_dict)
            speaker_int_label = speaker_dict[speaker_str_label]
            for filename in filtered_files:
                utt_path = os.path.join(root, filename)
                utt_tuples.append((utt_path, speaker_int_label))

    outSpeakers = [None]*len(speaker_dict)
    for key, index in speaker_dict.items():
        outSpeakers[index] = key

    print("find {} speakers".format(len(outSpeakers)))
    print("find {} utterance".format(len(utt_tuples)))

    # return [(utt_path:speaker_int_label), ...], [id00012, id00031, ...]
    return utt_tuples, outSpeakers


if __name__ == "__main__":
    data_dir = '/home2/database/sre/CN-Celeb-2022/task1/cn_2/data'
    train_list_path = 'data/cn2_train_list.csv'
    dataset = CNCeleb(train_list_path, data_dir, 200)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    for idx, batch in enumerate(loader):
        data, label = batch
        print('data:', data.shape, data)
        print('label', label.shape, label)
        break


