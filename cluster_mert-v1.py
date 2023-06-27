import os

import numpy as np
from transformers import AutoModel, Wav2Vec2FeatureExtractor
from datasets import load_dataset
import librosa

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, OPTICS
import matplotlib.pyplot as plt
import torch


def preprocess_audio(file_path):
    # Load audio
    audio, _ = librosa.load(file_path, sr=44100)

    resampled_audio = librosa.resample(audio, orig_sr=44100, target_sr=24000)

    return resampled_audio


if __name__ == '__main__':
    inst_dataset = load_dataset("ItaSeg/instruments_dataset")

    # loading our model weights
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
    # loading the corresponding preprocessor config
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True)

    audio_files = 'DB'

    # Process each audio file
    dataset = []
    for file in os.listdir(audio_files):
        spectrogram = preprocess_audio(f'{audio_files}/{file}')
        dataset.append(spectrogram)
    dataset = np.array(dataset)

    inputs = processor(dataset, sampling_rate=24000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    all_layer_hidden_states = torch.stack(outputs.hidden_states).squeeze()

    # for utterance level classification tasks, you can simply reduce the representation in time
    time_reduced_hidden_states = all_layer_hidden_states.mean(-4)
    time_reduced_hidden_states = time_reduced_hidden_states.mean(-2)

    embeddings_reduced = TSNE(n_components=2, random_state=1,
                              square_distances=True).fit_transform(time_reduced_hidden_states)

    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(embeddings_reduced)

    clustering = SpectralClustering(n_clusters=4,
                                    assign_labels='discretize', random_state=0).fit(embeddings_reduced)

    opc = OPTICS(min_samples=2).fit(embeddings_reduced)

    kmeans_accuracy = np.mean(kmeans.labels_ == inst_dataset['train']['label_num'])
    sp_cl_accuracy = np.mean(clustering.labels_ == inst_dataset['train']['label_num'])
    opc_accuracy = np.mean(opc.labels_ == inst_dataset['train']['label_num'])

    print(f'KMeans Accuracy: {kmeans_accuracy:.2f}')
    print(f'Spectral Clustering Accuracy: {sp_cl_accuracy:.2f}')
    print(f'OPTIC Accuracy: {opc_accuracy:.2f}')

    labels = ['Keyboards', 'Strings', 'Winds', 'Percassions']
    emd = np.c_[embeddings_reduced, inst_dataset['train']['label_num']]
    keys = emd[emd[:, -1] == 0]
    strings = emd[emd[:, -1] == 1]
    winds = emd[emd[:, -1] == 2]
    perc = emd[emd[:, -1] == 3]
    plt.scatter(keys[:, 0], keys[:, 1], color='blue', label='Keyboards')
    plt.scatter(strings[:, 0], strings[:, 1], color='red', label='Strings')
    plt.scatter(winds[:, 0], winds[:, 1], color='green', label='Winds')
    plt.scatter(perc[:, 0], perc[:, 1], color='yellow', label='Percussions')
    plt.legend()
    plt.title('Embedding using MERT-v1 model')
    plt.show()
