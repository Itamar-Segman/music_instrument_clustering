import os
import numpy as np
import librosa

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering, OPTICS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from datasets import load_dataset


def preprocess_audio(file_path):
    # Load audio
    audio, _ = librosa.load(file_path, sr=44100)

    freq = np.abs(librosa.stft(audio))

    # Convert to Mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(y=np.abs(freq), sr=44100)

    return spectrogram


if __name__ == '__main__':
    inst_dataset = load_dataset("ItaSeg/instruments_dataset")

    audio_files = 'DB'

    # Process each audio file
    dataset = []
    for file in os.listdir(audio_files):
        spectrogram = preprocess_audio(f'{audio_files}/{file}')
        dataset.append(spectrogram)
    dataset = np.array(dataset)

    dataset_reduced = dataset.mean(-1)
    dataset_reduced = dataset_reduced.mean(-1)

    embeddings_reduced = TSNE(n_components=2, random_state=1,
                              square_distances=True).fit_transform(dataset_reduced)

    kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(dataset_reduced)

    clustering = SpectralClustering(n_clusters=4,
                                    assign_labels='discretize', random_state=0).fit(dataset_reduced)

    opc = OPTICS(min_samples=2).fit(dataset_reduced)

    X_train, X_test, y_train, y_test = train_test_split(dataset_reduced, inst_dataset['train']['label_num'],
                                                        test_size=0.33, random_state=42)
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    rf_accuracy = clf.score(X_test, y_test)
    rf_train_accuracy = clf.score(X_train, y_train)

    kmeans_accuracy = np.mean(kmeans.labels_ == inst_dataset['train']['label_num'])
    sp_cl_accuracy = np.mean(clustering.labels_ == inst_dataset['train']['label_num'])
    opc_accuracy = np.mean(opc.labels_ == inst_dataset['train']['label_num'])

    print(f'KMeans Accuracy: {kmeans_accuracy:.2f}')
    print(f'Spectral Clustering Accuracy: {sp_cl_accuracy:.2f}')
    print(f'OPTIC Accuracy: {opc_accuracy:.2f}')

    print(f'Random Forest Train Accuracy: {rf_train_accuracy:.2f}')
    print(f'Random Forest Test Accuracy: {rf_accuracy:.2f}')

    # Plot
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
    plt.title('Embedding using Spectrogram')
    plt.show()
