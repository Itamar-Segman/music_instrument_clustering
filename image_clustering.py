import os
import numpy as np
import librosa

# models
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.models import Model

from keras.applications.vgg16 import preprocess_input

# clustering and dimension reduction
from sklearn.cluster import KMeans, SpectralClustering, OPTICS
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.manifold import TSNE


def load_spectrogram_dataset():
    dir_name = 'Spectrogram_DB'
    dataset = []
    for file in os.listdir(dir_name):
        image = tf.keras.preprocessing.image.load_img(f'{dir_name}/{file}', color_mode='rgb',
                                                      target_size=(224, 224))
        image = np.array(image)
        dataset.append(image)
    return dataset


def extract_features(img, model):
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features


if __name__ == '__main__':
    inst_dataset = load_dataset("ItaSeg/instruments_dataset")
    spec_img_datasets = load_spectrogram_dataset()

    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

    features_dataset = []

    for img in spec_img_datasets:
        feature = extract_features(img, model)
        features_dataset.append(feature)

    features_dataset = np.array(features_dataset)
    features_dataset = features_dataset.reshape(-1, 4096)

    # reduce the amount of dimensions in the feature vector
    embeddings_reduced = TSNE(n_components=2, random_state=1,
                              square_distances=True).fit_transform(features_dataset)

    # cluster feature vectors
    kmeans = KMeans(n_clusters=4, random_state=22)
    kmeans.fit(embeddings_reduced)

    clustering = SpectralClustering(n_clusters=4,
                                    assign_labels='discretize', random_state=0).fit(embeddings_reduced)

    opc = OPTICS(min_samples=2).fit(embeddings_reduced)

    kmeans_accuracy = np.mean(kmeans.labels_ == inst_dataset['train']['label_num'])
    sp_cl_accuracy = np.mean(clustering.labels_ == inst_dataset['train']['label_num'])
    opc_accuracy = np.mean(opc.labels_ == inst_dataset['train']['label_num'])
    print(f'KMeans Accuracy: {kmeans_accuracy:.2f}')
    print(f'Spectral Clustering Accuracy: {sp_cl_accuracy:.2f}')
    print(f'OPTIC Accuracy: {opc_accuracy:.2f}')

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
    plt.title('Embedding using VGG16 model')
    plt.show()
