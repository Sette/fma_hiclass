import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler

import os
import json
import ast
import math

from tqdm import tqdm

from sklearn.utils import shuffle
from math import ceil

from utils.dir import create_dir

def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'emb': tf.io.FixedLenFeature([], tf.string),
        'track_id': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    track_id = content['track_id']
    emb = content['emb']

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(emb, out_type=tf.float32)
    return (feature, track_id)


def get_dataset(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


def load_features(path, dataset='music_style'):
    tfrecords_path = os.path.join(path, 'tfrecords', dataset)

    tfrecords_path = [os.path.join(tfrecords_path, path) for path in os.listdir(tfrecords_path)]
    dataset = get_dataset(tfrecords_path)

    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=['feature', 'track_id']
    )

    df.dropna(inplace=True)

    try:
        df.feature = df.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print('Erro ao carregar features')
    return df


def load_dataset_tfrecords(path, dataset='val'):
    tfrecords_path = os.path.join(path, 'tfrecords', dataset)

    tfrecords_path = [os.path.join(tfrecords_path, path) for path in os.listdir(tfrecords_path)]
    dataset = get_dataset(tfrecords_path)
    labels = []
    for level in range(1, max_depth+1):
        labels.append(f'label_{level}')
    
    df = pd.DataFrame(
        dataset.as_numpy_iterator(),
        columns=labels.extend(['feature', 'track_id'])
    )

    df.dropna(inplace=True)

    try:
        df.feature = df.feature.apply(lambda x: x[0] if x.shape[0] != 0 else None)
    except:
        print('Erro ao carregar features')
    return df


def load_dataset(metadata):
    df_val = pd.read_csv(metadata['val_csv'])
    df_test = pd.read_csv(metadata['test_csv'])
    df_train = pd.read_csv(metadata['train_csv'])
    df_train = pd.concat([df_train, df_val])
    
    return df_train, df_test

# Função para converter listas em strings
def convert_list_to_string(lst):
    return '_'.join([str(x) for x in lst])



def __split_data__(group, percentage=0.1):
    if len(group) == 1:
        return group, group

    shuffled = shuffle(group.values)
    finish_test = int(ceil(len(group) * percentage))

    first = pd.DataFrame(shuffled[:finish_test], columns=group.columns)
    second = pd.DataFrame(shuffled[finish_test:], columns=group.columns)

    return first, second


def select_dataset(df, args):
    #     dataset_testset_path = os.path.join(tfrecord_path,'test')
    #     dataset_validationset_path = os.path.join(tfrecord_path,'val')
    #     dataset_trainset_path = os.path.join(tfrecord_path,'train')
    
    # Inicializa listas para armazenar os dados divididos

    #df_features = load_dataset(args.dataset_path, dataset=args.embeddings)

    #df_features.dropna(inplace=True)

    #df = df.merge(df_features, on='track_id')

    tests = []
    trains = []
    validations = []
    
    labels_strings = df['full_genre_id'].apply(convert_list_to_string)
    # Agrupa o DataFrame com base nos rótulos hierárquicos
    groups = df.groupby(labels_strings)
    
    count = 0
    items_count = 0
    total = len(groups)
    total_items = len(df)
    oversampling_size = 2  # int(group_sizes.mean() + group_sizes.std() * 2)
    print(f"oversampling_size: {oversampling_size}")
    
    for code, group in tqdm(groups):
        test, train_to_split = __split_data__(group, 0.2)  # 20%
        train_to_split = train_to_split
        validation, train = __split_data__(train_to_split, 0.1)  # %10
    
        tests.append(test)
        validations.append(validation)
    
        ## this increase the numner of samples when classes has low quantity
        count_train = len(train)
        if count_train < oversampling_size:
            print(f'Oversampling: {train.full_genre_id.iloc[0]}')
            train = train.sample(oversampling_size, replace=True)
    
        trains.append(train)
    
        count += 1
        items_count += count_train

    df_test = pd.concat(tests, sort=False).sample(frac=1).reset_index(drop=True)
    # .to_csv(dataset_testset_path, index=False,quoting=csv.QUOTE_ALL)
    df_val = pd.concat(validations, sort=False).sample(frac=1).reset_index(drop=True)
    df_train = pd.concat(trains, sort=False).sample(frac=1).reset_index(drop=True)

    return df_train, df_test, df_val


def _bytes_feature(value):
    ### Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    ### Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64List_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature(value):
    ###  Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array
    

def parse_single_music(data, labels):
    track_id, categories, music = data
    max_depth = len(categories[0])
    #print(categories)
    data = {}
    for level in range(1, max_depth+1):
        level_labels = []
        for cat in categories:
            label = labels[f'label_{level}'][cat[level-1]]
            #print(cat[level-1], label)
            #label = np.array([label, labels[f'label_{level}_count']], np.int64)
            level_labels.append(label)
        data[f'label{level}'] =  _int64List_feature(level_labels)

    data['features'] = _float_feature(music)
    data['track_id'] = _int64_feature(track_id)
    
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def generate_tf_record(df, df_features, args, tf_path='val'):
    create_dir(tf_path)
    
    df = df.merge(df_features, on='track_id')

    batch_size = 1024 * 50  # 50k records from each file batch
    count = 0
    total = math.ceil(len(df) / batch_size)

    for i in range(0, len(df), batch_size):
        batch_df = df[i:i + batch_size]

        tfrecords = [parse_single_music(data, args.labels) for data in batch_df.values]

        path = f"{tf_path}/{str(count).zfill(10)}.tfrecord"

        # with tf.python_io.TFRecordWriter(path) as writer:
        with tf.io.TFRecordWriter(path) as writer:
            for tfrecord in tfrecords:
                writer.write(tfrecord.SerializeToString())

        print(f"{count} {len(tfrecords)} {path}")
        count += 1
        print(f"{count}/{total} batchs / {count * batch_size} processed")

    print(f"{count}/{total} batchs / {len(df)} processed")

    return tf_path

def pre_process(df, features, columns, multi_label=False, verbose=False):
    if not multi_label:
        # Assign an integer value to each genre.
        enc = LabelEncoder()
        labels = df['track', 'genre_top']
        #y = enc.fit_transform(tracks['track', 'genre_top'])
    else:
        # Create an indicator matrix.
        enc = MultiLabelBinarizer()
        index = df.track_id
        labels = df['full_genre_id']
        #labels = tracks['track', 'genres']

    # Split in training, validation and testing sets.
    #y = enc.fit_transform(labels)
    y = labels.values
    X = features.loc[index, columns].values
    
    X, y = shuffle(X, y, random_state=42)
    
    # Standardize features by removing the mean and scaling to unit variance.
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(X)
    
    return X, y

# def pre_process(tracks, features, columns, multi_label=False, verbose=False):
#     if not multi_label:
#         # Assign an integer value to each genre.
#         enc = LabelEncoder()
#         labels = tracks['track', 'genre_top']
#         #y = enc.fit_transform(tracks['track', 'genre_top'])
#     else:
#         # Create an indicator matrix.
#         enc = MultiLabelBinarizer()
#         labels = tracks['track', 'genres_all']
#         #labels = tracks['track', 'genres']

#     train = tracks.index[tracks['set', 'split'] == 'training']
#     val = tracks.index[tracks['set', 'split'] == 'validation']
#     test = tracks.index[tracks['set', 'split'] == 'test']

#     # Split in training, validation and testing sets.
#     y_train = enc.fit_transform(labels[train])
#     y_val = enc.transform(labels[val])
#     y_test = enc.transform(labels[test])
#     X_train = features.loc[train, columns].values
#     X_val = features.loc[val, columns].values
#     X_test = features.loc[test, columns].values
    
#     X_train, y_train = shuffle(X_train, y_train, random_state=42)
    
#     # Standardize features by removing the mean and scaling to unit variance.
#     scaler = StandardScaler(copy=False)
#     scaler.fit_transform(X_train)
#     scaler.transform(X_val)
#     scaler.transform(X_test)
    
#     return y_train, y_val, y_test, X_train, X_val, X_test

def create_metadata(args):
    with open(args.metadata_train_path, 'w+') as f:
        levels_size = []
        for lv in range(1, args.max_depth):
            levels_size.append(args.labels[f'label_{lv}_count'])
        f.write(json.dumps({
            'sequence_size': args.sequence_size,
            'max_depth': args.max_depth,
            'levels_size': levels_size,
            'val_path': args.val_path,
            'train_path': args.train_path,
            'test_path': args.test_path,
            'val_csv': args.val_csv,
            'train_csv': args.train_csv,
            'test_csv': args.test_csv,
            'trainset_count': args.train_len,
            'validationset_count': args.val_len,
            'testset_count': args.test_len
        }))

