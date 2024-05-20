#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import shutil
import json
import ast
import os
from tqdm.notebook import tqdm

from utils.dir import create_dir, __load_json__
from dataset.labels import __create_labels__, get_labels_name, split_label, get_all_structure, convert_label_to_string, parse_label
from dataset.dataset import select_dataset, generate_tf_record, create_metadata, load_features

# In[2]:


tqdm.pandas()

# In[3]:

def get_labels_per_level():
    print('teste')

def remover_sublistas_redundantes(lista_de_listas):
    elementos_vistos = set()
    sublistas_filtradas = []

    for sublista in lista_de_listas:
        if not any(elem in elementos_vistos for elem in sublista):
            sublistas_filtradas.append(sublista)
            elementos_vistos.update(sublista)

    return sublistas_filtradas


def prepare_paths(args):
    ## Define job paths
    fma_path = os.path.join(args.root_dir,"fma")
    job_path = os.path.join(fma_path,"trains")
    args['job_path'] = os.path.join(job_path, args.train_id)
    args['tfrecord_path'] = os.path.join(args.job_path, "tfrecords")
    args['models_path'] = os.path.join(args.root_dir, "models")
    args['metadata_path'] = os.path.join(fma_path, "fma_metadata")
    args['metadata_train_path'] = os.path.join(args['job_path'], "metadata.json")
    args['categories_labels_path'] = os.path.join(args['job_path'], "labels.json")

    ## Remove files from path
    #shutil.rmtree(args['job_path'])
    
    # In[8]:

    ## Create poth if it isn't exist
    create_dir(args['job_path'])
    # Load tracks_df
    tracks_df = pd.read_csv(os.path.join(args.metadata_path, 'tracks_valid.csv'))

    # Loand genres df
    genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # In[13]:

    ## Get sample size from args parameter
    tracks_df = tracks_df.sample(frac=args.sample_size)

    tracks_df["track_genres_all"] = tracks_df.track_genres_all.apply(lambda x: ast.literal_eval(x))
    tracks_df.drop(columns=['track_genres'], inplace=True)
    tracks_df.dropna(inplace=True)
    tracks_df.rename(columns={'track_id_':'track_id'},inplace=True)


    return args, tracks_df

def prepare_labels(tracks_df,args):
    ##### Labels
    # Loand genres df
    genres_df = pd.read_csv(os.path.join(args.metadata_path, 'genres.csv'))
    # Mapear os identificadores numéricos de gêneros para os nomes dos gêneros
    genre_names = dict(zip(genres_df['genre_id'], genres_df['title']))

    # Inicialize uma lista para armazenar todos os caminhos de gêneros para cada exemplo
    estruturas = []

    # Iterar sobre as faixas e seus gêneros associados
    for track_genres in tracks_df['track_genres_all']:
        #caminho_name = [genre_names[genre_id] for genre_id in track_genres]
        caminho_id = [get_all_structure(genre_id, genres_df) for genre_id in track_genres]
        estruturas.append(caminho_id)

    for idx, caminho in enumerate(estruturas):
        caminho.sort(key=len, reverse=True)
        estruturas[idx] = remover_sublistas_redundantes(caminho)

    ## Get structure form hierarchical classification
    tracks_df['full_genre_id'] = estruturas
    tracks_df = tracks_df[['track_id', 'full_genre_id']]

    ## Calculate labels_size
    max_depth = tracks_df.full_genre_id.apply(lambda x: len(x))
    max_depth = int(max_depth.max())
    args['max_depth'] = max_depth
    print(f'max depth: {max_depth}')
    
    labels_name = []
    for level in range(max_depth):
        labels_name.append(f'label_{level+1}')
    tqdm.pandas()
    ## Gnetare categories_df
    #labels =  tracks_df.full_genre_id.progress_apply(lambda x: get_labels_per_level(x))

    #all_levels = categories_df.label5.progress_apply(lambda x: split_label(x))
    all_labels = []
    for idx, row in tqdm(enumerate(tracks_df.full_genre_id)):
        for labels in row:
            labels.extend([0] * (max_depth - len(labels)))
            all_labels.append(labels)
            
    categories_df = pd.DataFrame(all_labels, columns=labels_name).drop_duplicates()

    categories_df[f'label_{max_depth}_name'] = [get_labels_name(categorie, genres_df) for categorie in categories_df.values]

    data = __create_labels__(categories_df, max_depth)

    args['labels'] = data
    # Write labels file
    with open(args.categories_labels_path, 'w+') as f:
        f.write(json.dumps(data))

    return tracks_df, args

def split_dataset(tracks_df,args):
    #### Split dataset

    df_train, df_test, df_val = select_dataset(tracks_df, args)

    args['val_path'] = os.path.join(args.tfrecord_path, 'val')
    args['test_path'] = os.path.join(args.tfrecord_path, 'test')
    args['train_path'] = os.path.join(args.tfrecord_path, 'train')

    args['train_csv'] = os.path.join(args.job_path, "train.csv")
    args['test_csv'] = os.path.join(args.job_path, "test.csv")
    args['val_csv'] = os.path.join(args.job_path, "val.csv")

    df_train.to_csv(args['train_csv'], index=False)
    df_test.to_csv(args['train_csv'], index=False)
    df_val.to_csv(args['val_csv'], index=False)

    df_features = load_features(args.dataset_path, dataset=args.embeddings)

    df_features.dropna(inplace=True)
        
    val_path = generate_tf_record(df_val, df_features, args, tf_path=args['val_path'])
    test_path = generate_tf_record(df_test, df_features, args, tf_path=args['test_path'])
    train_path = generate_tf_record(df_train, df_features, args, tf_path=args['train_path'])

    args['val_len'] = df_val.shape[0]
    args['test_len'] = df_test.shape[0]
    args['train_len'] = df_train.shape[0]

    ## Create metadata file
    create_metadata(args)

   

def run():
    args = pd.Series({
        "root_dir": "/mnt/disks/data/",
        "dataset_path": "/mnt/disks/data/fma/fma_large",
        "embeddings": "music_style",
        "sequence_size": 1280,
        "train_id": "hierarchical_hiclass",
        'sample_size': 1/100
    })

    args, tracks_df = prepare_paths(args)
    tracks_df = prepare_labels(tracks_df,args)
    #split_dataset(tracks_df,args)




