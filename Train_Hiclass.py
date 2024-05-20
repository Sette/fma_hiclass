#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import ast
import os
import json
import time
import logging
from datetime import datetime as dt

import numpy as np


# In[2]:


from tqdm.notebook import tqdm


# In[3]:


from sklearn import svm
from sklearn import tree
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from hiclass.MultiLabelHierarchicalClassifier import MultiLabelHierarchicalClassifier
from hiclass.MultiLabelLocalClassifierPerNode import MultiLabelLocalClassifierPerNode
from hiclass.metrics import precision, recall, f1

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier


# In[4]:


from dataset.dataset import load_features, load_dataset, pre_process


# In[5]:


from utils.data import load 
from utils.dir import create_dir


# In[6]:


# Configurando o nível de registro
logging.basicConfig(level=logging.INFO)


# In[7]:


def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp


# In[8]:


args = pd.Series({
    "root_dir": "/mnt/disks/data/",
    "dataset_path": "/mnt/disks/data/fma/fma_large", 
    "metadata_path": "/mnt/disks/data/fma/fma_metadata", 
    "embeddings": "music_style",
    "sequence_size": 1280,
    "train_id": "hierarchical_hiclass_tworoots"
})


# In[11]:


job_path = "/mnt/disks/data/fma/trains"

# In[15]:


train_path = os.path.join(job_path, args.train_id)

# In[16]:


base_path = os.path.join(args.root_dir, "fma")

# In[17]:


models_path = os.path.join(args.root_dir, "models")


# In[18]:


metadata_file = os.path.join(train_path, "metadata.json")
labels_file = os.path.join(train_path, "labels.json")
hiclass_path = os.path.join(train_path, 'hiclass_baseline')


# In[12]:


create_dir(hiclass_path)


# In[13]:


metadata = __load_json__(metadata_file)


# In[14]:


labels = __load_json__(labels_file)


# In[15]:


labels


# ### Load baseline features

# In[16]:


features = load(os.path.join(args.metadata_path, 'features.csv'))


# In[17]:


#echonest = load(os.path.join(args.metadata_path, 'echonest.csv'))


# In[30]:


features.columns.values


# ### Load splited dataset

# In[19]:


df_train, df_test = load_dataset(metadata)


# ### Norml labels

# In[20]:


def norm_labels(label):
    label = ast.literal_eval(label)
    #label = [[elemento for elemento in sublist if elemento != 0] for sublist in label]
    return label
    


# In[21]:


df_train['full_genre_id'] = df_train.full_genre_id.apply(lambda x: norm_labels(x))


# In[26]:


df_train


# In[24]:


df_test['full_genre_id'] = df_test.full_genre_id.apply(lambda x: norm_labels(x))


# In[31]:


#features[features.index == 128973]


# ## Create classifiers

# In[32]:


import xgboost as xgb


# In[33]:


xgb_model = xgb.XGBClassifier(random_state=42, eval_metric="auc", n_jobs=20)


# In[34]:


# Show all messages, including ones pertaining to debugging
xgb.set_config(verbosity=0)

# Get current value of global configuration
# This is a dict containing all parameters in the global configuration,
# including 'verbosity'
config = xgb.get_config()
assert config['verbosity'] == 0

# Example of using the context manager xgb.config_context().
# The context manager will restore the previous value of the global
# configuration upon exiting.
assert xgb.get_config()['verbosity'] == 0  # old value restored


# In[35]:


# xgb_model.fit(df_train.Feature.values.tolist()[:100], df_train.Label.values.tolist()[:100], eval_set=[(df_val.Feature.values.tolist(), df_val.Label.values.tolist())])


# In[36]:


xgb_estimator = xgb.XGBClassifier(random_state=42, eval_metric="auc", n_jobs=20)


# #### Preprocessing

# In[46]:


feature_sets = {
   # 'echonest_audio': ('echonest', 'audio_features'),
   # 'echonest_temporal': ('echonest', 'temporal_features'),
    #'mfcc': 'mfcc',
    #'mfcc/contrast/chroma/centroid/tonnetz': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'tonnetz'],
    'mfcc/contrast/chroma/centroid/zcr': ['mfcc', 'spectral_contrast', 'chroma_cens', 'spectral_centroid', 'zcr', 'tonnetz'],
}


# In[47]:


classifiers = {
    #'XGB': xgb_estimator,
    #'LR': LogisticRegression(),
    #'kNN': KNeighborsClassifier(n_neighbors=5),
    #'SVCrbf': SVC(kernel='rbf',probability=True),
    #'SVCpoly1': SVC(kernel='poly', degree=1,probability=True),
    #'linSVC1': SVC(kernel="linear",probability=True),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    #'DT': DecisionTreeClassifier(max_depth=5, max_features='sqrt'),
    #'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #'AdaBoost': AdaBoostClassifier(n_estimators=10),
    #'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
    'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
    #'NB': GaussianNB(),
    #'QDA': QuadraticDiscriminantAnalysis(),
}


# In[48]:


def test_classifiers_features(classifiers, features_all, feature_sets, multi_label=False):
    columns = list(classifiers.keys()).insert(0, 'dim')
    times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
    for fset_name, fset in tqdm(feature_sets.items(), desc='features'):
        X_train, y_train = pre_process(df_train, features_all, fset, multi_label)
        X_test, y_test = pre_process(df_test, features_all, fset, multi_label)
        for clf_name, clf in tqdm(classifiers.items(), desc='classifiers'):  # tqdm_notebook(classifiers.items(), desc='classifiers', leave=False):
            t = time.process_time()
            df_test_pred = df_test.copy(deep=True)
            clf_path = os.path.join(hiclass_path, clf_name)
            create_dir(clf_path)
            logging.info(f'Init training step for {clf_name}')
            hclf = MultiLabelLocalClassifierPerNode(local_classifier=clf, n_jobs=8, verbose=1)
            hclf.fit(X_train, y_train)
            y_pred = hclf.predict(X_test)
            # Convertendo para lista de inteiros
            y_pred = [[[str(num) if num != '' else '' for num in sublist] for sublist in lista] for lista in y_pred]
            df_test_pred['y_pred'] = y_pred
            end_time = time.process_time() - t
            logging.info(f'End training with {end_time}')
            times.loc[fset_name, clf_name] = end_time
            # Escreve a duração do treinamento em um arquivo
            with open(os.path.join(clf_path,"time.txt"), "w") as f:
                f.write("Tempo de Treinamento: {} segundos".format(end_time))
            
            print("Tempo de Treinamento:", end_time, "segundos")
            df_test_pred.to_csv(os.path.join(clf_path,'predict.csv'))
    return times


# In[49]:


#scores, times, y_pred = test_classifiers_features(classifiers, features, feature_sets, multi_label=True)
times = test_classifiers_features(classifiers, features, feature_sets, multi_label=True)


# In[ ]:





# In[ ]:




