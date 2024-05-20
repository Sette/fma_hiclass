#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import ast
import os
import json
import time
import logging
from datetime import datetime as dt


# In[96]:


from tqdm.notebook import tqdm


# In[97]:


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


# In[98]:


from dataset.dataset import load_features, load_dataset, pre_process


# In[99]:


from utils.data import load 
from utils.dir import create_dir


# In[100]:


def __load_json__(path):
    with open(path, 'r') as f:
        tmp = json.loads(f.read())

    return tmp


# In[101]:


args = pd.Series({
    "root_dir": "/mnt/disks/data/",
    "dataset_path": "/mnt/disks/data/fma/fma_large", 
    "metadata_path": "/mnt/disks/data/fma/fma_metadata", 
    "embeddings": "music_style",
    "sequence_size": 1280,
    "train_id": "hierarchical_hiclass_tworoots"
})



# In[102]:


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

hiclass_path = os.path.join(train_path, 'hiclass_effnet')


# In[103]:


create_dir(hiclass_path)


# In[104]:


metadata = __load_json__(metadata_file)


# In[105]:


labels = __load_json__(labels_file)


# In[106]:


labels


# ### Load splited dataset

# In[107]:


df_train, df_test = load_dataset(metadata)


# ### Norml labels

# In[108]:


def norm_labels(label):
    label = ast.literal_eval(label)
    #label = [[elemento for elemento in sublist if elemento != 0] for sublist in label]
    return label

    


# In[109]:


df_train['full_genre_id'] = df_train.full_genre_id.apply(lambda x: norm_labels(x))#


# In[111]:


df_test['full_genre_id'] = df_test.full_genre_id.apply(lambda x: norm_labels(x))


# ### Get features

# In[112]:


df_features = load_features(args.dataset_path, dataset=args.embeddings)

df_features.dropna(inplace=True)

# In[113]:


def process_dataset(dataset, features):
    df = dataset.merge(features, on='track_id')
    return df


# In[114]:


df_train = process_dataset(df_train, df_features)


# In[115]:


df_test = process_dataset(df_test, df_features)


# In[116]:


df_train.dropna(inplace=True)


# In[117]:


df_test.feature.values


# ## Create classifiers


# #### Preprocessing

# In[124]:


classifiers = {
    #'LR': LogisticRegression(),
    #'kNN': KNeighborsClassifier(n_neighbors=200),
    #'SVCrbf': SVC(kernel='rbf',probability=True),
    #'SVCpoly1': SVC(kernel='poly', degree=1,probability=True),
    #'linSVC1': SVC(kernel="linear",probability=True),
    #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    #'DT': DecisionTreeClassifier(max_depth=5, max_features='sqrt'),
    #'RF': RandomForestClassifier(max_depth=5, n_estimators=100, min_samples_split=5, min_samples_leaf=5,  max_features='log2', random_state=48, class_weight='balanced', max_samples=25),
    #'AdaBoost': AdaBoostClassifier(n_estimators=10),
    #'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
    'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
    #'NB': GaussianNB(),
    #'QDA': QuadraticDiscriminantAnalysis(),
}


def converter_segundos(segundos):
    horas = segundos // 3600
    minutos = (segundos % 3600) // 60
    segundos = segundos % 60
    return horas, minutos, segundos

# In[125]:


def test_classifiers(classifiers):
    columns = list(classifiers.keys()).insert(0, 'dim')
    
    X_train = df_train.feature.values.tolist()
    y_train = df_train.full_genre_id.values
    X_test = df_test.feature.values.tolist()
    y_test = df_test.full_genre_id.values
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
        end_time_ft = converter_segundos(end_time)
        
        # Escreve a duração do treinamento em um arquivo
        with open(os.path.join(clf_path,"time.txt"), "w") as f:
            f.write("Tempo de Treinamento: {}".format(end_time))
        
        df_test_pred.to_csv(os.path.join(clf_path,'predict.csv'))
        print("Tempo de Treinamento:", end_time, "segundos")

# In[126]:


#scores, times, y_pred = test_classifiers_features(classifiers, features, feature_sets, multi_label=True)
test_classifiers(classifiers)

