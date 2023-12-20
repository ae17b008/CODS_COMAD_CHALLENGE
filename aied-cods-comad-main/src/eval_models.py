# script to evaluate models

from sklearn import metrics
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch


# Load the dev dataset
data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
dev_data = data[data['kfold'] == 1][['label', 'pre requisite', 'concept']]

meta_data = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']


def get_transcript(video_name):
    val = meta_data.get(video_name, default=None)
    if val is None :
        return video_name
    else:
        return val 

# model_path = os.path.join('..', 'models', 'firstmodel')
model_path = 'stsb-distilbert-base'

model = SentenceTransformer(model_path)

vf = np.vectorize(lambda x : get_transcript(x))

pre_req_sents = vf(dev_data['pre requisite'].values)

concept_sents = vf(dev_data['concept'].values)

labels = dev_data['label'].values

print(f'Length of Sentence Lists {len(pre_req_sents)}')

pre_req_embds = torch.tensor(model.encode(pre_req_sents, show_progress_bar=True))

concept_embds = torch.tensor(model.encode(concept_sents, show_progress_bar=True))

# similarity_matrix = util.cos_sim(pre_req_embds, concept_embds)

diff = torch.norm(pre_req_embds - concept_embds, dim = 1)

print(diff)

for margin in np.linspace(start=1, stop=20, num = 19):
    print(f'for margin {margin} the accuraty is {metrics.accuracy_score(labels, (diff <= margin))}')



