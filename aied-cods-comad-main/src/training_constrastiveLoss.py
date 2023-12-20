from sentence_transformers import SentenceTransformer, losses, util, evaluation
import pandas as pd
import os
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

def get_transcript(video_name):
    val = meta_data.get(video_name, default=None)
    if val is None :
        return video_name
    else:
        return val 

print("*************** Script Started *****************\n")
modelPath = os.path.join('..', 'models', 'stsb-distilbert-base')
model = SentenceTransformer(modelPath)
print("\n\n******* MODEL LOADED ***********")
print(model)
print("********************************")
num_epochs = 100
train_batch_size = 64
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
output_path = os.path.join('..', 'models', 'secondmodel')
# create the directory if it doesnt exisit

os.makedirs(output_path, exist_ok=True)

margin = 0.5

print("******************* load and slice data ********************")
################## LOAD AND SLICE DATA #####################
data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
train_data = data[data['kfold'] != 1]
dev_data = data[data['kfold'] == 1]
meta_data = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']


train_samples = []
for _, row in train_data.iterrows():
    sample = InputExample(texts = [get_transcript(row['pre requisite']), get_transcript(row['concept'])], label=int(row['label']))
    train_samples.append(sample)

print(f'Lenght of train set {len(train_samples)}')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size = train_batch_size)
train_loss = losses.OnlineContrastiveLoss(model = model, distance_metric=distance_metric, margin = margin)

################ EVAL ###################

print(f'len of dev_data {len(dev_data)}')

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
for _, row in dev_data.iterrows():
    dev_sentences1.append(get_transcript(row['pre requisite']))
    dev_sentences2.append(get_transcript(row['concept']))
    dev_labels.append(int(row['label']))


binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels, batch_size=32)

seq_evaluator = evaluation.SequentialEvaluator([binary_acc_evaluator], main_score_function=lambda scores: scores[-1])

seq_evaluator(model, epoch=0, steps=0, output_path=output_path)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=seq_evaluator,
    epochs = num_epochs,
    warmup_steps=10,
    output_path=output_path,
    show_progress_bar=False
)