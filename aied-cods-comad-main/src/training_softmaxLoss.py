from sentence_transformers import SentenceTransformer, losses, util, evaluation
import pandas as pd
import os
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import argparse

print("Starting script training_SoftmaxLoss.py")
print("*************** Script Started *****************\n")

# initialize the parser
parser = argparse.ArgumentParser(description='Training the model')

# add the parameter for selecting the dataset with or without the agumentation with the default being with agumentation also add a shotcut
parser.add_argument('--dataset', '-d', default='with_agumentation', choices=['with_agumentation', 'without_agumentation'], help='Select the dataset with or without the agumentation')

# add the parameter to select how the senteces are to be selected either the transcript or the extracted sentences
parser.add_argument('--sentence_selection', '-s', default='transcript', choices=['transcript', 'extracted'], help='Select the sentences to be used for training')

# parse the arguments
args = parser.parse_args()

print(args)


def get_sentence(video_name):
    if args.sentence_selection == 'transcript':
        val = meta_data_transcript.get(video_name, default=None)
        if val is None :
            return video_name
        else:
            return val
    if args.sentence_selection == 'extracted':
        val = meta_data_extracted.get(video_name, default=None)
        if val is None :
            return video_name
        else:
            return val

print("*************** Script Started *****************\n")
modelPath = os.path.join('..', 'models', 'stsb-distilbert-base')
model = SentenceTransformer(modelPath, device='cuda')
print("\n\n******* MODEL LOADED ***********")
print(model)
print("********************************")
num_epochs = 100
train_batch_size = 64
# distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
output_path = os.path.join('..', 'models', 'firstmodel')
# create the directory if it doesnt exisit

os.makedirs(output_path, exist_ok=True)


print("******************* load and slice data ********************")
################## LOAD AND SLICE DATA #####################
# depending on the dataset selected load the data
if args.dataset == 'with_agumentation':
    data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
else:
    data = pd.read_csv(os.path.join('..', 'input', 'train_folds_wo_agumentation.csv'))
train_data = data[data['kfold'] != 1]
dev_data = data[data['kfold'] == 1]
meta_data_transcript = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']
meta_data_extracted = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['extracted']



train_samples = []
for _, row in train_data.iterrows():
    sample = InputExample(texts = [get_sentence(row['pre requisite']), get_sentence(row['concept'])], label=int(row['label']))
    train_samples.append(sample)

print(f'Lenght of train set {len(train_samples)}')

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size = train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=2)

################ EVAL ###################

print(f'len of dev_data {len(dev_data)}')

dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
for _, row in dev_data.iterrows():
    dev_sentences1.append(get_sentence(row['pre requisite']))
    dev_sentences2.append(get_sentence(row['concept']))
    dev_labels.append(int(row['label']))


binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels, batch_size=32)

seq_evaluator = evaluation.SequentialEvaluator([binary_acc_evaluator], main_score_function=lambda scores: scores[-1])


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    evaluator=seq_evaluator,
    warmup_steps=1000,
    output_path=output_path,
    show_progress_bar=False
)