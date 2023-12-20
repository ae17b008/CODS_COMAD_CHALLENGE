from datasets import Dataset
import pandas as pd
import os
from transformers import AutoModel, AutoTokenizer
import torch
from transformers.optimization import get_linear_schedule_with_warmup
import argparse

print("Starting script training_hf_spftmaxLoss.py")
print("*************** Script Started *****************\n")


# initialize the parser
parser = argparse.ArgumentParser(description='Training the model')

# add the parameter for selecting the dataset with or without the agumentation with the default being with agumentation also add a shotcut
parser.add_argument('--dataset', '-d', default='without_agumentation', choices=['with_agumentation', 'without_agumentation'], help='Select the dataset with or without the agumentation')

# add the parameter to select how the senteces are to be selected either the transcript or the extracted sentences
parser.add_argument('--sentence_selection', '-s', default='extracted', choices=['transcript', 'extracted'], help='Select the sentences to be used for training')

# add the parameter to select the number of epochs ( int ) default it to 10
parser.add_argument('--epochs', '-e', default=100, type=int, help='Select the number of epochs')

# add the parameter to select the model, it is a string
parser.add_argument('--model', '-m', required = True, type =str, help = 'Select the appropriate model' )

# add the parameter to enable testing the model, it is a boolean
parser.add_argument('--test', '-t', action='store_true', help='Enable testing the model')

# parse the arguments
args = parser.parse_args()

def convert_string_list_to_sentence(string_list):
    # Step 1: Remove outer brackets and spaces
    string_list = string_list.strip('[] ').strip()

    # Step 2: Split the string by commas
    elements_as_strings = string_list.split(',')

    # Step 3: Strip whitespace and quotes from elements
    python_list = [element.strip(" ' ") for element in elements_as_strings]

    return " ".join(python_list)


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
            return convert_string_list_to_sentence(val)
    
# set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# read the train and dev datasets
if args.test:
    data = pd.read_csv(os.path.join('..', 'input', 'train_folds_test.csv'))
elif args.dataset == 'with_agumentation':
    data = pd.read_csv(os.path.join('..', 'input', 'train_folds.csv'))
else:
    data = pd.read_csv(os.path.join('..', 'input', 'train_folds_wo_agumentation.csv'))
# data = pd.read_csv(os.path.join('..', 'input', 'train_folds_wo_agumentation.csv'))
train_df = data[data['kfold'] != 1][['pre requisite', 'concept', 'label']]
dev_df = data[data['kfold'] == 1][['pre requisite', 'concept', 'label']]
meta_data_transcript = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']
meta_data_extracted = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['extracted']


train_dataset = Dataset.from_pandas(train_df).remove_columns(['__index_level_0__']).rename_column('pre requisite', 'pre_requisite')
dev_dataset = Dataset.from_pandas(dev_df).remove_columns(['__index_level_0__']).rename_column('pre requisite', 'pre_requisite')

train_dataset = train_dataset.map(lambda x: {'pre_requisite' : get_sentence(x['pre_requisite']), 'concept' : get_sentence(x['concept'])})

dev_dataset = dev_dataset.map(lambda x: {'pre_requisite' : get_sentence(x['pre_requisite']), 'concept': get_sentence(x['concept'])})
max_length = 1024

# load the tokenizer and model
modelPath = os.path.join('..', 'models', args.model)
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModel.from_pretrained(modelPath).to(device)
model.config.model_max_length = max_length


full_datasets = [train_dataset, dev_dataset]
transformed_datasets = []

for dataset in full_datasets:
    for part in ['pre_requisite', 'concept']:
        dataset = dataset.map(
            lambda x : tokenizer(
                x[part], max_length = 128, padding = 'max_length',truncation = True
            ), batched=True
        )
        for col in ['input_ids', 'attention_mask']:
            dataset = dataset.rename_column(col, part+ '_' + col)
    transformed_datasets.append(dataset)
        
train_dataset, dev_dataset = set(transformed_datasets)

train_dataset.set_format(type='torch', columns=train_dataset.column_names)
dev_dataset.set_format(type='torch', columns=train_dataset.column_names)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size, shuffle = True
)

dev_loader = torch.utils.data.DataLoader(
    dev_dataset, batch_size, shuffle = True
)


def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

ffnn = torch.nn.Linear(768 * 3, 2)
ffnn.to(device)

loss_func = torch.nn.CrossEntropyLoss()


optim = torch.optim.Adam(model.parameters(), lr=2e-5)
# and setup a warmup for the first ~10% steps
total_steps = int(len(train_dataset) / batch_size)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
		optim, num_warmup_steps=warmup_steps,
  	num_training_steps=total_steps - warmup_steps
)

num_epoch = args.epochs
for epoch in range(num_epoch):
    #train
    model.train()
    # initilize a empty torch tenosr to store the predictions
    train_preds = torch.Tensor().to(device)
    train_labels = torch.Tensor().to(device)
    for batch in train_loader:
        optim.zero_grad()
        input_ids_a = batch['pre_requisite_input_ids'].to(device)
        input_ids_b = batch['concept_input_ids'].to(device)
        attention_a = batch['pre_requisite_attention_mask'].to(device)
        attention_b = batch['concept_attention_mask'].to(device)
        label = batch['label'].to(device)
        train_labels = torch.cat([train_labels, label])
        embed_a = model(input_ids_a, attention_a)[0]
        embed_b = model(input_ids_b, attention_b)[0]
        u = mean_pool(embed_a, attention_a)
        v = mean_pool(embed_b, attention_b)

        # build the |u-v| tensor
        uv = torch.abs(torch.abs(u - v))
        # concatenate the [u, v, |u-v|] tensors
        x = torch.cat([u, v, uv], -1)
        # pass through the final linear layer
        x = ffnn(x)
        # compute the loss
        loss = loss_func(x, label)
        loss.backward()
        optim.step()
        scheduler.step()
        # print(f'epoch : {epoch}, loss : {loss.item():.4f}')
        # make predictions
        preds = torch.argmax(x, dim=1)
        train_preds = torch.cat([train_preds, preds])
        # compute the accuracy
    acc = (train_preds == train_labels).float().mean()
    print(f'TRAIN : epoch : {epoch}, acc : {acc.item():.4f}')
    #eval
    eval_preds = torch.Tensor().to(device)
    eval_labels = torch.Tensor().to(device)
    model.eval()
    for batch in dev_loader:
        input_ids_a = batch['pre_requisite_input_ids'].to(device)
        input_ids_b = batch['concept_input_ids'].to(device)
        attention_a = batch['pre_requisite_attention_mask'].to(device)
        attention_b = batch['concept_attention_mask'].to(device)
        label = batch['label'].to(device)
        eval_labels = torch.cat([eval_labels, label])
        embed_a = model(input_ids_a, attention_a)[0]
        embed_b = model(input_ids_b, attention_b)[0]
        u = mean_pool(embed_a, attention_a)
        v = mean_pool(embed_b, attention_b)

        # build the |u-v| tensor
        uv = torch.abs(torch.abs(u - v))
        # concatenate the [u, v, |u-v|] tensors
        x = torch.cat([u, v, uv], -1)
        # pass through the final linear layer
        x = ffnn(x)
        # compute the loss
        loss = loss_func(x, label)
        # print(f'epoch : {epoch}, loss : {loss.item():.4f}')
        # make predictions
        preds = torch.argmax(x, dim=1)
        eval_preds = torch.cat([eval_preds, preds])
        # compute the accuracy
    acc = (eval_preds == eval_labels).float().mean()
    print(f'EVAL : epoch : {epoch}, acc : {acc.item():.4f}')

# save the model usig save_pretrained
model.save_pretrained(f'{args.model}-finetuned')
