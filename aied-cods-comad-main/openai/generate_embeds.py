from datasets import load_dataset
import os
import torch
import pandas as pd
import numpy as np 
import argparse
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
import openai
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm
import logging
from tqdm import tqdm, tqdm_pandas
from secret_key import openapi_key


def _get_gpt_embedding(example, stratagy, meta_data_extracted, meta_data_transcript):
    '''
    example: A single example from the dataset
    code: Boolean, whether to get the code or the docstring embedding

    Get the embedding for a single example from the dataset ( either from gpt or from codebert )
    '''
    
    # text retrieval stratagy
    if stratagy == 'videonames_concatenated':
        text = example['pre requisite'] + " " + example['concept']
    
    if stratagy == 'videonames_extracted_concatenated':
        video_A = example['pre requisite']
        video_B = example['concept']
        text = example['pre requisite'] + " " + example['concept'] + " " 
        text += meta_data_extracted.get(video_A, ' ')
        text += meta_data_extracted.get(video_B, ' ')

    return torch.tensor(openai.Embedding.create(input = [text], model=args.model)['data'][0]['embedding'])
    # return torch.rand(512)


# def get_embeddings(model, batch_df, code, args):
#     '''
#     batch: A batch of Data
#     model : codebert or gpt model
#     code: Boolean, whether to get the code or the docstring embedding

#     Generate a batch of embeddings ( len(batch) X size_of_embedding )
#     '''

#     batch_df['embedding'] = batch_df.apply(lambda x: _get_gpt_embedding(x, args), axis=1)

#     return torch.stack((batch_df['embedding']).to_list())

if __name__ == "__main__":

    # get the arguments
    parser = argparse.ArgumentParser()
    
    
    openai.api_key = openapi_key

    #get the strategy
    parser.add_argument('--strategy', type=str, default='videonames_extracted_concatenated', help='The strategy to use for text retrieval')
    parser.add_argument('--model', type=str, default='text-embedding-ada-002', help='The model to use for text retrieval')
    args = parser.parse_args()

    # get the data
    train_data = pd.read_csv(os.path.join("..", "input", "train.csv"))
    test_data = pd.read_csv(os.path.join("..", "input", "test.csv"))
        
    meta_data_transcript = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['transcript']
    meta_data_extracted = pd.read_csv(os.path.join('..', 'input', 'metadata.csv'), index_col='video name')['extracted']


    # get the embeddings
    train_data['embedding'] = train_data.apply(lambda x: _get_gpt_embedding(x, args.strategy, meta_data_extracted, meta_data_transcript), axis=1)
    test_data['embedding'] = test_data.apply(lambda x: _get_gpt_embedding(x, args.strategy, meta_data_extracted, meta_data_transcript), axis=1)

    # save the embeddings using torch.save
    torch.save(train_data['embedding'], os.path.join("..", "output", "embeddings" , f"train_embeds_{args.strategy}.pt"))
    torch.save(test_data['embedding'], os.path.join("..", "output", "embeddings" , f"test_embeds_{args.strategy}.pt"))
    # train_data.to_csv(os.path.join("..", "output", "embeddings" , f"train_embeds_{args.strategy}.csv"))
    # test_data.to_csv(os.path.join("..", "output", "embeddings" , f"test_embeds_{args.strategy}.csv"))



