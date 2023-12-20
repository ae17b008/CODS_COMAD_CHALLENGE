print("starting filee")
# from sentence_transformers import SentenceTransformer
import os
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer
import argparse
print("import is done")

# initialize the parser
parser = argparse.ArgumentParser(description='Downloading the model locally')

# add the parameter to select the model, it is a string
parser.add_argument('--model', '-m', required = True, type =str, help = 'Select the appropriate model' )

args = parser.parse_args()

print(args)

modelPath = os.path.join('..', 'models', args.model)

model_name = args.model

# Step 2: Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Step 3: Save the tokenizer and model to the specified directory
tokenizer.save_pretrained(modelPath)
model.save_pretrained(modelPath)



# modelPath = os.path.join('..', 'models', 'stsb-distilbert-base') 

# model = SentenceTransformer('stsb-distilbert-base')
# model.save(modelPath)
# model = SentenceTransformer(modelPath)
# print(model)