# create a sample of the data for testing purposes add argument --test for any python file to load the sample data that is genrated

from sklearn import model_selection
import pandas as pd
import numpy as np
import os

def main():

    # read the data 
    data = pd.read_csv(os.path.join('..', 'input', 'train.csv')).sample(frac = 0.05).reset_index(drop = True)

    print(f'Length of the original dataset {len(data)}')

    # data augumention if ( A->B then B!->A)
    data_label1 = data[data['label'] == 1]
    data_label0 = data[data['label'] == 0]

    data_label_change = data_label1.rename(columns={'pre requisite': 'concept', 'concept': 'pre requisite'}, inplace=False)

    data_label0 = data_label0.append(data_label_change).drop_duplicates(subset=['concept', 'pre requisite'])
    
    data_label0['label'] = 0 # change the label for the agumentation
    
    data = pd.concat([data_label0, data_label1])

    print(f'Length of the augumented dataset {len(data)}')

    print(data['label'].value_counts())

    # seperate into folds

    data['kfold'] = -1

    data = data.sample(frac = 1).reset_index(drop = True)

    y = data.label.values

    print(y)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X = data, y = y)):
        
        data.loc[v_, 'kfold'] = f

    print(data['kfold'].value_counts())

    data.to_csv(os.path.join('..', 'input', 'train_folds_test.csv'), index = False)

if __name__ == '__main__':
    main()




