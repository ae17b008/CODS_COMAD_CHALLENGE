from sklearn import model_selection
import pandas as pd
import numpy as np
import os

def main():

    # read the data 
    data = pd.read_csv(os.path.join('..', 'input', 'train.csv'))

    print(f'Length of the original dataset {len(data)}')

    # seperate into folds

    data['kfold'] = -1

    data = data.sample(frac = 1).reset_index(drop = True)

    y = data.label.values

    print(y)

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X = data, y = y)):
        
        data.loc[v_, 'kfold'] = f

    print(data['kfold'].value_counts())

    data.to_csv(os.path.join('..', 'input', 'train_folds_wo_agumentation.csv'), index = False)

if __name__ == '__main__':
    main()




