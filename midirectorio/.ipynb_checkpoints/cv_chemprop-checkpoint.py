from chemprop.data import MoleculeDataset
from chemprop.data.utils import split_data
from chemprop.train import cross_validate, train
from chemprop.args import TrainArgs
from chemprop.train.predict import predict
from chemprop.args import PredictArgs
import pandas as pd
import os

data = pd.read_csv('scaled_df.csv')
data['combined_smiles'] = data['smiles'] + '.' + data['solvent']
data = data[['combined_smiles', 'peakwavs_max']]

data.to_csv('chemprop_data.csv', index=False)


chemprop_train \
  --data_path chemprop_data.csv \
  --dataset_type regression \
  --target_columns peakwavs_max \
  --split_type random \
  --num_folds 5 \
  --metric mae \
  --epochs 20 \
  --batch_size 50 \
  --depth 5 \
  --hidden_size 300 \
  --save_dir random_cv

chemprop_train \
  --data_path chemprop_data.csv \
  --dataset_type regression \
  --target_columns peakwavs_max \
  --split_type scaffold_balanced \
  --num_folds 5 \
  --metric mae \
  --epochs 20 \
  --batch_size 50 \
  --depth 5 \
  --hidden_size 300 \
  --save_dir scaffold_cv
