#!/bin/bash

# Predicción 
echo "Haciendo predicción con el modelo entrenado..."
chemprop_predict \
  --test_path /home/xavi/Escritorio/midirectorio/chemprop/chemprop_data_val.csv \
  --checkpoint_path /home/xavi/Escritorio/midirectorio/chemprop/random_cv/fold_1/model_0/model.pt \
  --preds_path /home/xavi/Escritorio/midirectorio/chemprop/preds.csv \
  --smiles_columns combined_smiles \

