import numpy as np
import pandas as pd
import logging
import os
import json
import subprocess

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from rdkit.Chem import AllChem as Chem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import Descriptors

from padelpy import padeldescriptor

from scaffold_splits import scaffold_split

def get_morgan_fingerprints(df, mol_or_solv='molecules', nbits=None):
    """Gets Morgan fingerprints for a dataframe with SMILES columns.

    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame that contains columns of 'smiles' and/or 'solvent'
    mol_or_solv : 'molecules' or 'solvents'
        A flag to specify which column to look for and what default
        value of nbits to use
    nbits : int
        Number of bits for Morgan fingerprint (default is 1024 for smiles
        and 256 for solvent)

    Returns
    -------
    df : pandas DataFrame
        An updated DataFrame including the Morgan fingerprints, one bit 
        per column
    col_names_list : list of str
        A list of the names of the Morgan fingerprint columns ('mfpX' 
        or 'sfpX' for X between 0 and nbits)
        
    """
    
    if mol_or_solv=='molecules':
        radius = 4
        if nbits is None:
            nbits = 1024
        col_name = 'mfp'
        smiles_col = 'smiles'
    elif mol_or_solv=='solvents':
        radius = 4
        if nbits is None:
            nbits = 256
        col_name = 'sfp'
        smiles_col = 'solvent'

    unique_df = df.drop_duplicates(subset=[smiles_col])[[smiles_col]]
    smiles_list = list(unique_df[smiles_col])

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros((0,), dtype=np.int8)
        fp = Chem.GetMorganFingerprintAsBitVect(mol,radius,nbits)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    
    col_names_list = []
    for i in range(0,nbits):
        unique_df[col_name+str(i+1)] = np.nan
        col_names_list.append(col_name+str(i+1))
    
    unique_df[col_names_list] = fps

    df = df.merge(unique_df)

    return df, col_names_list

def handle_duplicates(df, cutoff=5, agg_source_col='multiple'):
    """Aggregates duplicate measurements in a DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with required columns: 'smiles', 'solvent', 'peakwavs_max' 
    cutoff : int
        Wavelength cutoff in nm. Duplicate measurements of the same smiles-solvent
        pair with standard deviation less than cutoff are averaged. Those with 
        standard deviation greater than cutoff are dropped.
        
    Returns
    -------
    df : pandas DataFrame
        An updated DataFrame with duplicates aggregated or removed
        
    """
    
    cols = [x for x in df.columns if x not in ['smiles','solvent','peakwavs_max','source']]
    
    agg_dict = {'peakwavs_max':['mean','std']}
    if agg_source_col=='multiple':
        agg_dict['source'] = lambda x: 'multiple' if len(x) > 1 else x, 
    elif agg_source_col=='random':
        np.random.seed(0)
        agg_dict['source'] = np.random.choice
        
    for col in cols:
        agg_dict[col] = 'mean'
    
    # For all smiles+solvent pairs, find mean and std of peakwavs_max
    # If std > cutoff, drop; elif std <= cutoff, take mean
    df = df.groupby(['smiles','solvent']).agg(agg_dict).reset_index()
    high_std_idx = df[df['peakwavs_max']['std']>cutoff].index
    df.drop(index=high_std_idx, inplace=True)
    df.drop(columns='std', level=1, inplace=True)
    df.columns = df.columns.get_level_values(0)
    
    return df

def data_split_and_write(X, feature_names=None, sizes= (0.8, 0.2),target_names=['peakwavs_max'], solvation=False, split_type='scaffold', 
                         scale_targets=False, write_files=False, random_seed=0):
    """Writes train, val, test CSV files for Chemprop with a given dataset.

    Parameters
    ----------
    X : pandas DataFrame
        DataFrame to be split (has columns: 'smiles', target_names, and feature_names 
        (and 'solvent' if solvation=True))
    feature_names : list of str or None
        Names of feature columns in X to be added to feature files (default is None)
    target_names : list of str
        Names of target columns to be printed to files (default is ['peakwavs_max'])
    solvation : bool
        Specify whether to include solvents in target file (default is False)
    split_type : str
        which type of splitting to use ('scaffold', 'group_by_smiles', or 'random')
    scale_targets : bool
        whether to scale targets to have mean 0 and standard deviation 1 (default is False)
    write_files : bool
        whether to write the resulting splits to files
    random_seed : int or None
        number to provide for the seed / random_state arguments to make splits reproducible
        (use None if doing multiple splits for cross validation)
        
    """
    
    if split_type=='scaffold':
        X_train, X_test = scaffold_split(X, sizes , balanced=True, seed=random_seed)
    elif split_type=='group_by_smiles': # Randomly split into train, val, and test sets such that no SMILES is in multiple sets
        gss1 = GroupShuffleSplit(n_splits=2, train_size=0.8, random_state=random_seed)
        train_idx, temp_idx = list(gss1.split(X, groups=X['smiles']))[0]
        X_train, X_temp = X.iloc[train_idx,:], X.iloc[temp_idx,:]
        gss2 = GroupShuffleSplit(n_splits=2, train_size=0.5, random_state=random_seed)
        val_idx, test_idx = list(gss2.split(X_temp, groups=X_temp['smiles']))[0]
        X_val, X_test = X_temp.iloc[val_idx, :], X_temp.iloc[test_idx, :]    
    elif split_type=='random': # Randomly split into train, val, and test sets
        X_train = X.sample(frac=0.8, random_state=random_seed)
        X_temp = X.drop(X_train.index)
        X_val = X_temp.sample(frac=0.5, random_state=random_seed)
        X_test = X_temp.drop(X_val.index)
        
    if scale_targets:
        scaler = StandardScaler()
        scaler.fit(X_train[target_names])
        X_train[target_names] = scaler.transform(X_train[target_names])
        X_val[target_names] = scaler.transform(X_val[target_names])
        X_test[target_names] = scaler.transform(X_test[target_names])
    
    if write_files:
        # Name files
        train_target_file = 'CV/smiles_target_train.csv'
        #val_target_file = 'smiles_target_val.csv'
        test_target_file = 'CV/smiles_target_test.csv'
        train_features_file = 'CV/features_train.csv'
        #val_features_file = 'features_val.csv'
        test_features_file = 'CV/features_test.csv'

        # Write splits to CSVs
        if solvation:
            X_train[['smiles','solvent']+target_names].to_csv(train_target_file, index=False)
            #X_val[['smiles','solvent']+target_names].to_csv(val_target_file, index=False)
            X_test[['smiles','solvent']+target_names].to_csv(test_target_file, index=False)
        else:
            X_train[['smiles']+target_names].to_csv(train_target_file, index=False)
            #X_val[['smiles']+target_names].to_csv(val_target_file, index=False)
            X_test[['smiles']+target_names].to_csv(test_target_file, index=False)
        if feature_names:
            X_train[feature_names].to_csv(train_features_file, index=False)
            #X_val[feature_names].to_csv(val_features_file, index=False)
            X_test[feature_names].to_csv(test_features_file, index=False)
    
    return X_train, X_test

# Make Bash Scripts to Run Chemprop on MIT Supercloud
def write_chemprop_script(work_dir, chemprop_dir='/home/gridsan/kgreenman/chemprop', solvation=False, fp_only=False,
                          ensemble_size=1, metric='rmse', prenormalized_features=False, 
                          hyperparam_file_name='sigopt_chemprop_best_hyperparams.json'):
    """Write a single bash slurm script to execute Chemprop on the MIT Supercloud.

    Parameters
    ----------
    work_dir : str
        Current working directory
    chemprop_dir : str
        Location of chemprop on the HPC cluster
    solvation : bool
        Whether this job should use the solvent MPNN representation and 
        expect 2 SMILES per line input (default False)
    fp_only : bool
        Whether the molecular representation is exclusively a MPNN 
        fingerprint (meaning that Chemprop should not expect a separate 
        file with feature inputs) (default False)
    ensemble_size : int
        Size of the Chemprop model ensemble (default 1)
    metric : str
        Which metric to use as a training objective in Chemprop ('mae' 
        or 'rmse') (default 'rmse')
    prenormalized_feautres : bool
        Whether the features being input to Chemprop have already been 
        normalized (meaning Chemprop should not normalize them again)
        (default False)
    hyperparam_file_name : str or None
        If a str, append "--config_path" and this name to the command. 
        If None, append nothing. (default 'sigopt_chemprop_best_hyperparams.json')
        
    """

    # Components of Run Script
    header = """#!/bin/bash
#SBATCH -p normal
#SBATCH -J uvvis_pred
#SBATCH -o uvvis_pred-%j.out
#SBATCH -t 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=30gb
#SBATCH --gres=gpu:volta:1

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
cat $0
echo ""

source /etc/profile
module load anaconda/2020a
source activate chemprop
"""

#     if split_type=='random':
    train_command = f"""python {chemprop_dir}/train.py \
--data_path smiles_target_train.csv \
--separate_val_path smiles_target_val.csv \
--separate_test_path smiles_target_test.csv \
--dataset_type regression --save_dir $(pwd) \
--metric {metric} --epochs 200 --gpu 0 --ensemble_size {ensemble_size}"""

    predict_command = f"""python {chemprop_dir}/predict.py \
--test_path smiles_target_test.csv \
--checkpoint_dir $(pwd) \
--preds_path preds.csv --gpu 0"""
        
        
    if not fp_only:
        train_command = train_command+""" --features_path features_train.csv \
--separate_val_features_path features_val.csv --separate_test_features_path features_test.csv"""
        predict_command = predict_command+" --features_path features_test.csv"
    
    if hyperparam_file_name is not None:
        train_command = train_command+f""" --config_path {hyperparam_file_name}"""
    
    if solvation:
        train_command = train_command+" --number_of_molecules 2"
        predict_command = predict_command+" --number_of_molecules 2"
        
    if prenormalized_features:
        train_command = train_command+" --no_features_scaling"
        predict_command = predict_command+" --no_features_scaling"
        
    if ensemble_size > 1:
        predict_command = predict_command+" --ensemble_variance"
        
    # Create run script from components
    commands_list = [header, train_command, predict_command]
    run_script = "\n\n".join(commands_list)

    with open('run_chemprop.sh','w') as f:
        f.write(run_script)
        
    return

def make_chemprop_files_and_scripts_bulk(
    dataset_to_df_dict,
    WORKING_DIR,
    RESULTS_DIR, 
    feature_names_dict,
    solvent_reps_list=['minnesota_descriptors', 'morgan_fingerprint', 'solvent_mpnn', 'chemfluor'],
    mol_reps_list=['fp', 'fp_rdkit', 'fp_tddft', 'fp_tddft_rdkit'],
    dataset_list=['all', 'chemfluor', 'dsscdb', 'dyeagg', 'jcole', 'joung'],
    ensemble_sizes=[1],
    split_type='random',
    metric='rmse',
    prenormalized_features=False,
    ):
    """Create all input files for Chemprop benchmark runs in the appropriate directories.

    Parameters
    ----------
    dataset_to_df_dict : dict
        Dictionary mapping str names of each data source to its corresponding 
        DataFrame
    WORKING_DIR : str
        Current working directory
    RESULTS_DIR : str
        Location of the results directory
    feature_names_dict : dict
        Dictionary mapping a str name for each type of feature to their 
        corresponding lists of features
    solvent_reps_list : list of str
        List of solvent representations (can be any or all of 
        'minnesota_descriptors', 'morgan_fingerprint', 'solvent_mpnn', 'chemfluor')
    mol_reps_list : list of str
        List of molecular feature representations (can be any or all of 
        'fp', 'fp_rdkit', 'fp_tddft', 'fp_tddft_rdkit')
    dataset_list : list of str
        List of datasets to train models on (can be any or all of 'all', 
        'chemfluor', 'dsscdb', 'dyeagg', 'jcole', 'joung')    
    ensemble_size : list of int
        List of ensemble sizes to make for each model  
    split_type : str
        Type of split ('random' or 'scaffold') (default 'random')
    metric : str
        Which metric to use as a training objective in Chemprop ('mae' 
        or 'rmse') (default 'rmse')
    prenormalized_feautres : bool
        Whether the features being input to Chemprop have already been 
        normalized (meaning Chemprop should not normalize them again)
        (default False)
        
    """

    CHEMPROP_RESULTS_DIR = os.path.join(RESULTS_DIR, 'chemprop')
#     CHEMPROP_RESULTS_DIR = os.path.join(RESULTS_DIR, 'chemprop_tddft')

    chemprop_molecule_rep_to_features_dict = {'fp': None, 
                                              'fp_rdkit': ['rdkit'], 
                                              'fp_tddft': ['tddft'], 
                                              'fp_tddft_rdkit': ['tddft', 'rdkit']}
    
    solvent_rep_to_features_dict = {'minnesota_descriptors': 'minnesota', 
                                    'morgan_fingerprint': 'sfp',
                                    'solvent_mpnn': None,
                                    'chemfluor': 'cgsd'}
    
    for solvent_representation in solvent_reps_list:
        for molecule_representation in mol_reps_list:
            
            if (solvent_representation=='solvent_mpnn') and (molecule_representation=='fp'):
                fp_only = True
            else:
                fp_only = False
                
            for dataset in dataset_list:
                for ensemble_size in ensemble_sizes:
                    if len(ensemble_sizes)>1:
                        model_dir = os.path.join(
                            WORKING_DIR, 
                            CHEMPROP_RESULTS_DIR, 
                            solvent_representation, 
                            molecule_representation, 
                            dataset,
                            'ensemble_size_'+str(ensemble_size),
                                                )
                    elif len(ensemble_sizes)==1:
                        model_dir = os.path.join(
                            WORKING_DIR, 
                            CHEMPROP_RESULTS_DIR, 
                            solvent_representation, 
                            molecule_representation, 
                            dataset,
                                                )

                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                        
                    os.chdir(model_dir)

                    features = []
                    if solvent_representation != 'solvent_mpnn':
                        features.extend(feature_names_dict[solvent_rep_to_features_dict[solvent_representation]])
                    if molecule_representation != 'fp':
                        for features_list in chemprop_molecule_rep_to_features_dict[molecule_representation]:
                            features.extend(feature_names_dict[features_list])

                    if len(features)==0:
                        features = None

                    if solvent_representation=='solvent_mpnn':
                        solvation = True
                    else:
                        solvation = False

                    data_split_and_write(dataset_to_df_dict[dataset], 
                                         feature_names=features,
                                         target_names=['peakwavs_max'], 
                                         solvation=solvation,
                                         split_type=split_type,
                                         scale_targets=False,
                                         write_files=True)

                    write_chemprop_script(os.getcwd(), 
                                          chemprop_dir='/home/gridsan/kgreenman/chemprop', 
                                          solvation=solvation,
                                          fp_only=fp_only,
                                          ensemble_size=ensemble_size, 
                                          metric=metric,
                                          prenormalized_features=prenormalized_features)
    
                    hyperparams_file = os.path.join(WORKING_DIR, 'sigopt/sigopt_chemprop_best_hyperparams.json')
                    subprocess.call(["cp", hyperparams_file, 'sigopt_chemprop_best_hyperparams.json'])
    
    return

def calculate_loss_metrics(results_dir, model, verbose=False):
    """Calculate the MAE, RMSE, and R2 for a model's predictions.

    Parameters
    ----------
    results_dir : str
        Directory where results are stored
    model : str
        'chemprop' or 'xgboost' 
    verbose : bool
        Whether to print messages while calculating metrics
        (default False)
    
    Returns
    -------
    mae : float
        Mean absolute error
    rmse : float
        Root mean square error
    r2 : float
        R^2 score
        
    """
    
    if verbose: 
        print(results_dir)
    
    if (model == 'chemprop') or (model == 'chemprop_tddft'):
        train_df = pd.read_csv(os.path.join(results_dir, 'smiles_target_train.csv'))
        test_df = pd.read_csv(os.path.join(results_dir, 'smiles_target_test.csv'))
        try:
            preds_df = pd.read_csv(os.path.join(results_dir, 'preds.csv'))
        except:
            print(results_dir)
            return np.nan, np.nan, np.nan
        preds_df.rename(columns={'peakwavs_max':'peakwavs_max_pred'}, inplace=True)
        results_df = pd.concat([test_df, preds_df[['peakwavs_max_pred']]], axis=1)
    elif model == 'chemfluor_gbrt':
        results_df = pd.read_csv(os.path.join(results_dir, 'chemfluor_gbrt_preds.csv'))
        
    mae = mean_absolute_error(results_df['peakwavs_max'], results_df['peakwavs_max_pred'])
    rmse = mean_squared_error(results_df['peakwavs_max'], results_df['peakwavs_max_pred'], squared=False)
    r2 = r2_score(results_df['peakwavs_max'], results_df['peakwavs_max_pred'])
    
    if verbose:
        print('MAE: {0:.2f}, RMSE: {1:.2f}, R2: {2:.2f}'.format(mae, rmse, r2))

    # Set extreme outliers to have predicted value equal to the mean of the training set
    # 4 extreme outlier predictions are present due to erroneous values in the RDKit or TDDFT features
    outlier_idx = results_df.query('peakwavs_max_pred < 0 | peakwavs_max_pred > 2000').index
    if len(outlier_idx)>0:
        bad_smiles = results_df.loc[outlier_idx, 'smiles'].values
        true_value = results_df.loc[outlier_idx, 'peakwavs_max'].values
        pred_value = results_df.loc[outlier_idx, 'peakwavs_max_pred'].values
        mean_of_training_set = train_df['peakwavs_max'].mean() # this will fail if XGBoost has an outlier because we didn't write the training set to a file
        results_df.loc[outlier_idx, 'peakwavs_max_pred'] = mean_of_training_set
        if verbose: 
            print(f'SMILES {bad_smiles} with experimental value(s) {true_value} have bad prediction(s) of {pred_value}')
            print(f'Setting all predictions outside of range 0 - 2000 nm to mean of training set ({mean_of_training_set} nm)')

        mae = mean_absolute_error(results_df['peakwavs_max'], results_df['peakwavs_max_pred'])
        rmse = mean_squared_error(results_df['peakwavs_max'], results_df['peakwavs_max_pred'], squared=False)
        r2 = r2_score(results_df['peakwavs_max'], results_df['peakwavs_max_pred'])
        if verbose:
            print('New MAE: {0:.2f}, New RMSE: {1:.2f}, New R2: {2:.2f}'.format(mae, rmse, r2))
        
    if verbose:    
        print()
        
    return mae, rmse, r2

def get_padel_descriptors(df, descriptor_types='chemfluor'):
    """Calculate PaDEL descriptors for a DataFrame of molecules.

    Parameters
    ----------
    df : pandas DataFrame
        A DataFrame containing the column 'smiles'
    descriptor_types : str
        Determines which XML file to use for specifying features 
        ('chemfluor', 'all_2d', or 'all')
    
    Returns
    -------
    df : pandas DataFrame
        An updated DataFrame including the PaDEL features, one 
        per column
    padel_feature_names : list of str
        A list of the names of the PaDEL features columns
        
    """
        
    if descriptor_types=='chemfluor':
        descriptortypes_file = 'preprint_fp_features.xml'
    elif descriptor_types=='all_2d':
        descriptortypes_file = 'padel_2d_descriptors_no_fp_features.xml'
    elif descriptor_types=='all':
        descriptortypes_file = 'all_descriptors_and_fp_features.xml'

    unique_df = df.drop_duplicates(subset=['smiles'])[['smiles']]
    smiles_list = list(unique_df['smiles'])

    input_file_name = 'smiles_padel_input.smi'
    output_file_name = 'smiles_padel_output.csv'
    
    os.chdir('padel')

    with open(input_file_name,'w') as f:
        for x in smiles_list:
            f.write(x+'\n')

    padeldescriptor(maxruntime = -1, 
                    waitingjobs = -1,
                    threads = -1, 
                    d_2d = False, 
                    d_3d = False, 
                    convert3d = False,
                    descriptortypes = descriptortypes_file,
                    detectaromaticity = False, # not deterministic if True
                    mol_dir = input_file_name, 
                    d_file = output_file_name, 
                    fingerprints = True,
                    log = True, 
                    maxcpdperfile = 0,
                    removesalt = False, 
                    retain3d = False,
                    retainorder = False, 
                    standardizenitro = False,
                    standardizetautomers = False,
                    tautomerlist = None,
                    usefilenameasmolname = True,
                    sp_timeout = None)

    padel_df = pd.read_csv(output_file_name)

    smiles_numbered_dict = dict(enumerate(smiles_list))

    if len(smiles_list)==1: # no number on end of name if only one SMILES given
        padel_df['Name'] = smiles_list
    elif len(smiles_list)>1: # need to do it this way because list out of order
        smiles_numbered_dict = dict(enumerate(smiles_list))
        padel_df['Name'] = padel_df['Name'].map(lambda x: smiles_numbered_dict[int(x.split('_')[-1]) - 1])
        
    padel_df.rename(columns={'Name':'smiles'}, inplace=True)

    padel_feature_names = list(padel_df.columns[1:]) # all columns except smiles

    unique_df = unique_df.merge(padel_df)

    df = df.merge(unique_df)
    
    os.chdir('..')

    return df, padel_feature_names
    
