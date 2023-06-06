import logging
import sys
import optuna

import numpy as np
import time

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from train_optuna import train_test_BTTR_model

GPUS = 1

work_dir = ''
#log_file_name = 'optimize_HME_100k_reduced_CAN'
log_file_name = 'optimize_Base_Soma_Subtr_BTTR'


def evaluation_trial(trial):

    #lr_monitor = LearningRateMonitor(logging_interval='epoch')
    #checkpoint_callback = ModelCheckpoint(monitor='val_ExpRate', save_top_k=1, mode='max', filename='{epoch}-{step}-{val_ExpRate:.4f}')
    #early_stopping = EarlyStopping('val_ExpRate', patience=5, mode='max')
    #callbacks = [lr_monitor, checkpoint_callback, early_stopping]

    params = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, #'callbacks': callbacks,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': trial.suggest_categorical('d_model', [64, 128, 256]),  # default: 256
                               'growth_rate': trial.suggest_categorical('growthRate', [8, 16, 24]), # default: 24,
                               'num_layers': trial.suggest_categorical('num_layers', [4, 8, 16]), # default: 16,
                               'nhead': trial.suggest_categorical('nhead', [2, 4, 8]), # default: 8,
                               'num_decoder_layers': trial.suggest_categorical('num_decoder_layers', [1, 2, 3]), # default: 3,
                               'dim_feedforward': trial.suggest_categorical('dim_feedforward', [256, 512, 1024]), # default: 1024,
                               'dropout': trial.suggest_categorical('dropout', [0.001, 0.3]), # default: 0.3,
                               'beam_size': trial.suggest_categorical('beam_size', [5, 10]), # default: 10,
                               'max_len': 200, 'alpha': 1.0, 'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5,
                              'data_augmentation':trial.suggest_categorical('data_augmentation', [100, 10])} # default: 0
                        )
    
    print(params)

    trial.set_user_attr("params", params)

    test_exp_rate, train_loss = train_test_BTTR_model(params=params, trial=trial)

    trial.set_user_attr("train_loss", train_loss)

    return test_exp_rate

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = log_file_name
storage_name = "sqlite:///{}.db".format(work_dir + study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction='maximize',
                            sampler=optuna.samplers.TPESampler())

print('Trials:', len(study.trials))

study.optimize(evaluation_trial, n_trials=100)

print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)