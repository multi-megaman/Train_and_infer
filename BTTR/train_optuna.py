from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities import seed
from bttr.lit_bttr import LitBTTR
from bttr.datamodule import CROHMEDatamodule
from optuna.integration import PyTorchLightningPruningCallback

GPUS = 1

def train_test_BTTR_model(params=None, trial=None):
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_ExpRate', save_top_k=1, mode='max', filename='{epoch}-{step}-{val_ExpRate:.4f}')
    early_stopping = EarlyStopping('val_ExpRate', patience=6, mode='max')
    callbacks = [lr_monitor, checkpoint_callback, early_stopping]

    #if trial is not None:
    #    callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val_ExpRate"))

    if params is None:
        params = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': callbacks,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':400},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 8, 'nhead': 8, 'num_decoder_layers': 1,
                               'dim_feedforward': 256, 'dropout': 0.3, 'beam_size': 10, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    #seed.seed_everything(params['seed_everything'], workers=True)

    model = LitBTTR(
        d_model=params['model']['d_model'],
        growth_rate=params['model']['growth_rate'],
        num_layers=params['model']['num_layers'],
        nhead=params['model']['nhead'],
        num_decoder_layers=params['model']['num_decoder_layers'],
        dim_feedforward=params['model']['dim_feedforward'],
        dropout=params['model']['dropout'],
        beam_size=params['model']['beam_size'],
        max_len=params['model']['max_len'],
        alpha=params['model']['alpha'],
        learning_rate=params['model']['learning_rate'],
        patience=params['model']['patience']
        )
    
    dm = CROHMEDatamodule(
        zipfile_path=params['data']['zipfile_path'],
        test_year=params['data']['test_year'],
        batch_size=params['data']['batch_size'],
        num_workers=params['data']['num_workers'],
        data_augmentation=params['data']['data_augmentation']
    )

    trainer = Trainer(
        #seed_everything=params['seed_everything'],
        #deterministic=True,
        checkpoint_callback=params['trainer']['checkpoint_callback'],
        callbacks=callbacks,
        gpus=GPUS,
        check_val_every_n_epoch=params['trainer']['check_val_every_n_epoch'],
        max_epochs=params['trainer']['max_epochs']
    )
    trainer.fit(model, datamodule=dm)

    #return trainer.callback_metrics["val_ExpRate"].item(), trainer.callback_metrics["train_loss"].item()
    return checkpoint_callback.best_model_score.item(), trainer.callback_metrics["train_loss"].item()
    #return trainer.callback_metrics


if __name__ == '__main__':
    print(train_test_BTTR_model())
