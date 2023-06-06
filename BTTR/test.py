from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

test_year = "test"
ckp_path = "lightning_logs/version_9/checkpoints/epoch=1-step=2297-val_ExpRate=0.0000.ckpt"

if __name__ == "__main__":
    trainer = Trainer(logger=False, gpus=1)

    dm = CROHMEDatamodule(test_year=test_year, zipfile_path="../bases/Base_soma_subtracao.zip")

    model = LitBTTR.load_from_checkpoint(ckp_path)

    trainer.test(model, datamodule=dm)
