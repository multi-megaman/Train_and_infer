from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

#parser = ArgumentParser()
#parser.add_argument("--config")
#args = parser.parse_args()
#print(args.config)
cli = LightningCLI(LitBTTR, CROHMEDatamodule)
