from pytorch_lightning.cli import LightningCLI

from tamer.datamodule import CROHMEDatamodule
from tamer.lit_tamer import LitTAMER

cli = LightningCLI(
    LitTAMER,
    CROHMEDatamodule,
)
