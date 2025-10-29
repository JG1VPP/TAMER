import argparse
from pathlib import Path
from pickle import dumps

from comer.datamodule import CROHMEDatamodule
from comer.lit_comer import LitCoMER
from pytorch_lightning import Trainer, seed_everything

seed_everything(7)


def options():
    args = argparse.ArgumentParser()

    args.add_argument("--ckpt", type=str, required=True)
    args.add_argument("--data", type=str, required=True)
    args.add_argument("--split", type=str, required=True)
    args.add_argument("--store", type=str, required=True)

    return vars(args.parse_args())


def main(ckpt: str, data: str, split: str, store: str):
    result = dict(ckpt=ckpt, data=data, split=split)

    trainer = Trainer(logger=False, devices=1)

    dm = CROHMEDatamodule(
        path=data,
        test_split=split,
        w=224,
        h=224,
        fill=255,
        line=1,
        train_batch_size=4,
        eval_batch_size=4,
        num_workers=1,
    )

    model = LitCoMER.load_from_checkpoint(ckpt)
    data = trainer.predict(model, datamodule=dm)

    scores = dict(CER=model.cer, EM=model.exp_rate)
    result.update(scores=scores, data=sum(data, []))

    path = Path(store).expanduser()
    path.write_bytes(dumps(result))

    print(scores)


if __name__ == "__main__":
    main(**options())
