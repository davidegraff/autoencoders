from __future__ import annotations

from typing import Optional, Sequence, Union
from typing_extensions import Self
import warnings

import numpy as np
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.rdBase import BlockLogs
import torch
from torch import Tensor, optim, nn
from torch.nn.utils import rnn

from ae_utils.utils import Configurable, LoggingMixin, SaveAndLoadMixin
from ae_utils.modules import RnnEncoder, RnnDecoder
from ae_utils.schedulers import CyclicalScheduler, Scheduler
from ae_utils.char.tokenizer import Tokenizer
from ae_utils.supervisors import Supervisor, DummySupervisor, SupervisorRegistry

block = BlockLogs()
warnings.filterwarnings("ignore", "Trying to infer the `batch_size`", UserWarning)
warnings.filterwarnings("ignore", "dropout option adds dropout after all but last", UserWarning)


class LitCVAE(pl.LightningModule, Configurable, LoggingMixin, SaveAndLoadMixin):
    """A variational autoencoder for learning latent representations of strings using
    character-based RNN encoder/decoder pair

    NOTE: The encoder and decoder both utilize an embedding layer, and in a typical VAE, this layer
    is shared between the two modules. This is not strictly required for a VAE to work and is thus
    not enforced

    Parameters
    ----------
    tokenizer : Tokenizer
        the :class:`~autoencoders.char.tokenizer.Tokenizer` to use for measuring generation quality
        during validation
    encoder : CharEncoder
        the encoder module to project from tokenized sequences into the latent space
    decoder : CharDecoder
        the decoder module to generate tokenized sequences from latent representations
    supervisor : Optional[Supervisor] = None
    lr : float, default=3e-4
        the learning rate
    v_reg : Union[float, Scheduler, None], default=None
        the regularization loss weight scheduler. One of:

        * `Scheduler`: the scheduler to use
        * `float`: use a constant weight schedule (i.e., no scheudle)
        * `None`: use a `~autoencoders.char.schedulers.Scheduler` from 0->0.1 over 20 epochs

    v_sup : Union[float, Scheduler], default=0
        the supervision loss weight scheduler. One of:

        * `Scheduler`: the scheduler to use
        * `float`: use a constant weight schedule (i.e., no scheudle)

    Raises
    ------
    ValueError

        * if the supplied CharEncoder and CharDecoder do not have the same latent dimension
        * if the Supervisor does not have the correct input dimension (= the VAE's latent dimension)
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        encoder: RnnEncoder,
        decoder: RnnDecoder,
        supervisor: Optional[Supervisor] = None,
        lr: float = 3e-4,
        v_reg: Union[float, Scheduler, None] = None,
        v_sup: Union[float, Scheduler] = 0,
    ):
        super().__init__()

        if encoder.d_z != decoder.d_z:
            raise ValueError(
                "'encoder' and 'decoder' have mismatched latent dimension sizes! "
                f"got: {encoder.d_z} and {decoder.d_z}, respectively."
            )
        if len(tokenizer) != encoder.d_v:
            raise warnings.warn(
                "tokenizer and encoder have mismatched vocabulary sizes! "
                f"got: {len(tokenizer)} and {encoder.d_v}, respectively."
            )
        if len(tokenizer) != decoder.d_v:
            raise warnings.warn(
                "tokenizer and decoder have mismatched vocabulary sizes! "
                f"got: {len(tokenizer)} and {decoder.d_v}, respectively."
            )

        self.tokenizer = tokenizer
        self.encoder = encoder
        self.decoder = decoder
        self.supervisor = supervisor or DummySupervisor()
        self.lr = lr

        self.v_reg = v_reg
        self.v_sup = v_sup

        self.rec_metric = nn.CrossEntropyLoss(reduction="sum", ignore_index=self.encoder.PAD)

        self.supervisor.check_input_dim(self.d_z)

    @property
    def d_z(self) -> int:
        return self.encoder.d_z

    @property
    def v_reg(self) -> Scheduler:
        return self.__v_reg

    @v_reg.setter
    def v_reg(self, v_reg: Union[float, Scheduler, None]):
        if isinstance(v_reg, (int, float)):
            self.__v_reg = Scheduler([v_reg], "reg")
        elif v_reg is None:
            self.__v_reg = Scheduler(np.linspace(0, 0.1, 21), "reg")
        else:
            self.__v_reg = v_reg

    @property
    def v_sup(self) -> Scheduler:
        return self.__v_sup

    @v_sup.setter
    def v_sup(self, v: Union[float, Scheduler]):
        self.__v_sup = Scheduler([v], "sup") if isinstance(v, (int, float)) else v

    def encode(self, xs: Sequence[Tensor]) -> Tensor:
        return self.encoder(xs)
    
    def decode(self, Z: Tensor, max_len: int = 80) -> list[Tensor]:
        return self.decoder(Z, max_len)

    def reconstruct(self, xs: Sequence[Tensor]) -> list[Tensor]:
        return self.decode(self.encode(xs))

    forward = encode
    generate = decode

    def on_train_start(self):
        self.v_reg.reset()
        self.v_sup.reset()

    def training_step(self, batch: Sequence[Tensor], batch_idx) -> Tensor:
        xs, Y = batch

        Z, l_reg = self.encoder.train_step(xs)
        X_logits = self.decoder.train_step(xs, Z)

        X_logits_packed = X_logits[:, :-1].contiguous().view(-1, X_logits.shape[-1])
        X_packed = rnn.pad_sequence(xs, True, self.decoder.PAD)[:, 1:].contiguous().view(-1)

        l_rec = self.rec_metric(X_logits_packed, X_packed) / len(xs)
        l_sup = self.supervisor(Z, Y)

        metrics = dict(rec=l_rec, reg=l_reg, sup=l_sup)
        self._log_split("train", metrics)
        self.log("loss", l_rec + l_reg + l_sup)

        return l_rec + self.v_reg.v * l_reg + self.v_sup.v * l_sup

    def validation_step(self, batch: Sequence[Tensor], batch_idx):
        xs, Y = batch

        Z, l_reg = self.encoder.train_step(xs)
        X_logits = self.decoder.train_step(xs, Z)

        X_logits_packed = X_logits[:, :-1].contiguous().view(-1, X_logits.shape[-1])
        X_packed = rnn.pad_sequence(xs, True, self.decoder.PAD)[:, 1:].contiguous().view(-1)

        l_rec = self.rec_metric(X_logits_packed, X_packed) / len(xs)
        l_sup = self.supervisor(Z, Y)
        acc = sum(map(torch.equal, xs, self.reconstruct(xs))) / len(xs)

        return l_rec, l_reg, l_sup, acc, len(xs)

    def predict_step(self, batch, batch_idx: int, dataloader_idx=0) -> Tensor:
        return self.encode(batch)

    def on_train_epoch_start(self):
        self.log(f"v/{self.v_reg.name}", self.v_reg.v)
        self.log(f"v/{self.v_sup.name}", self.v_sup.v)
        
    def training_epoch_end(self, *args):
        self.v_reg.step()
        self.v_sup.step()

    def validation_epoch_end(self, outputs):
        *losses, sizes = torch.tensor(outputs).split(1, 1)
        l_rec, l_reg, l_sup, acc = ((l * sizes).sum() / sizes.sum() for l in losses)
        metrics = dict(rec=l_rec, reg=l_reg, sup=l_sup, loss=l_rec + l_reg + l_sup, acc=acc)

        n = 1000
        f_valid, f_unique = self.check_gen_quality(torch.randn(n, self.d_z, device=self.device))
        metrics.update({f"valid@{n//1000}k": f_valid, f"unique@{n//1000}k": f_unique})

        self._log_split("val", metrics)

    def configure_optimizers(self):
        params = set(self.encoder.parameters())
        params.update(self.decoder.parameters())
        params.update(self.supervisor.parameters())

        return optim.Adam(params, self.lr)

    def check_gen_quality(self, Z: Tensor):
        smis = [self.tokenizer.decode(x.tolist()) for x in self.decode(Z)]
        smis = [smi for smi in smis if Chem.MolFromSmiles(smi) is not None]

        f_valid = len(smis) / len(Z)
        f_unique = 0 if len(smis) == 0 else len(set(smis)) / len(smis)

        return f_valid, f_unique

    def to_config(self) -> dict:
        return {
            "tokenizer": self.tokenizer.to_config(),
            "encoder": self.encoder.to_config(),
            "decoder": self.decoder.to_config(),
            "supervisor": {"alias": self.supervisor.alias, "config": self.supervisor.to_config()},
            "lr": self.lr,
            "v_reg": {
                "cyclic": isinstance(self.v_reg, CyclicalScheduler),
                "config": self.v_reg.to_config(),
            },
            "v_sup": {
                "cyclic": isinstance(self.v_sup, CyclicalScheduler),
                "config": self.v_sup.to_config(),
            },
            "shared_enb": self.encoder.emb is self.decoder.emb,
        }

    @classmethod
    def from_config(cls, config: dict) -> Self:
        enc_emb_config = config["encoder"]["embedding"]
        dec_emb_config = config["decoder"]["embedding"]

        tok = Tokenizer.from_config(config["tokenizer"])
        enc = RnnEncoder.from_config(config["encoder"])
        dec = RnnDecoder.from_config(config["decoder"])

        args = [tok, enc, dec]
        kwargs = {}

        if "supervisor" in config:
            sup_alias = config["supervisor"]["alias"]
            sup_config = config["supervisor"]["config"]
            sup = SupervisorRegistry[sup_alias].from_config(sup_config)
            kwargs["supervisor"] = sup

        if "lr" in config:
            kwargs["lr"] = config["lr"]

        if "v_reg" in config:
            sched_cls = CyclicalScheduler if config["v_reg"]["cyclic"] else Scheduler
            config = config["v_reg"]["config"]
            v_reg = sched_cls.from_config(config)
            kwargs["v_reg"] = v_reg

        if "v_sup" in config:
            sched_cls = CyclicalScheduler if config["v_sup"]["cyclic"] else Scheduler
            config = config["v_sup"]["config"]
            v_sup = sched_cls.from_config(config)
            kwargs["v_sup"] = v_sup

        if config.get("shared_emb", True) and enc_emb_config == dec_emb_config:
            enc.emb = dec.emb

        return cls(*args, **kwargs)
