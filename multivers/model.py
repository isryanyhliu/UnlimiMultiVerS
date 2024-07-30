from argparse import ArgumentParser
import pytorch_lightning as pl
import math
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import LongformerModel
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning.core.decorators import auto_move_data

from allennlp_nn_util import batched_index_select
from allennlp_feedforward import FeedForward
from metrics import SciFactMetrics
import util
import os
import sys
import time

# 将 multivers 和 unlimiformers/src 路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'multivers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs', 'unlimiformers', 'src')))

from libs.unlimiformers.src.unlimiformer import Unlimiformer
from libs.unlimiformers.src.random_training_unlimiformer import RandomTrainingUnlimiformer

def masked_binary_cross_entropy_with_logits(input, target, weight, rationale_mask):
    mask = torch.where(target > -1, 1, 0).to(input.device)
    float_target = target.clone().to(torch.float).to(input.device)
    float_target[float_target == -1] = 0
    losses = F.binary_cross_entropy_with_logits(input, float_target, reduction="none")
    losses = losses * mask
    n_sents = mask.sum(dim=1)
    totals = losses.sum(dim=1)
    means = totals / n_sents
    final_loss = (means * weight * rationale_mask).sum()
    return final_loss

def move_to_device(batch, device):
    if isinstance(batch, dict):
        moved_batch = {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        moved_batch = [move_to_device(v, device) for v in batch]
    elif isinstance(batch, torch.Tensor):
        moved_batch = batch.to(device)
    else:
        moved_batch = batch
    return moved_batch





class MultiVerSModel(pl.LightningModule):
    def __init__(self, hparams, unlimiformer_args=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.nei_label = 1
        self.label_threshold = getattr(hparams, "label_threshold", None)
        self.rationale_threshold = getattr(hparams, "rationale_threshold", 0.5)
        self.label_weight = hparams.label_weight
        self.rationale_weight = hparams.rationale_weight
        self.frac_warmup = hparams.frac_warmup
        self.encoder_name = hparams.encoder_name
        self.encoder = self._get_encoder(hparams, unlimiformer_args)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        hidden_size = self.encoder.config.hidden_size
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        self.label_classifier = FeedForward(
            input_dim=hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=activations,
            dropout=dropouts
        )
        self.rationale_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=activations,
            dropout=dropouts
        )
        self.lr = hparams.lr
        fold_names = ["train", "valid", "test"]
        metrics = {}
        for name in fold_names:
            metrics[f"metrics_{name}"] = SciFactMetrics(compute_on_step=False)
        self.metrics = nn.ModuleDict(metrics)

    @staticmethod
    def _get_encoder(hparams, unlimiformer_args):
        starting_encoder_name = "allenai/longformer-large-4096"
        encoder = LongformerModel.from_pretrained(
            starting_encoder_name,
            gradient_checkpointing=hparams.gradient_checkpointing
        )
        orig_state_dict = encoder.state_dict()
        checkpoint_prefixed = torch.load(util.get_longformer_science_checkpoint())
        new_state_dict = {}
        for k, v in checkpoint_prefixed.items():
            if "lm_head." in k:
                continue
            new_key = k[8:]
            new_state_dict[new_key] = v
        ADD_TO_CHECKPOINT = ["embeddings.position_ids"]
        for name in ADD_TO_CHECKPOINT:
            if name in orig_state_dict:
                new_state_dict[name] = orig_state_dict[name]
            else:
                print(f"Warning: {name} not found in the original state_dict, skipping.")
        target_embed_size = new_state_dict['embeddings.word_embeddings.weight'].shape[0]
        encoder.resize_token_embeddings(target_embed_size)
        filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in orig_state_dict and orig_state_dict[k].shape == v.shape}
        encoder.load_state_dict(filtered_state_dict, strict=False)
        unlimiformer_kwargs = {
            'layer_begin': unlimiformer_args.unlimiformer_layer_begin,
            'layer_end': unlimiformer_args.unlimiformer_layer_end,
            'unlimiformer_head_num': unlimiformer_args.unlimiformer_head_num,
            'exclude_attention': unlimiformer_args.unlimiformer_exclude_attention,
            'model_encoder_max_len': unlimiformer_args.unlimiformer_max_len,
            'chunk_overlap': unlimiformer_args.unlimiformer_chunk_overlap,
            'verbose': unlimiformer_args.unlimiformer_verbose,
            'tokenizer': unlimiformer_args.tokenizer
        }
        encoder = Unlimiformer.convert_model(encoder, **unlimiformer_kwargs)
        return encoder

    def forward(self, tokenized, abstract_sent_idx):
        device = next(self.parameters()).device
        tokenized = move_to_device(tokenized, device)
        abstract_sent_idx = abstract_sent_idx.to(device)

        encoded = self.encoder(**tokenized)

        # 确保 encoded 的输出在正确的设备上
        encoded.last_hidden_state = encoded.last_hidden_state.to(device)
        encoded.pooler_output = encoded.pooler_output.to(device)

        pooled_output = self.dropout(encoded.pooler_output)
        label_logits = self.label_classifier(pooled_output)

        label_probs = F.softmax(label_logits, dim=1).detach()
        if self.label_threshold is None:
            predicted_labels = label_logits.argmax(dim=1)
        else:
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)

        hidden_states = self.dropout(encoded.last_hidden_state).contiguous().to(device)
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx)

        pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states).to(device)
        rationale_input = torch.cat([pooled_rep, sentence_states], dim=2)
        rationale_logits = self.rationale_classifier(rationale_input).squeeze(2)

        rationale_probs = torch.sigmoid(rationale_logits).detach()
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)

        return {
            "label_logits": label_logits,
            "rationale_logits": rationale_logits,
            "label_probs": label_probs,
            "rationale_probs": rationale_probs,
            "predicted_labels": predicted_labels,
            "predicted_rationales": predicted_rationales
        }



    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--encoder_name", type=str, default="allenai/longformer-base-4096")
        parser.add_argument("--label_weight", type=float, default=1.0)
        parser.add_argument("--rationale_weight", type=float, default=15.0)
        parser.add_argument("--num_labels", type=int, default=3)
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--frac_warmup", type=float, default=0.1, help="The fraction of training to use for warmup.")
        parser.add_argument("--scheduler_total_epochs", default=None, type=int, help="If given, pass as total # epochs to LR scheduler.")
        parser.add_argument("--label_threshold", default=None, type=float, help="Threshold for non-NEI label.")
        parser.add_argument("--rationale_threshold", default=0.5, type=float, help="Threshold for rationale.")
        return parser







    def training_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        res = self(batch["tokenized"], batch["abstract_sent_idx"])
        label_loss = F.cross_entropy(
            res["label_logits"], batch["label"].to(self.device), reduction="none")
        label_loss = (batch["weight"].to(self.device) * label_loss).sum()
        rationale_loss = masked_binary_cross_entropy_with_logits(
            res["rationale_logits"], batch["rationale"].to(self.device), batch["weight"].to(self.device),
            batch["rationale_mask"].to(self.device))
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        self.log("label_loss", label_loss)
        self.log("rationale_loss", rationale_loss)
        self.log("loss", loss)
        self._invoke_metrics(res, batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "valid")

    def validation_epoch_end(self, outs):
        self._log_metrics("train")
        self._log_metrics("valid")

    def test_step(self, batch, batch_idx):
        batch = move_to_device(batch, self.device)
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "test")

    def test_epoch_end(self, outs):
        self._log_metrics("train")
        self._log_metrics("test")

    def _invoke_metrics(self, pred, batch, fold):
        assert fold in ["train", "valid", "test"]
        detached = {k: v.detach() for k, v in pred.items()}
        self.metrics[f"metrics_{fold}"](detached, batch)

    def _log_metrics(self, fold):
        the_metric = self.metrics[f"metrics_{fold}"]
        to_log = the_metric.compute()
        the_metric.reset()
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v)

    def configure_optimizers(self):
        hparams = self.hparams.hparams
        optimizer = transformers.AdamW(self.parameters(), lr=self.lr)
        if hparams.fast_dev_run or hparams.debug:
            return optimizer
        if isinstance(hparams.gpus, str):
            n_gpus = len([x for x in hparams.gpus.split(",") if x])
        else:
            n_gpus = int(hparams.gpus)
        steps_per_epoch = math.ceil(
            hparams.num_training_instances /
            (n_gpus * hparams.train_batch_size * hparams.accumulate_grad_batches))
        if hparams.scheduler_total_epochs is not None:
            n_epochs = hparams.scheduler_total_epochs
        else:
            n_epochs = hparams.max_epochs
        num_steps = n_epochs * steps_per_epoch
        warmup_steps = num_steps * self.frac_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)
        lr_dict = {"scheduler": scheduler, "interval": "step"}
        res = {"optimizer": optimizer, "lr_scheduler": lr_dict}
        return res

    @auto_move_data
    def predict(self, batch, force_rationale=False):
        batch = move_to_device(batch, self.device)
        with torch.no_grad():   
            output = self(batch["tokenized"], batch["abstract_sent_idx"])
        return self.decode(output, batch, force_rationale)




    @staticmethod
    def decode(output, batch, force_rationale=False):
        label_lookup = {0: "CONTRADICT", 1: "NEI", 2: "SUPPORT"}
        instances = util.unbatch(batch, ignore=["tokenized"])
        output_unbatched = util.unbatch(output)
        predictions = []
        for this_instance, this_output in zip(instances, output_unbatched):
            predicted_label = label_lookup[this_output["predicted_labels"]]
            rationale_ix = this_instance["abstract_sent_idx"] > 0
            rationale_indicators = this_output["predicted_rationales"][rationale_ix]
            predicted_rationale = rationale_indicators.nonzero()[0].tolist()
            predicted_rationale = [int(x) for x in predicted_rationale]
            if predicted_label != "NEI" and not predicted_rationale and force_rationale:
                candidates = this_output["rationale_probs"][rationale_ix]
                predicted_rationale = [candidates.argmax()]
            res = {
                "claim_id": int(this_instance["claim_id"]),
                "abstract_id": int(this_instance["abstract_id"]),
                "predicted_label": predicted_label,
                "predicted_rationale": predicted_rationale,
                "label_probs": this_output["label_probs"],
                "rationale_probs": this_output["rationale_probs"][rationale_ix]
            }
            predictions.append(res)
        return predictions
