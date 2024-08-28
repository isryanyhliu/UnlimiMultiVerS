from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.utilities.parsing import lightning_getattr
import math
import torch
from torch.utils.checkpoint import checkpoint
from torch import nn
from torch.nn import functional as F
import transformers
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning.core.decorators import auto_move_data

from transformers import LongformerModel

from allennlp_nn_util import batched_index_select
from allennlp_feedforward import FeedForward
from metrics import SciFactMetrics

from lion_pytorch import Lion

import util

import os
import sys

# 获取当前文件所在的目录（即 multivers 目录）
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录的路径
project_root = os.path.dirname(current_dir)

# 将 libs 目录添加到 sys.path
libs_path = os.path.join(project_root, 'libs')
sys.path.append(libs_path)

from unlimiformers.src.unlimiformer import UnlimiformerLongformer


def masked_binary_cross_entropy_with_logits(input, target, weight, rationale_mask):
    """
    Binary cross entropy loss. Ignore values where the target is -1. Compute
    loss as a "mean of means", first taking the mean over the sentences in each
    row, and then over all the rows.
    """
    mask = torch.where(target > -1, 1, 0)
    float_target = target.clone().to(torch.float)
    float_target[float_target == -1] = 0
    losses = F.binary_cross_entropy_with_logits(input, float_target, reduction="none")
    losses = losses * mask
    n_sents = mask.sum(dim=1)
    totals = losses.sum(dim=1)
    means = totals / n_sents
    final_loss = (means * weight * rationale_mask).sum()

    return final_loss


class MultiVerSModel(pl.LightningModule):
    """
    Multi-task SciFact model that encodes claim / abstract pairs using
    Longformer and then predicts rationales and labels in a multi-task fashion.
    """
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()

        # Initialize metrics for train, validation, and test
        self.train_metrics = SciFactMetrics(compute_on_step=False)
        self.valid_metrics = SciFactMetrics(compute_on_step=False)
        self.test_metrics = SciFactMetrics(compute_on_step=False)

        # Unlimiformer-specific arguments
        self.test_unlimiformer = getattr(hparams, 'test_unlimiformer', True)
        self.random_unlimiformer_training = getattr(hparams, 'random_unlimiformer_training', True)
        self.unlimiformer_training = getattr(hparams, 'unlimiformer_training', True)
        self.unlimiformer_verbose = getattr(hparams, 'unlimiformer_verbose', False)
        self.unlimiformer_layer_begin = getattr(hparams, 'layer_begin', 0)
        self.unlimiformer_layer_end = getattr(hparams, 'layer_end', None)
        self.unlimiformer_chunk_overlap = getattr(hparams, 'unlimiformer_chunk_overlap', 0.5)
        self.unlimiformer_chunk_size = getattr(hparams, 'unlimiformer_chunk_size', 4096)
        self.unlimiformer_head_num = getattr(hparams, 'unlimiformer_head_num', None)
        self.unlimiformer_exclude = getattr(hparams, 'unlimiformer_exclude', False)
        self.use_datastore = getattr(hparams, 'use_datastore', False)
        self.flat_index = getattr(hparams, 'flat_index', False)
        self.test_datastore = getattr(hparams, 'test_datastore', False)
        self.reconstruct_embeddings = getattr(hparams, 'reconstruct_embeddings', False)
        self.gpu_datastore = getattr(hparams, 'gpu_datastore', True)
        self.gpu_index = getattr(hparams, 'gpu_index', True)
        self.label_weight = hparams.label_weight
        self.rationale_weight = hparams.rationale_weight
        self.nei_label = 1


        # 初始化Longformer编码器
        self.encoder = self._get_encoder(hparams)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

        # 定义分类器等其他组件
        hidden_size = self.encoder.config.hidden_size
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.dropout.p, 0]
        self.label_classifier = FeedForward(
            input_dim=hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, hparams.num_labels],
            activations=activations,
            dropout=dropouts)
        self.rationale_classifier = FeedForward(
            input_dim=2 * hidden_size,
            num_layers=2,
            hidden_dims=[hidden_size, 1],
            activations=activations,
            dropout=dropouts)

        # 初始化学习率和其他超参数
        self.lr = hparams.lr
        self.frac_warmup = getattr(hparams, 'frac_warmup', 0.1)
        self.label_threshold = getattr(hparams, 'label_threshold', None)
        self.rationale_threshold = getattr(hparams, 'rationale_threshold', 0.5)

        # 定义 metrics, 使用不同的名字避免冲突
        self.metrics_dict = nn.ModuleDict({
            "train_metrics": SciFactMetrics(compute_on_step=False),
            "valid_metrics": SciFactMetrics(compute_on_step=False),
            "test_metrics": SciFactMetrics(compute_on_step=False)
        })


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--encoder_name", type=str, default="allenai/longformer-base-4096", help="模型名称或路径")
        parser.add_argument("--label_weight", type=float, default=1.0, help="标签预测的损失权重")
        parser.add_argument("--rationale_weight", type=float, default=15.0, help="理由选择的损失权重")
        parser.add_argument("--num_labels", type=int, default=3, help="标签类别的数量")
        parser.add_argument("--gradient_checkpointing", action="store_true", help="是否使用梯度检查点")
        parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
        parser.add_argument("--frac_warmup", type=float, default=0.1, help="训练中用于预热的部分")
        parser.add_argument("--scheduler_total_epochs", default=None, type=int, help="LR调度器的总epoch数")
        parser.add_argument("--label_threshold", default=None, type=float, help="非NEI标签的阈值")
        parser.add_argument("--rationale_threshold", default=0.5, type=float, help="理由选择的阈值")

        # Unlimiformer 相关参数
        parser.add_argument("--test_unlimiformer", type=bool, default=True, help="是否使用 Unlimiformer")
        parser.add_argument("--layer_begin", type=int, default=0, help="Unlimiformer开始应用的层数")
        parser.add_argument("--layer_end", type=int, default=None, help="Unlimiformer结束应用的层数")
        parser.add_argument("--unlimiformer_chunk_overlap", type=float, default=0.5, help="Unlimiformer的chunk重叠率")
        parser.add_argument("--unlimiformer_chunk_size", type=int, default=None, help="Unlimiformer的chunk大小")

        return parser

    def _get_encoder(self, hparams):
        "Start from the Longformer science checkpoint."
        starting_encoder_name = "allenai/longformer-large-4096"
        encoder = LongformerModel.from_pretrained(
            starting_encoder_name,
            gradient_checkpointing=hparams.gradient_checkpointing,
        )
        
        orig_state_dict = encoder.state_dict()
        checkpoint_prefixed = torch.load(util.get_longformer_science_checkpoint())

        # New checkpoint
        new_state_dict = {}
        for k, v in checkpoint_prefixed.items():
            if "lm_head." in k:
                continue
            new_key = k[8:]  # Remove 'roberta.' prefix
            new_state_dict[new_key] = v

        encoder.resize_token_embeddings(new_state_dict['embeddings.word_embeddings.weight'].shape[0])
        encoder.load_state_dict(new_state_dict, strict=False)

        # Apply Unlimiformer if needed
        if self.test_unlimiformer:
            unlimiformer_kwargs = {
                'layer_begin': self.unlimiformer_layer_begin,
                'layer_end': self.unlimiformer_layer_end,
                'chunk_overlap': self.unlimiformer_chunk_overlap,
                'model_encoder_max_len': self.unlimiformer_chunk_size,
            }
            encoder = UnlimiformerLongformer.convert_model(encoder, **unlimiformer_kwargs)
            print("----- Unlimiformer applied to encoder.")
        
        return encoder

    def forward(self, tokenized, abstract_sent_idx):
        encoded = self.encoder(**tokenized)
        pooled_output = self.dropout(encoded.pooler_output)
        label_logits = self.label_classifier(pooled_output)
        label_probs = F.softmax(label_logits, dim=1).detach()
        if self.label_threshold is None:
            predicted_labels = label_logits.argmax(dim=1)
        else:
            label_probs_truncated = label_probs.clone()
            label_probs_truncated[:, self.nei_label] = self.label_threshold
            predicted_labels = label_probs_truncated.argmax(dim=1)
        hidden_states = self.dropout(encoded.last_hidden_state).contiguous()
        sentence_states = batched_index_select(hidden_states, abstract_sent_idx)
        pooled_rep = pooled_output.unsqueeze(1).expand_as(sentence_states)
        rationale_input = torch.cat([pooled_rep, sentence_states], dim=2)
        rationale_logits = self.rationale_classifier(rationale_input).squeeze(2)
        rationale_probs = torch.sigmoid(rationale_logits).detach()
        predicted_rationales = (rationale_probs >= self.rationale_threshold).to(torch.int64)
        return {"label_logits": label_logits,
                "rationale_logits": rationale_logits,
                "label_probs": label_probs,
                "rationale_probs": rationale_probs,
                "predicted_labels": predicted_labels,
                "predicted_rationales": predicted_rationales}

    def training_step(self, batch, batch_idx):
        res = self(batch["tokenized"], batch["abstract_sent_idx"])
        label_loss = F.cross_entropy(res["label_logits"], batch["label"], reduction="none")
        label_loss = (batch["weight"] * label_loss).sum()
        rationale_loss = masked_binary_cross_entropy_with_logits(
            res["rationale_logits"], batch["rationale"], batch["weight"], batch["rationale_mask"])
        loss = self.label_weight * label_loss + self.rationale_weight * rationale_loss
        self.log("label_loss", label_loss)
        self.log("rationale_loss", rationale_loss)
        self.log("loss", loss)
        self._invoke_metrics(res, batch, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "valid")

    def validation_epoch_end(self, outs):
        self._log_metrics("train")
        self._log_metrics("valid")

    def test_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "test")

    def test_epoch_end(self, outs):
        self._log_metrics("train")
        self._log_metrics("test")

    def _invoke_metrics(self, pred, batch, fold):
        assert fold in ["train", "valid", "test"]

        if fold == "train":
            self.train_metrics(pred, batch)
        elif fold == "valid":
            self.valid_metrics(pred, batch)
        elif fold == "test":
            self.test_metrics(pred, batch)

            
    def _log_metrics(self, fold):
        if fold == "train":
            to_log = self.train_metrics.compute()
            self.train_metrics.reset()
        elif fold == "valid":
            to_log = self.valid_metrics.compute()
            self.valid_metrics.reset()
        elif fold == "test":
            to_log = self.test_metrics.compute()
            self.test_metrics.reset()
        
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v)


    def configure_optimizers(self):
        hparams = self.hparams.hparams
        optimizer = Lion(self.parameters(), lr=self.lr)
        if hparams.fast_dev_run or hparams.debug:
            return optimizer
        if isinstance(hparams.gpus, str):
            n_gpus = len([x for x in hparams.gpus.split(",") if x])
        else:
            n_gpus = int(hparams.gpus)
        steps_per_epoch = math.ceil(
            hparams.num_training_instances /
            (n_gpus * hparams.train_batch_size * hparams.accumulate_grad_batches))
        n_epochs = hparams.scheduler_total_epochs if hparams.scheduler_total_epochs is not None else hparams.max_epochs
        num_steps = n_epochs * steps_per_epoch
        warmup_steps = num_steps * self.frac_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)
        lr_dict = {"scheduler": scheduler, "interval": "step"}
        res = {"optimizer": optimizer, "lr_scheduler": lr_dict}
        return res

    @auto_move_data
    def predict(self, batch, force_rationale=False):
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
