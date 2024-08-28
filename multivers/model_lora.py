from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import LongformerModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from allennlp_nn_util import batched_index_select
from allennlp_feedforward import FeedForward
from metrics import SciFactMetrics
import util


def masked_binary_cross_entropy_with_logits(input, target, weight, rationale_mask):
    """
    Binary cross entropy loss. Ignore values where the target is -1. Compute
    loss as a "mean of means", first taking the mean over the sentences in each
    row, and then over all the rows.
    """
    # Mask to indicate which values contribute to loss.
    mask = torch.where(target > -1, 1, 0)

    # Need to convert target to float, and set -1 values to 0 in order for the
    # computation to make sense. We'll ignore the -1 values later.
    float_target = target.clone().to(torch.float)
    float_target[float_target == -1] = 0
    losses = F.binary_cross_entropy_with_logits(
        input, float_target, reduction="none")
    # Mask out the values that don't matter.
    losses = losses * mask

    # Take "sum of means" over the sentence-level losses for each instance.
    # Take means so that long documents don't dominate.
    # Multiply by `rationale_mask` to ignore sentences where we don't have
    # rationale annotations.
    n_sents = mask.sum(dim=1)
    totals = losses.sum(dim=1)
    means = totals / n_sents
    final_loss = (means * weight * rationale_mask).sum()

    return final_loss

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, lora_alpha=32):
        super(LoRALinear, self).__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.lora_a = nn.Parameter(torch.zeros(r, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, r))
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_a)
        nn.init.zeros_(self.lora_b)

    def forward(self, x):
        return F.linear(x, self.weight + self.scaling * (self.lora_b @ self.lora_a))


class MultiVerSModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        
        # 如果 train_batch_size 不在 hparams 中，手动添加它
        if not hasattr(self.hparams, 'train_batch_size'):
            self.hparams.train_batch_size = 8  # 设置默认值或根据实际情况调整

        if not hasattr(self.hparams, 'accumulate_grad_batches'):
            self.hparams.accumulate_grad_batches = 1  # 设置默认值或根据实际情况调整

        if not hasattr(self.hparams, 'scheduler_total_epochs'):
            self.hparams.scheduler_total_epochs = None
            
        if not hasattr(self.hparams, 'max_epochs'):
            self.hparams.max_epochs = 20

        # 如果 hparams 中没有定义 label_threshold，则设置默认值
        if hasattr(hparams, 'label_threshold'):
            self.label_threshold = hparams.label_threshold
        else:
            self.label_threshold = 0.5  # 默认值，依据你的需求设置

        # 初始化 rationale_threshold
        if hasattr(hparams, 'rationale_threshold'):
            self.rationale_threshold = hparams.rationale_threshold
        else:
            self.rationale_threshold = 0.5  # 默认值，依据你的需求设置

        # 其他初始化代码...
        self.num_training_instances = hparams.num_training_instances
        self.encoder = self._get_encoder()
        self.label_weight = hparams.label_weight
        self.rationale_weight = hparams.rationale_weight
        self.frac_warmup = hparams.frac_warmup
        self.lr = hparams.lr
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)  # 假设你需要使用 encoder 的 dropout 概率

        hidden_size = self.encoder.config.hidden_size
        activations = [nn.GELU(), nn.Identity()]
        dropouts = [self.encoder.config.hidden_dropout_prob, 0]

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

        # Metrics
        fold_names = ["train", "valid", "test"]
        metrics = {f"metrics_{name}": SciFactMetrics(compute_on_step=False) for name in fold_names}
        self.metrics = nn.ModuleDict(metrics)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        encoder: The transformer encoder that gets the embeddings.
        label_weight: The weight to assign to label prediction in the loss function.
        rationale_weight: The weight to assign to rationale selection in the loss function.
        num_labels: The number of label categories.
        gradient_checkpointing: Whether to use gradient checkpointing with Longformer.
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--encoder_name", type=str, default="allenai/longformer-base-4096")
        parser.add_argument("--label_weight", type=float, default=1.0)
        parser.add_argument("--rationale_weight", type=float, default=15.0)
        parser.add_argument("--num_labels", type=int, default=3)
        parser.add_argument("--gradient_checkpointing", action="store_true")
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--frac_warmup", type=float, default=0.1,
                            help="The fraction of training to use for warmup.")
        parser.add_argument("--scheduler_total_epochs", default=None, type=int,
                            help="If given, pass as total # epochs to LR scheduler.")
        parser.add_argument("--label_threshold", default=None, type=float,
                            help="Threshold for non-NEI label.")
        parser.add_argument("--rationale_threshold", default=0.5, type=float,
                            help="Threshold for rationale.")

        return parser

    def _get_encoder(self):
        "Load Longformer model and apply LoRA to linear layers."
        starting_encoder_name = "allenai/longformer-large-4096"
        gradient_checkpointing = getattr(self.hparams, "gradient_checkpointing", False)
        encoder = LongformerModel.from_pretrained(
            starting_encoder_name,
            gradient_checkpointing=gradient_checkpointing, 
        )

        # Load the science checkpoint
        orig_state_dict = encoder.state_dict()
        checkpoint_prefixed = torch.load(util.get_longformer_science_checkpoint())

        # New checkpoint
        new_state_dict = {}
        for k, v in checkpoint_prefixed.items():
            if "lm_head." in k:
                continue
            new_key = k[8:]
            new_state_dict[new_key] = v

        # Add items from the Huggingface state dict to align with the new checkpoint
        ADD_TO_CHECKPOINT = ["embeddings.position_ids"]
        for name in ADD_TO_CHECKPOINT:
            if name in orig_state_dict:
                new_state_dict[name] = orig_state_dict[name]
            else:
                print(f"Key {name} not found in original state dict, skipping.")

        target_embed_size = new_state_dict['embeddings.word_embeddings.weight'].shape[0]
        encoder.resize_token_embeddings(target_embed_size)
        encoder.load_state_dict(new_state_dict, strict=False)

        # Apply LoRA and freeze original weights
        def replace_with_lora_and_freeze(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    lora_layer = LoRALinear(child.in_features, child.out_features)
                    lora_layer.weight.data = child.weight.data.clone()
                    setattr(module, name, lora_layer)
                else:
                    replace_with_lora_and_freeze(child)
            return module

        encoder = replace_with_lora_and_freeze(encoder)

        # Freeze all parameters except those in LoRA layers
        for param in encoder.parameters():
            param.requires_grad = False

        for name, module in encoder.named_modules():
            if isinstance(module, LoRALinear):
                for param in module.parameters():
                    param.requires_grad = True

        return encoder

    def forward(self, tokenized, abstract_sent_idx):
        "Forward pass for the model."
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
        "Multi-task loss on a batch of inputs."
        res = self(batch["tokenized"], batch["abstract_sent_idx"])
        label_loss = F.cross_entropy(res["label_logits"], batch["label"], reduction="none")
        label_loss = (batch["weight"] * label_loss).sum()

        rationale_loss = masked_binary_cross_entropy_with_logits(
            res["rationale_logits"], batch["rationale"], batch["weight"],
            batch["rationale_mask"])

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
        "Log metrics at end of validation."
        self._log_metrics("train")
        self._log_metrics("valid")

    def test_step(self, batch, batch_idx):
        pred = self(batch["tokenized"], batch["abstract_sent_idx"])
        self._invoke_metrics(pred, batch, "test")

    def test_epoch_end(self, outs):
        "Log metrics at end of test."
        self._log_metrics("train")
        self._log_metrics("test")

    def _invoke_metrics(self, pred, batch, fold):
        "Invoke metrics for a single step of train / validation / test."
        assert fold in ["train", "valid", "test"]
        detached = {k: v.detach() for k, v in pred.items()}
        self.metrics[f"metrics_{fold}"](detached, batch)

    def _log_metrics(self, fold):
        "Log metrics for this epoch."
        the_metric = self.metrics[f"metrics_{fold}"]
        to_log = the_metric.compute()
        the_metric.reset()
        for k, v in to_log.items():
            self.log(f"{fold}_{k}", v)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        steps_per_epoch = math.ceil(
            self.num_training_instances /
            (self.hparams.train_batch_size * self.hparams.accumulate_grad_batches))

        if self.hparams.scheduler_total_epochs is not None:
            n_epochs = self.hparams.scheduler_total_epochs
        else:
            n_epochs = self.hparams.max_epochs

        num_steps = n_epochs * steps_per_epoch
        warmup_steps = int(num_steps * self.frac_warmup)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

        lr_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}

    @pl.core.decorators.auto_move_data
    def predict(self, batch, force_rationale=False):
        "Make predictions on a batch passed in from the dataloader."
        with torch.no_grad():
            output = self(batch["tokenized"], batch["abstract_sent_idx"])
        return self.decode(output, batch, force_rationale)

    @staticmethod
    def decode(output, batch, force_rationale=False):
        "Run decoding to get output in human-readable form."
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


