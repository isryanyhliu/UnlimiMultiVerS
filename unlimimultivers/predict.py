import argparse
import torch
import os
import sys

from pathlib import Path
from tqdm import tqdm


# 将 multivers 和 unlimiformers/src 路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'multivers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs', 'unlimiformers', 'src')))

from libs.unlimiformers.src.unlimiformer import Unlimiformer
from libs.unlimiformers.src.random_training_unlimiformer import RandomTrainingUnlimiformer
from multivers.model import MultiVerSModel
from multivers.data import get_dataloader
from multivers import util
from unlimiformer_config import UnlimiformerArguments  # 配置文件



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--no_nei", action="store_true", help="If given, never predict NEI."
    )
    parser.add_argument(
        "--force_rationale",
        action="store_true",
        help="If given, always predict a rationale for non-NEI.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()

def apply_unlimiformer(model, tokenizer, unlimiformer_args):
    if unlimiformer_args.test_unlimiformer:
        unlimiformer_kwargs = {
            'layer_begin': unlimiformer_args.layer_begin,
            'layer_end': unlimiformer_args.layer_end,
            'unlimiformer_head_num': unlimiformer_args.unlimiformer_head_num,
            'exclude_attention': unlimiformer_args.unlimiformer_exclude,
            'chunk_overlap': unlimiformer_args.unlimiformer_chunk_overlap,
            'model_encoder_max_len': unlimiformer_args.unlimiformer_chunk_size,
            'verbose': unlimiformer_args.unlimiformer_verbose,
            'tokenizer': tokenizer,
            'unlimiformer_training': unlimiformer_args.unlimiformer_training,
            'use_datastore': unlimiformer_args.use_datastore,
            'flat_index': unlimiformer_args.flat_index,
            'test_datastore': unlimiformer_args.test_datastore,
            'reconstruct_embeddings': unlimiformer_args.reconstruct_embeddings,
            'gpu_datastore': unlimiformer_args.gpu_datastore,
            'gpu_index': unlimiformer_args.gpu_index,
            'index_devices': unlimiformer_args.index_devices,
            'datastore_device': unlimiformer_args.datastore_device
        }
        if unlimiformer_args.random_unlimiformer_training:
            model = RandomTrainingUnlimiformer.convert_model(model, **unlimiformer_kwargs)
        else:
            model = Unlimiformer.convert_model(model, **unlimiformer_kwargs)
    return model

def get_predictions(args):
    # Manually extract hparams from checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)

    # Define default hparams in case they are missing in the checkpoint
    default_hparams = {
        "label_weight": 1.0,
        "rationale_weight": 15.0,
        "frac_warmup": 0.1,
        "encoder_name": "allenai/longformer-large-4096",
        "num_labels": 3,
        "lr": 5e-5,
        "scheduler_total_epochs": 3,
        "label_threshold": None,
        "rationale_threshold": 0.5,
        "gradient_checkpointing": False
    }

    hparams_dict = checkpoint.get('hparams', checkpoint.get('hyper_parameters', default_hparams))
    # Ensure all default hparams are included
    for key, value in default_hparams.items():
        if key not in hparams_dict:
            hparams_dict[key] = value
    hparams = argparse.Namespace(**hparams_dict)

    # Initialize model with hparams
    model = MultiVerSModel(hparams)

    # Load model state dict if it exists in the checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Filter out keys that don't match
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

    # If not predicting NEI, set the model label threshold to 0.
    if args.no_nei:
        model.label_threshold = 0.0

    # Load Unlimiformer arguments
    unlimiformer_args = UnlimiformerArguments()

    # Initialize tokenizer
    tokenizer = ...  # 根据你的情况初始化tokenizer

    # Apply Unlimiformer to the model
    model = apply_unlimiformer(model, tokenizer, unlimiformer_args)

    # Since we're not running the training loop, gotta put model on GPU.
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()

    dataloader = get_dataloader(args)

    # Make predictions.
    predictions_all = []

    for batch in tqdm(dataloader):
        preds_batch = model.predict(batch, args.force_rationale)
        predictions_all.extend(preds_batch)

    return predictions_all

def format_predictions(args, predictions_all):
    # Need to get the claim ID's from the original file, since the data loader
    # won't have a record of claims for which no documents were retrieved.
    claims = util.load_jsonl(args.input_file)
    claim_ids = [x["id"] for x in claims]
    assert len(claim_ids) == len(set(claim_ids))

    formatted = {claim: {} for claim in claim_ids}

    # Dict keyed by claim.
    for prediction in predictions_all:
        # If it's NEI, skip it.
        if prediction["predicted_label"] == "NEI":
            continue

        # Add prediction.
        formatted_entry = {
            prediction["abstract_id"]: {
                "label": prediction["predicted_label"],
                "sentences": prediction["predicted_rationale"],
            }
        }
        formatted[prediction["claim_id"]].update(formatted_entry)

    # Convert to jsonl.
    res = []
    for k, v in formatted.items():
        to_append = {"id": k, "evidence": v}
        res.append(to_append)

    return res

def main():
    args = get_args()
    outname = Path(args.output_file)
    predictions = get_predictions(args)

    # Save final predictions as json.
    formatted = format_predictions(args, predictions)
    util.write_jsonl(formatted, outname)

if __name__ == "__main__":
    main()
