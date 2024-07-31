import argparse
import torch
import os
import sys
import time

from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer 
from transformers import HfArgumentParser

# 将 multivers 和 unlimiformers/src 路径添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'multivers')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'libs', 'unlimiformers', 'src')))

from unlimiformer_config import UnlimiformerArguments  # 配置文件
from libs.unlimiformers.src.unlimiformer import Unlimiformer
from libs.unlimiformers.src.random_training_unlimiformer import RandomTrainingUnlimiformer
from multivers.model import MultiVerSModel
from multivers.data import get_dataloader
import multivers.util as util

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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--no_nei", action="store_true", help="If given, never predict NEI.")
    parser.add_argument("--force_rationale", action="store_true", help="If given, always predict a rationale for non-NEI.")
    parser.add_argument("--debug", action="store_true")

    # Add Unlimiformer arguments
    parser = UnlimiformerArguments.add_arguments_to_parser(parser)

    return parser.parse_args()


import torch.autograd.profiler as profiler
def get_predictions(args, unlimiformer_args):
    import time
    checkpoint = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
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
    for key, value in default_hparams.items():
        if key not in hparams_dict:
            hparams_dict[key] = value
    hparams = argparse.Namespace(**hparams_dict)
    model = MultiVerSModel(hparams, unlimiformer_args)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape}
    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)
    if args.no_nei:
        model.label_threshold = 0.0
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()

    dataloader = get_dataloader(args)

    predictions_all = []

    start_time = time.time()
    print(f"DataLoader initialized in {time.time() - start_time:.2f} seconds")

    for batch in tqdm(dataloader):
        start_time = time.time()
        batch = move_to_device(batch, f"cuda:{args.device}")
        print(f"Data moved to device in {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        preds_batch = model.predict(batch, args.force_rationale)
        print(f"Prediction completed in {time.time() - start_time:.2f} seconds")

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
    unlimiformer_args = UnlimiformerArguments()
    unlimiformer_args.unlimiformer_layer_begin = args.unlimiformer_layer_begin
    unlimiformer_args.unlimiformer_layer_end = args.unlimiformer_layer_end
    unlimiformer_args.unlimiformer_head_num = args.unlimiformer_head_num
    unlimiformer_args.unlimiformer_exclude_attention = args.unlimiformer_exclude_attention
    unlimiformer_args.unlimiformer_max_len = args.unlimiformer_max_len
    unlimiformer_args.unlimiformer_chunk_overlap = args.unlimiformer_chunk_overlap
    unlimiformer_args.unlimiformer_verbose = args.unlimiformer_verbose
    unlimiformer_args.tokenizer = args.tokenizer
    unlimiformer_args.random_unlimiformer_training = args.random_unlimiformer_training
    unlimiformer_args.unlimiformer_training = args.unlimiformer_training
    unlimiformer_args.use_datastore = args.use_datastore
    unlimiformer_args.flat_index = args.flat_index
    unlimiformer_args.test_datastore = args.test_datastore
    unlimiformer_args.reconstruct_embeddings = args.reconstruct_embeddings
    unlimiformer_args.gpu_datastore = args.gpu_datastore
    unlimiformer_args.gpu_index = args.gpu_index

    outname = Path(args.output_file)
    predictions = get_predictions(args, unlimiformer_args)

    # Save final predictions as json.
    formatted = format_predictions(args, predictions)
    util.write_jsonl(formatted, outname)

if __name__ == "__main__":
    main()