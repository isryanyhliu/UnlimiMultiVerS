### 官方原版的, 但是只能predict scifact

# from tqdm import tqdm
# import argparse
# from pathlib import Path

# from model import MultiVerSModel
# from data import get_dataloader
# import util


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint_path", type=str)
#     parser.add_argument("--input_file", type=str)
#     parser.add_argument("--corpus_file", type=str)
#     parser.add_argument("--output_file", type=str)
#     parser.add_argument("--batch_size", type=int, default=1)
#     parser.add_argument("--device", default=0, type=int)
#     parser.add_argument("--num_workers", default=4, type=int)
#     parser.add_argument(
#         "--no_nei", action="store_true", help="If given, never predict NEI."
#     )
#     parser.add_argument(
#         "--force_rationale",
#         action="store_true",
#         help="If given, always predict a rationale for non-NEI.",
#     )
#     parser.add_argument("--debug", action="store_true")

#     return parser.parse_args()


# def get_predictions(args):
#     # Set up model and data.
#     model = MultiVerSModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)
#     # If not predicting NEI, set the model label threshold to 0.
#     if args.no_nei:
#         model.label_threshold = 0.0

#     # Since we're not running the training loop, gotta put model on GPU.
#     model.to(f"cuda:{args.device}")
#     model.eval()
#     model.freeze()

#     # Grab model hparams and override using new args, when relevant.
#     hparams = model.hparams["hparams"]
#     del hparams.precision  # Don' use 16-bit precision during evaluation.
#     for k, v in vars(args).items():
#         if hasattr(hparams, k):
#             setattr(hparams, k, v)

#     dataloader = get_dataloader(args)

#     # Make predictions.
#     predictions_all = []

#     for batch in tqdm(dataloader):
#         preds_batch = model.predict(batch, args.force_rationale)
#         predictions_all.extend(preds_batch)

#     return predictions_all


# def format_predictions(args, predictions_all):
#     # Need to get the claim ID's from the original file, since the data loader
#     # won't have a record of claims for which no documents were retireved.
#     claims = util.load_jsonl(args.input_file)
#     claim_ids = [x["id"] for x in claims]
#     assert len(claim_ids) == len(set(claim_ids))

#     formatted = {claim: {} for claim in claim_ids}

#     # Dict keyed by claim.
#     for prediction in predictions_all:
#         # If it's NEI, skip it.
#         if prediction["predicted_label"] == "NEI":
#             continue

#         # Add prediction.
#         formatted_entry = {
#             prediction["abstract_id"]: {
#                 "label": prediction["predicted_label"],
#                 "sentences": prediction["predicted_rationale"],
#             }
#         }
#         formatted[prediction["claim_id"]].update(formatted_entry)

#     # Convert to jsonl.
#     res = []
#     for k, v in formatted.items():
#         to_append = {"id": k, "evidence": v}
#         res.append(to_append)

#     return res


# def main():
#     args = get_args()
#     outname = Path(args.output_file)
#     predictions = get_predictions(args)

#     # Save final predictions as json.
#     formatted = format_predictions(args, predictions)
#     util.write_jsonl(formatted, outname)


# if __name__ == "__main__":
#     main()







### 修改过 第一版, 用于predict finetuning 之后的 healthver


# from tqdm import tqdm
# import argparse
# from pathlib import Path

# from model import MultiVerSModel
# from data import get_dataloader
# import util



# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--checkpoint_path", type=str)
#     parser.add_argument("--input_file", type=str)
#     parser.add_argument("--corpus_file", type=str)
#     parser.add_argument("--output_file", type=str)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--device", default=0, type=int)
#     parser.add_argument("--num_workers", default=4, type=int)
#     parser.add_argument(
#         "--no_nei", action="store_true", help="If given, never predict NEI."
#     )
#     parser.add_argument(
#         "--force_rationale",
#         action="store_true",
#         help="If given, always predict a rationale for non-NEI.",
#     )
#     parser.add_argument("--debug", action="store_true")

#     return parser.parse_args()




# def format_predictions(args, predictions_all):
#     # Need to get the claim ID's from the original file, since the data loader
#     # won't have a record of claims for which no documents were retrieved.
#     claims = util.load_jsonl(args.input_file)
#     claim_ids = [x["id"] for x in claims]
#     assert len(claim_ids) == len(set(claim_ids))

#     formatted = {claim: {} for claim in claim_ids}

#     # Dict keyed by claim.
#     for prediction in predictions_all:
#         # If it's NEI, skip it.
#         if prediction["predicted_label"] == "NEI":
#             continue

#         # Add prediction.
#         formatted_entry = {
#             prediction["abstract_id"]: {
#                 "label": prediction["predicted_label"],
#                 "sentences": prediction["predicted_rationale"],
#             }
#         }
#         formatted[prediction["claim_id"]].update(formatted_entry)

#     # Convert to jsonl.
#     res = []
#     for k, v in formatted.items():
#         to_append = {"id": k, "evidence": v}
#         res.append(to_append)
    
#     return res

        


# def get_predictions(args):
#     # Load the checkpoint
#     checkpoint = MultiVerSModel.load_from_checkpoint(checkpoint_path=args.checkpoint_path)

#     # Extract hyperparameters from the checkpoint
#     hparams = checkpoint.hparams

#     # Initialize the model with the hyperparameters from the checkpoint
#     model = MultiVerSModel(hparams)
#     model.load_state_dict(checkpoint.state_dict())

#     # If not predicting NEI, set the model label threshold to 0.
#     if args.no_nei:
#         model.label_threshold = 0.0

#     # Since we're not running the training loop, gotta put model on GPU.
#     model.to(f"cuda:{args.device}")
#     model.eval()
#     model.freeze()

#     # Grab model hparams and override using new args, when relevant.
#     if hasattr(hparams, "precision"):
#         del hparams['precision']  # Don't use 16-bit precision during evaluation.
    
#     for k, v in vars(args).items():
#         if hasattr(hparams, k):
#             hparams[k] = v

#     dataloader = get_dataloader(args)

#     # Make predictions.
#     predictions_all = []

#     for batch in tqdm(dataloader):
#         preds_batch = model.predict(batch, args.force_rationale)
#         predictions_all.extend(preds_batch)

#     return predictions_all

# def main():
#     args = get_args()
#     outname = Path(args.output_file)
#     predictions = get_predictions(args)

#     # Save final predictions as json.
#     formatted = format_predictions(args, predictions)
#     util.write_jsonl(formatted, outname)



# if __name__ == "__main__":
#     main()














### 修改过 第二版, 用于predict longformer_large_science
### 解决  hparams.label_weight 问题
### OK: fever, feversci, longformer -> healthver, covidfact

### AttributeError: 'Namespace' object has no attribute 'label_weight'


import argparse
from pathlib import Path
from tqdm import tqdm
from model import MultiVerSModel
from data import get_dataloader
import util
import torch




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


### 第三版, 升级了transformer之后报错 (unlimiformer 需要升级版本)
### KeyError: 'embeddings.position_ids'
