"""Training and validation utilities for EE-PAT on Charades."""

import argparse
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import optim
from tqdm import tqdm

from apmeter import APMeter
from charades_dataloader import Charades, collate_fn_unisize
from utils import AsymmetricLoss, mask_probs, str2bool
from .model import EEPAT


NUM_CLASSES = 157


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EE-PAT for dense multi-label action detection on Charades."
    )
    parser.add_argument("--mode", choices=("rgb", "flow"), default="rgb")
    parser.add_argument("--train", type=str2bool, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--rgb-root")
    parser.add_argument("--flow-root")
    parser.add_argument("--output-dir", default="outputs/eepat")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=(7, 14))
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-clips", type=int, default=256)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--validation-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fine-weight", type=float, default=0.1)
    parser.add_argument("--coarse-weight", type=float, default=0.9)
    args = parser.parse_args()

    feature_root = args.flow_root if args.mode == "flow" else args.rgb_root
    if not feature_root:
        parser.error(f"--{args.mode}-root is required for --mode {args.mode}")
    if args.batch_size <= 0 or args.num_clips <= 0 or args.epochs <= 0:
        parser.error("--batch-size, --num-clips, and --epochs must be positive")
    if any(milestone <= 0 for milestone in args.lr_milestones):
        parser.error("--lr-milestones must contain positive epochs")
    if args.lr_gamma <= 0:
        parser.error("--lr-gamma must be positive")
    if args.fine_weight < 0 or args.coarse_weight < 0:
        parser.error("prediction weights must be non-negative")
    if args.fine_weight + args.coarse_weight == 0:
        parser.error("at least one prediction weight must be positive")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random seed:", seed)


def create_dataloaders(args, feature_root, device):
    print("Padding all sequences to", args.num_clips, "clips")
    collate = collate_fn_unisize(args.num_clips).charades_collate_fn_unisize
    print("Loading features from", feature_root)
    train_data = Charades(
        args.annotation_file, "training", feature_root, args.batch_size,
        NUM_CLASSES, args.num_clips, args.skip,
    )
    validation_data = Charades(
        args.annotation_file, "testing", feature_root, 1,
        NUM_CLASSES, args.num_clips, args.skip,
    )
    common = {"collate_fn": collate, "pin_memory": device.type == "cuda"}
    return {
        "train": torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, **common,
        ),
        "validation": torch.utils.data.DataLoader(
            validation_data, batch_size=1, shuffle=False,
            num_workers=args.validation_workers, **common,
        ),
    }


def create_model():
    return EEPAT(
        input_dim=1024,
        embedding_dim=512,
        num_classes=NUM_CLASSES,
        num_blocks=3,
        num_heads=8,
        max_length=256,
        mlp_ratio=8,
        granularity_strides=(2, 4, 8),
    )


def move_batch_to_device(batch, device):
    features, mask, labels, metadata, heatmap = batch
    features = features.to(device, non_blocking=True).squeeze(3).squeeze(3)
    return (
        features,
        mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        metadata,
        heatmap,
    )


def forward_batch(model, batch, device, loss_fn, fine_weight, coarse_weight,
                  assistant=False):
    features, mask, labels, metadata, heatmap = move_batch_to_device(batch, device)
    if assistant:
        logits = model.forward_assistant(labels)
        coarse_logits, fine_logits = logits, logits
    else:
        coarse_logits, fine_logits = model.forward_core(features)

    weight_sum = fine_weight + coarse_weight
    probabilities = (
        fine_weight * fine_logits.sigmoid()
        + coarse_weight * coarse_logits.sigmoid()
    ) / weight_sum
    valid = mask.bool()
    valid_count = valid.sum().clamp_min(1)
    fine_loss = loss_fn(fine_logits[valid], labels[valid]) / valid_count
    coarse_loss = loss_fn(coarse_logits[valid], labels[valid]) / valid_count
    loss = (fine_weight * fine_loss + coarse_weight * coarse_loss) / weight_sum
    return {
        "loss": loss,
        "probabilities": probabilities,
        "mask": mask,
        "labels": labels,
        "metadata": metadata,
        "heatmap": heatmap,
    }


def set_training_branch(model, assistant):
    for parameter in model.assistant.parameters():
        parameter.requires_grad = assistant
    for parameter in model.core.parameters():
        parameter.requires_grad = not assistant
    for parameter in model.video_classifier.parameters():
        parameter.requires_grad = False


def add_to_meter(meter, result):
    valid = result["mask"].bool()
    meter.add(
        result["probabilities"][valid].detach().cpu(),
        result["labels"][valid].detach().cpu(),
    )


def train_one_epoch(model, loader, device, optimizer, loss_fn, args):
    model.train()
    assistant_loss = core_loss = 0.0
    meter = APMeter()
    progress = tqdm(loader, desc="Training", unit="batch", dynamic_ncols=True)
    for index, batch in enumerate(progress, 1):
        set_training_branch(model, assistant=True)
        optimizer.zero_grad(set_to_none=True)
        assistant_result = forward_batch(
            model, batch, device, loss_fn, args.fine_weight,
            args.coarse_weight, assistant=True,
        )
        assistant_result["loss"].backward()
        optimizer.step()
        model.copy_assistant_classifier()

        set_training_branch(model, assistant=False)
        optimizer.zero_grad(set_to_none=True)
        core_result = forward_batch(
            model, batch, device, loss_fn, args.fine_weight, args.coarse_weight
        )
        core_result["loss"].backward()
        optimizer.step()

        add_to_meter(meter, core_result)
        assistant_loss += assistant_result["loss"].item()
        core_loss += core_result["loss"].item()
        progress.set_postfix(
            assistant_loss=f"{assistant_loss / index:.3f}",
            core_loss=f"{core_loss / index:.3f}",
        )
    count = max(len(loader), 1)
    return 100 * meter.value().mean().item(), assistant_loss / count, core_loss / count


@torch.no_grad()
def validate(model, loader, device, loss_fn, args):
    model.eval()
    meter = APMeter()
    total_loss = 0.0
    predictions = {}
    progress = tqdm(loader, desc="Validation", unit="video", dynamic_ncols=True)
    for index, batch in enumerate(progress, 1):
        result = forward_batch(
            model, batch, device, loss_fn, args.fine_weight, args.coarse_weight
        )
        add_to_meter(meter, result)
        total_loss += result["loss"].item()
        progress.set_postfix(loss=f"{total_loss / index:.3f}")
        video_id = result["metadata"][0][0]
        probabilities = result["probabilities"][0].detach().cpu().numpy()
        mask = result["mask"][0].detach().cpu().numpy()
        predictions[video_id] = mask_probs(probabilities, mask).squeeze().T
    return predictions, total_loss / max(len(loader), 1), 100 * meter.value().mean().item()


def save_checkpoint(path, model, optimizer, scheduler, epoch, validation_map, args):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation_map": validation_map,
            "args": vars(args),
        },
        path,
    )


def train(model, loaders, device, optimizer, scheduler, loss_fn, args, output_dir):
    best_map = float("-inf")
    start = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch}/{args.epochs - 1}\n----------")
        train_map, assistant_loss, core_loss = train_one_epoch(
            model, loaders["train"], device, optimizer, loss_fn, args
        )
        predictions, validation_loss, validation_map = validate(
            model, loaders["validation"], device, loss_fn, args
        )
        scheduler.step()
        print(
            f"train mAP={train_map:.4f}, stage-1 loss={assistant_loss:.4f}, "
            f"stage-2 loss={core_loss:.4f}"
        )
        print(f"validation mAP={validation_map:.4f}, loss={validation_loss:.4f}")
        print(
            f"epoch time={time.time() - epoch_start:.1f}s, "
            f"total time={time.time() - start:.1f}s"
        )
        prediction_path = output_dir / f"{epoch}.pkl"
        with prediction_path.open("wb") as output_file:
            pickle.dump(predictions, output_file, pickle.HIGHEST_PROTOCOL)
        print("Saved predictions to", prediction_path)
        if validation_map > best_map:
            best_map = validation_map
            save_checkpoint(
                output_dir / "best_checkpoint.pt", model, optimizer, scheduler,
                epoch, validation_map, args,
            )
            print(f"New best validation mAP: {best_map:.4f}")


def main():
    args = parse_args()
    seed_everything(args.seed)
    if not args.train:
        raise NotImplementedError("Evaluation-only mode is not implemented.")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    feature_root = args.flow_root if args.mode == "flow" else args.rgb_root
    loaders = create_dataloaders(args, feature_root, device)
    print("Training batches:", len(loaders["train"]))
    print("Validation videos:", len(loaders["validation"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
    )
    train(
        model, loaders, device, optimizer, scheduler,
        AsymmetricLoss(), args, output_dir,
    )
