import os
import yaml
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from restart.utils.plots import plot_losses
from restart.data.dataset import SKU110KDataset
from restart.model.detector import ObjectDetector
from restart.utils.visualize_detections import visualize_detections

# ---- LOAD CONFIG ----
with open("restart/config/training_config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

# ---- DYNAMIC RUN DIRECTORY ----
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"TR_{timestamp}"
run_dir = os.path.join(config["output_dir"], run_name)
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, config["save_dir"]), exist_ok=True)
os.makedirs(os.path.join(run_dir, config["visualization_dir"]), exist_ok=True)

# ---- DATASET ----
dataset = SKU110KDataset(
    data_dir=config["dataset_path"],
    split="train",
    resize_dims=tuple(config["resize_dims"])
)
if config.get("subset_size"):
    dataset = torch.utils.data.Subset(dataset, list(range(config["subset_size"])))

loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    collate_fn=SKU110KDataset.collate_fn
)

# ---- MODEL ----
model = ObjectDetector(
    pretrained_backbone=True,
    num_classes=1,
    num_anchors=9,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

losses_cls, losses_box = [], []
print("[✓] Starting Training...")

for epoch in range(config["num_epochs"]):
    model.train()
    epoch_cls_loss, epoch_box_loss = 0.0, 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
    for batch in pbar:
        images = batch["images"].to(DEVICE)
        boxes = [b.to(DEVICE) for b in batch["boxes"]]
        labels = [l.to(DEVICE) for l in batch["labels"]]

        optimizer.zero_grad()
        outputs = model(images, boxes=boxes, labels=labels)

        loss = outputs["cls_loss"] + outputs["box_loss"]
        loss.backward()
        optimizer.step()

        cls_val = outputs["cls_loss"].item()
        box_val = outputs["box_loss"].item()

        epoch_cls_loss += cls_val
        epoch_box_loss += box_val

        pbar.set_postfix({"ClsLoss": f"{cls_val:.4f}", "BoxLoss": f"{box_val:.4f}"})

    avg_cls = epoch_cls_loss / len(loader)
    avg_box = epoch_box_loss / len(loader)
    losses_cls.append(avg_cls)
    losses_box.append(avg_box)

    print(f"[Epoch {epoch+1}] ClsLoss: {avg_cls:.4f}, BoxLoss: {avg_box:.4f}")

    ckpt_path = os.path.join(run_dir, config["save_dir"], f"model_epoch_{epoch+1}.pt")
    torch.save(model.state_dict(), ckpt_path)

    # ---- PER-EPOCH VISUALIZATION ----
    if config.get("visualize_per_epoch", False):
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(loader))
            images = sample_batch["images"].to(DEVICE)
            boxes = sample_batch["boxes"]
            labels = sample_batch["labels"]
            orig_sizes = sample_batch["orig_sizes"]
            resize_sizes = sample_batch["resize_sizes"]

            outputs = model(images)

            for idx in range(min(config["num_visualizations"], len(images))):
                filename = f"epoch_{epoch+1}_img_{idx+1}.png"
                save_path = os.path.join(run_dir, config["visualization_dir"], filename)
                visualize_detections(
                    image_tensor=images[idx],
                    detections=outputs["detections"][idx],
                    ground_truths={"boxes": boxes[idx], "labels": labels[idx]},
                    orig_size=orig_sizes[idx],
                    resize_size=resize_sizes[idx],
                    title=f"[Epoch {epoch+1}] Pred vs GT",
                    save_path=os.path.join(run_dir, config["visualization_dir"], f"epoch_{epoch+1}_img_{idx+1}.png")
                )

# ---- PLOT TRAINING LOSS ----
plot_losses(losses_cls, losses_box, os.path.join(run_dir, "training_loss.png"))
