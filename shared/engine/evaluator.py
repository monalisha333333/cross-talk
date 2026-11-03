import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.metrics import compute_all_metrics, log_metrics


logger = logging.getLogger(__name__)


def evaluate_model(ds, model, device, dataset, batch_size, num_workers):

    model.eval()

    subset_metrics: dict[str, dict[str, float]] = {}
    all_labels: list = []
    all_scores: list = []
    # all_z_globals: list = []
    # all_z_locals: list = []

    for subset_name in dataset.get_subset_names():
        subset_indices = dataset.get_subset_indices(subset_name)
        subset_dataset = Subset(dataset, subset_indices)
        subset_loader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        subset_labels: list = []
        subset_scores: list = []
        subset_z_global: list = []
        subset_z_local: list = []
        subset_z_logits: list = []

        with torch.no_grad():
            for batch in subset_loader:
                # Subset-aware datasets may return either
                #     (image, label, subset_name)   or   (image, label).
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch

                images = images.to(device)
                labels = labels.to(device)
                _, logits, z_local, z_global = model(images, return_feature=True)
                # print(" Predictions: ", _features, labels)
                logits = logits.view(-1)
                probs = torch.sigmoid(logits)

                subset_labels.extend(labels.cpu().numpy())
                subset_scores.extend(probs.cpu().numpy())
                subset_z_global.extend(z_global.squeeze().cpu().numpy())
                subset_z_local.extend(z_local.squeeze().cpu().numpy())
                subset_z_logits.extend(logits.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
                # all_z_globals.extend(z_global.cpu().numpy())
                # all_z_locals.extend(z_local.cpu().numpy())

        subset_y_true = np.array(subset_labels)
        subset_y_score = np.array(subset_scores)
        subset_z_globals = np.array(subset_z_global)
        subset_z_locals = np.array(subset_z_local)
        subset_z_logit = np.array(subset_z_logits)
        metrics = compute_all_metrics(subset_y_true, subset_y_score)
        subset_metrics[subset_name] = metrics
        log_metrics(metrics, subset_name)
        # print("shape of subset_y_true ", subset_y_true.shape)
        # print(" shape of subset_y_score ", subset_y_score.shape)
        # print("shape of subset_z_local ", subset_z_local.shape)
        # print("shape of subset_z_local ", subset_z_local.shape)
        df = pd.DataFrame({
            "target" : subset_y_true,
            "predicted" : subset_y_score,
            "logits" : subset_z_logit,
            "z_global" : subset_z_globals,
            "z_local" : subset_z_locals
        })
        result_path = 'checkpoints/'+ds+'_results_'+subset_name+'.csv'
        df.to_csv(result_path, index=False)
        print("results saved to ", os.path.abspath(result_path))

    y_true = np.array(all_labels)
    y_score = np.array(all_scores)
    overall_metrics = compute_all_metrics(y_true, y_score)
    # z_globals = np.array(all_z_globals)
    # z_locals = np.array(all_z_locals)

    return subset_metrics, overall_metrics
