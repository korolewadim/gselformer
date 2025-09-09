import os
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def preprocess_function(examples, tokenizer):
    result = tokenizer(
        examples["selfies"],
        truncation=False,
        padding=True,
    )
    result["label"] = [l for l in examples["label"]]
    result["mol_index"] = [i for i in examples["mol_index"]]
    return result


def get_datasets(dataset_dir, tokenizer, cache_dir):
    cache_dir = ensure_dir(cache_dir) if cache_dir else None
    data_files = {
        split: os.path.join(dataset_dir, f"{split}.csv")
        for split in ["train", "test"]
    }
    datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
    tokenized_datasets = datasets.map(
        preprocess_function, batched=True, fn_kwargs=dict(tokenizer=tokenizer)
    )
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    return train_dataset, test_dataset


def get_config(train_dataset, cache_dir, model_name, model_revision="main"):
    cache_dir = ensure_dir(cache_dir) if cache_dir else None
    config_kwargs = {
        "num_labels": 1,
        "problem_type": "regression",
        "cache_dir": cache_dir,
        "revision": model_revision,
        "use_auth_token": None,
    }
    config = AutoConfig.from_pretrained(model_name, **config_kwargs)
    return config


def get_preds(outputs):
    labels = outputs.label_ids
    preds = outputs.predictions.flatten()
    return labels, preds


def compute_metrics(outputs):
    labels = outputs.label_ids
    preds = outputs.predictions.flatten()
    return {
        "r2_score": r2_score(labels, preds),
        "mae": mean_absolute_error(labels, preds),
        "rmse": root_mean_squared_error(labels, preds),
    }


def save_regression_outputs(labels, preds, test_mol_indices, test_selfies, output_dir):
    Path(os.path.join(output_dir)).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_dir, "y_true_flatten"), labels)
    np.save(os.path.join(output_dir, "y_pred_flatten"), preds)

    y_true_average = [
        np.mean(labels[np.argwhere(test_mol_indices == i)]).item()
        for i in np.unique(test_mol_indices)
    ]
    y_pred_average = [
        np.mean(preds[np.argwhere(test_mol_indices == i)]).item()
        for i in np.unique(test_mol_indices)
    ]
    np.save(os.path.join(output_dir, "y_true_average"), y_true_average)
    np.save(os.path.join(output_dir, "y_pred_average"), y_pred_average)

    y_true_sf = [
        labels[np.argwhere(test_mol_indices == i)[0][0]].item()
        for i in np.unique(test_mol_indices)
    ]
    y_pred_sf = [
        preds[np.argwhere(test_mol_indices == i)[0][0]].item()
        for i in np.unique(test_mol_indices)
    ]
    np.save(os.path.join(output_dir, "y_true_selfies"), y_true_sf)
    np.save(os.path.join(output_dir, "y_pred_selfies"), y_pred_sf)

    sf_indices = [
        np.argwhere(test_mol_indices == i)[0, 0] for i in np.unique(test_mol_indices)
    ]
    df_flatten = pd.DataFrame(
        np.array(
            [
                test_mol_indices,
                test_selfies,
                labels,
                preds,
            ]
        ).T,
        columns=["mol_index", "selfies", "y_true", "y_pred"],
    )
    df_average = pd.DataFrame(
        np.array(
            [
                np.unique(test_mol_indices),
                test_selfies[sf_indices],
                y_true_average,
                y_pred_average,
            ]
        ).T,
        columns=["mol_index", "selfies", "y_true", "y_pred"],
    )
    df_selfies = pd.DataFrame(
        np.array(
            [
                np.unique(test_mol_indices),
                test_selfies[sf_indices],
                y_true_sf,
                y_pred_sf,
            ]
        ).T,
        columns=["mol_index", "selfies", "y_true", "y_pred"],
    )
    df_flatten.to_csv(os.path.join(output_dir, "df_flatten.csv"), index=None)
    df_average.to_csv(os.path.join(output_dir, "df_average.csv"), index=None)
    df_selfies.to_csv(os.path.join(output_dir, "df_selfies.csv"), index=None)

    return y_true_average, y_pred_average, y_true_sf, y_pred_sf
