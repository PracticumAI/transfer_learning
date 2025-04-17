'''Helper functions for the Transfer Learning course'''


import arff  # Note that this is liac-arff, **not** arff
import datasets
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import requests
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, accuracy_score,
                           precision_recall_fscore_support, roc_auc_score)
import tarfile
import time
import torch
import torchvision
from torchvision import transforms
from transformers import  AutoTokenizer
import zipfile


def download_and_extract_data(download_url, file_name, dest_path, folder_names=None):
    """
    Download and extract a compressed file into the specified path.
    
    Parameters:
    -----------
    download_url : str
        URL to download the compressed file from
    file_name : str
        Name of the compressed file
    data_path : str
        Path to extract the file to
    folder_names : list, optional
        List of folder names expected after extraction to verify completion
        
    """
    
    # If folder_names provided, check if data already exists
    if folder_names is not None:
        all_folders_exist = all(os.path.exists(os.path.join(dest_path, folder)) 
                                for folder in folder_names)
        if all_folders_exist:
            print("Data is already downloaded.")
            return

    # File name is combination of the url and file_name
    download_file = os.path.join(download_url, file_name)

    # Create the data directory if it does not exist
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Download the file
    print(f"Downloading the data from {download_file}.\n This may take a few minutes.")
    with requests.get(download_file, stream=True) as r:
        r.raise_for_status()
        with open(os.path.join(dest_path, file_name), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Extract the file
    print(f"Extracting the data file {file_name}, this may take a few minutes.")
    if file_name.endswith('.tar.gz'):
        with tarfile.open(os.path.join(dest_path, file_name), "r:gz") as tar:
            tar.extractall(path=dest_path, filter="data")
    elif file_name.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(dest_path, file_name), 'r') as zip_ref:
            zip_ref.extractall(dest_path)
    
    # If folder_names provided, check if data already exists
    if folder_names is not None:
        all_folders_exist = all(os.path.exists(os.path.join(dest_path, folder)) 
                                for folder in folder_names)
        if all_folders_exist:
            # Remove the compressed file
            os.remove(os.path.join(dest_path, file_name))
            print(f"Data has been downloaded and extracted into {dest_path}")
            return
        else:
            print("Data extraction failed, please check the folder names.")

    
    return 

def explore_data(train_dir, val_dir, test_dir):
    """Explore the dataset by printing the number of images in each class."""
    # Count the number of images in each class
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir)
    val_dataset = torchvision.datasets.ImageFolder(root=val_dir)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dir)

    print(f"Number of training images: {len(train_dataset):,}")
    print(f"Number of validation images: {len(val_dataset):,}")
    print(f"Number of test images: {len(test_dataset):,}")
    

    # Visualize some images from the training set    
    # --- Set up the transforms and data loader for visualization ---
    visualization_transforms = transforms.Compose(
        [
            transforms.Resize(256),  # Resize the smaller edge to 256
            transforms.CenterCrop(224),  # Crop the center 224x224 pixels
            transforms.ToTensor(),  # Convert image to PyTorch Tensor (scales pixels to [0, 1])
        ]
    )

    vis_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=visualization_transforms)

    # Get class names from the dataset folders
    class_names = vis_dataset.classes

    # Get the total number of images in the training set

    print(
        f"\n\nThere are {len(class_names)} classes in the dataset, with the labels:"
    )

    # The class names are in the format: Crop_disease, or Crop_healthy.
    # Print the lst of categories, printing all those with the same "Crop_" prefix on one line
    last = None
    for class_name in class_names:
        if last == None:
            print(f"   {class_name},", end=" ")
        elif class_name.startswith(last):
            print(f"{class_name},", end=" ")
        else:
            print(f"\n   {class_name},", end=" ")
        last = class_name.split("_")[0]    


    # Select 9 random indices
    random_indices = random.sample(range(len(train_dataset)), 9)

    # --- Display the images ---
    plt.figure(figsize=(10, 10))  # Adjust figure size as needed
    plt.suptitle(
        "Nine Random Images from the Training Set", fontsize=16
    )  # Add a title to the figure

    for i, idx in enumerate(random_indices):
        ax = plt.subplot(3, 3, i + 1)

        # Get the image and label from the dataset using the random index
        image_tensor, label_index = vis_dataset[idx]

        # Image tensors from ToTensor() are CxHxW and values are [0, 1].
        # Matplotlib expects HxWxC and values [0, 1] for floats or [0, 255] for integers.
        # We need to rearrange dimensions using permute.
        image_for_plot = image_tensor.permute(1, 2, 0).numpy()

        # Display the image
        plt.imshow(image_for_plot)

        # Get the class name using the label index
        label_name = class_names[label_index]
        plt.title(f"Label: {label_name}")

        plt.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()

    return class_names


def plot_confusion_matrix(model, dataloader, class_names):
    '''Plot a confusion matrix for the model predictions.'''
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Iterate through the test set and make predictions
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    # Plot confusion matrix    
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.title("Confusion Matrix")
    plt.colorbar()  
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

def load_dataset(dataset):
    """ Load the dataset """
    # Check if the dataset file exists and is not empty
    try:
        # Use liac-arff to load the ARFF file
        with open(dataset, "r") as f:
            arff_data = arff.load(f)
            data = pd.DataFrame(
                arff_data["data"],
                columns=[attr[0] for attr in list(arff_data["attributes"])],
            )
    except FileNotFoundError:
        print(f"ERROR: Dataset file not found at {dataset}.")
        print("Please ensure you have downloaded the dataset and the path is correct.")
        data = None  # Set data to None if file not found

    if data is not None:
        # Convert to Pandas DataFrame
        df = pd.DataFrame(data)
        print(f"Loaded DataFrame shape: {df.shape}")

        # Decode byte strings (Common in ARFF)
        # Identify potential text columns (adjust names based on actual columns in meta/df.info())
        text_col = "text"
        label_col = "category"

        if text_col in df.columns and df[text_col].dtype == "object":
            # Check if decoding is needed (inspect first element)
            if isinstance(df[text_col].iloc[0], bytes):
                print(f"Decoding byte strings in column '{text_col}'...")
                df[text_col] = df[text_col].str.decode("utf-8")

        if label_col in df.columns and df[label_col].dtype == "object":
            if isinstance(df[label_col].iloc[0], bytes):
                print(f"Decoding byte strings in column '{label_col}'...")
                df[label_col] = df[label_col].str.decode("utf-8")

        # Map String Labels to Integer IDs
        unique_labels = df[label_col].unique()
        num_labels = len(unique_labels)

        # Create mappings
        label2id = {label: i for i, label in enumerate(unique_labels)}
        id2label = {i: label for label, i in label2id.items()}

        # Apply mapping to create a new 'label' column
        df["label"] = df[label_col].map(label2id)

        print(f"Number of classes: {num_labels}")
        print("Label mappings created:")
        print(f"  label2id: {label2id}")
        print(f"  id2label: {id2label}")

        # Inspect the DataFrame
        print("\nDataFrame Head:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()
        print("\nLabel Distribution:")
        print(df["label"].value_counts())

        # Keep only relevant columns (text and integer label)
        df = df[[text_col, "label"]]
        df = df.rename(columns={text_col: "text"})  # Ensure text column is named 'text'

        return df, label2id, id2label, num_labels
    else:
        print("Skipping DataFrame processing as data was not loaded.")


def prepare_data(df, model_name):
    """Convert Pandas DataFrame to Hugging Face Dataset"""

    hf_dataset = datasets.Dataset.from_pandas(df)
    print("\nConverted to Hugging Face Dataset:")
    print(hf_dataset)

    # Split into training and validation sets (e.g., 80% train, 20% validation)
    train_test_split_ratio = 0.20
    dataset_dict = hf_dataset.train_test_split(
        test_size=train_test_split_ratio, shuffle=True, seed=42
    )  # Use seed for reproducibility

    # Rename for clarity
    train_ds = dataset_dict["train"]
    eval_ds = dataset_dict["test"]

    print("\nSplit into Train and Validation Sets:")
    print(f"  Training examples: {len(train_ds)}")
    print(f"  Validation examples: {len(eval_ds)}")
    print(train_ds)  # Show structure

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize the dataset
    def tokenize_function(example):
        return tokenizer(
            example["text"], truncation=True, padding="max_length", max_length=128
        )

    train_ds = train_ds.map(tokenize_function, batched=True)
    eval_ds = eval_ds.map(tokenize_function, batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    return train_ds, eval_ds, tokenizer


def get_comprehensive_metrics(y_true, y_pred, y_scores=None):
    """
    Calculate comprehensive metrics for model evaluation

    Parameters:
    y_true: Ground truth labels
    y_pred: Predicted labels
    y_scores: Prediction probabilities (for ROC AUC)
    """
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # ROC AUC (if probabilities available)
    auc = None
    if y_scores is not None:
        # For binary classification
        if y_scores.shape[1] == 2:
            auc = roc_auc_score(y_true, y_scores[:, 1])
        # For multi-class
        else:
            auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="weighted")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def plot_model_comparison_bars(lora_metrics, fe_metrics):
    """Create bar chart comparing LoRA and Feature Extraction metrics"""
    metrics = list(lora_metrics.keys())

    # Filter out None values
    valid_metrics = [
        m for m in metrics if lora_metrics[m] is not None and fe_metrics[m] is not None
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(valid_metrics))
    width = 0.35

    # Plot bars
    ax.bar(x - width / 2, [lora_metrics[m] for m in valid_metrics], width, label="LoRA")
    ax.bar(
        x + width / 2,
        [fe_metrics[m] for m in valid_metrics],
        width,
        label="Feature Extraction",
    )

    # Add labels and formatting
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(valid_metrics)
    ax.legend()

    # Add value labels on bars
    for i, v in enumerate([lora_metrics[m] for m in valid_metrics]):
        ax.text(i - width / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)
    for i, v in enumerate([fe_metrics[m] for m in valid_metrics]):
        ax.text(i + width / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=9)

    plt.ylim(0, 1.1)  # Ensure there's space for labels
    plt.tight_layout()
    return fig


def compare_models_learning_curves(results_dict, metric_names=None):
    """Compare different models based on their evaluation metrics"""
    if metric_names is None:
        metric_names = ["eval_accuracy", "eval_f1"]

    # Create a figure with subplots for each metric
    fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
    if len(metric_names) == 1:
        axes = [axes]

    for i, metric in enumerate(metric_names):
        for model_name, results in results_dict.items():
            if metric in results:
                # Get the metric values
                values = results[metric]

                # Handle single tensor
                if torch.is_tensor(values):
                    values = values.detach().cpu().numpy()
                # Handle list of values that might be tensors
                elif isinstance(values, list):
                    values = [
                        v.detach().cpu().numpy() if torch.is_tensor(v) else v
                        for v in values
                    ]

                axes[i].plot(values, marker="o", label=f"{model_name}")

        axes[i].set_title(f"{metric} across epochs")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    return fig


def evaluate_and_compare(
    lora_model, feature_extraction_model, test_dataloader, device, id2label=None
):
    # Get predictions
    lora_preds, lora_true, lora_scores, lora_time = get_predictions(
        lora_model, test_dataloader, device
    )
    fe_preds, fe_true, fe_scores, fe_time = get_predictions(
        feature_extraction_model, test_dataloader, device
    )

    # Get metrics
    lora_metrics = get_comprehensive_metrics(lora_true, lora_preds, lora_scores)
    fe_metrics = get_comprehensive_metrics(fe_true, fe_preds, fe_scores)

    # Print comparison table
    print(f"{'Metric':<15} {'LoRA':<10} {'Feature Extraction':<20}")
    print("=" * 45)
    for metric in lora_metrics.keys():
        if lora_metrics[metric] is not None and fe_metrics[metric] is not None:
            print(f"{metric:<15} {lora_metrics[metric]:.4f}   {fe_metrics[metric]:.4f}")

    # Print efficiency metrics
    print(f"\nInference time (s):")
    print(f"LoRA: {lora_time:.4f}")
    print(f"Feature Extraction: {fe_time:.4f}")

    # Count trainable parameters
    lora_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    fe_trainable = sum(
        p.numel() for p in feature_extraction_model.parameters() if p.requires_grad
    )
    print(f"\nTrainable parameters:")
    print(f"LoRA: {lora_trainable:,}")
    print(f"Feature Extraction: {fe_trainable:,}")

    # Plot comparison bar chart
    plt.figure(1)
    plot_model_comparison_bars(lora_metrics, fe_metrics)
    plt.tight_layout()
    plt.show()

    # Plot confusion matrices with class labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Get class labels for confusion matrix
    labels = None
    if id2label:
        labels = [id2label[i] for i in range(len(id2label))]

    # Create confusion matrices with labels
    cm1 = ConfusionMatrixDisplay.from_predictions(
        lora_true,
        lora_preds,
        ax=ax1,
        normalize="true",
        display_labels=labels if labels else None,
    )
    ax1.set_title("LoRA Confusion Matrix")
    ax1.tick_params(axis="x", rotation=45)

    cm2 = ConfusionMatrixDisplay.from_predictions(
        fe_true,
        fe_preds,
        ax=ax2,
        normalize="true",
        display_labels=labels if labels else None,
    )
    ax2.set_title("Feature Extraction Confusion Matrix")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # Return metrics for further analysis if needed
    return {"lora": lora_metrics, "feature_extraction": fe_metrics}


def get_predictions(model, dataloader, device):
    """Get predictions, true labels, and measure inference time"""
    model.eval()
    model.to(device)
    predictions = []
    true_labels = []
    scores = []

    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            predictions.extend(preds)
            true_labels.extend(labels)
            scores.append(probs)

    inference_time = time.time() - start_time
    return (
        np.array(predictions),
        np.array(true_labels),
        np.vstack(scores),
        inference_time,
    )


# Plot learning curves for both models
def plot_learning_curves(lora_results, fe_results):
    # Combine results for comparison
    results_dict = {"LoRA": lora_results, "Feature Extraction": fe_results}

    # Plot learning curves
    fig = compare_models_learning_curves(
        results_dict,
        ["train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1"],
    )
    plt.show()
    return fig


# Interactive inference


def run_inference_on_text(text, lora_model, feature_model, tokenizer, device, id2label):
    """
    Run inference on user-provided text with both models

    Parameters:
    text (str): Input text for classification
    lora_model: Trained LoRA model
    feature_model: Trained Feature Extraction model
    tokenizer: Tokenizer for preprocessing
    device: Device to run inference on
    id2label (dict): Mapping from label ids to human-readable labels
    """
    # Move models to eval mode
    lora_model.eval()
    feature_model.eval()

    # Tokenize the input text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # Run inference with LoRA model
    with torch.no_grad():
        lora_outputs = lora_model(**inputs)
        fe_outputs = feature_model(**inputs)

        # Get predictions
        lora_logits = lora_outputs.logits
        fe_logits = fe_outputs.logits

        lora_probs = torch.nn.functional.softmax(lora_logits, dim=-1).cpu().numpy()[0]
        fe_probs = torch.nn.functional.softmax(fe_logits, dim=-1).cpu().numpy()[0]

        lora_pred_id = lora_logits.argmax(dim=-1).cpu().numpy()[0]
        fe_pred_id = fe_logits.argmax(dim=-1).cpu().numpy()[0]

        lora_pred_label = id2label[lora_pred_id]
        fe_pred_label = id2label[fe_pred_id]

    # Print results
    print("\n" + "=" * 50)
    print(f'Input Text: "{text}"')
    print("=" * 50)
    print("\nPredictions:")
    print(
        f"  LoRA Model: {lora_pred_label} (confidence: {lora_probs[lora_pred_id]:.4f})"
    )
    print(
        f"  Feature Extraction Model: {fe_pred_label} (confidence: {fe_probs[fe_pred_id]:.4f})"
    )

    print("\nConfidence Distribution:")
    print(f"{'Class':<20} {'LoRA':<10} {'Feature Extraction':<20}")
    print("-" * 50)
    for i, label in id2label.items():
        print(f"{label:<20} {lora_probs[i]:.4f}    {fe_probs[i]:.4f}")

    # Create a bar chart comparing prediction probabilities
    plt.figure(figsize=(10, 6))
    x = np.arange(len(id2label))
    width = 0.35

    plt.bar(x - width / 2, lora_probs, width, label="LoRA")
    plt.bar(x + width / 2, fe_probs, width, label="Feature Extraction")

    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
    plt.xticks(x, [id2label[i] for i in range(len(id2label))])
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return {
        "lora": {"label": lora_pred_label, "probs": lora_probs},
        "feature_extraction": {"label": fe_pred_label, "probs": fe_probs},
    }


def get_target_modules_for_model(model_name):
    """Return appropriate target modules based on the model architecture."""
    if "distilbert" in model_name:
        return ["q_lin", "k_lin", "v_lin"]
    elif "bert" in model_name:
        return ["query", "key", "value"]
    elif "roberta" in model_name:
        return ["query", "key", "value"]
    elif "gpt" in model_name:
        return ["q_proj", "k_proj", "v_proj"]
    elif "t5" in model_name:
        return ["q", "k", "v"]
    else:
        # Default to the most common pattern
        return ["query", "key", "value"]