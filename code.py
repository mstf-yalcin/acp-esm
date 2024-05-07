import os
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report, roc_curve, auc
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
from transformers.integrations import TrainerCallback, is_tensorboard_available
import shutil

# Define Model
model_checkpoint = "facebook/esm2_t12_35M_UR50D"

# Define Dataset Path
poz_path = '/dataset/deepgram/trainingCancer_422.csv'
non_path = '/dataset/deepgram/trainingNonCancer_422.csv'
val_poz_path = '/dataset/deepgram/independentCancer_150.csv'
val_non_path = '/dataset/deepgram/independentNonCancer_150.csv'


# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Tokenize data
def tokenize_data(tokenizer, data):
    return tokenizer(data, return_tensors="pt", truncation=True)

# Define dataset class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# Compute evaluation metrics
def compute_metrics(labels, preds):
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mcc': mcc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

def custom_rewrite_logs(d, mode):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if mode == 'eval' and k.startswith(eval_prefix):
            if k[eval_prefix_len:] == 'loss':
                new_d["combined/" + k[eval_prefix_len:]] = v
        elif mode == 'test' and k.startswith(test_prefix):
            if k[test_prefix_len:] == 'loss':
                new_d["combined/" + k[test_prefix_len:]] = v
        elif mode == 'train':
            if k == 'loss':
                new_d["combined/" + k] = v
    return new_d         

# Define TensorBoard callback
class CombinedTensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).
    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writers=None):
        has_tensorboard = is_tensorboard_available()
        if not has_tensorboard:
            raise RuntimeError(
                "TensorBoardCallback requires tensorboard to be installed. Either update your PyTorch version or"
                " install tensorboardX."
            )
        if has_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # noqa: F401

                self._SummaryWriter = SummaryWriter
            except ImportError:
                try:
                    from tensorboardX import SummaryWriter

                    self._SummaryWriter = SummaryWriter
                except ImportError:
                    self._SummaryWriter = None
        else:
            self._SummaryWriter = None
        self.tb_writers = tb_writers

    def _init_summary_writer(self, args, log_dir=None):
        log_dir = log_dir or args.logging_dir
        if self._SummaryWriter is not None:
            self.tb_writers = dict(train=self._SummaryWriter(log_dir=os.path.join(log_dir, 'train')),
                                   eval=self._SummaryWriter(log_dir=os.path.join(log_dir, 'eval')))

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        log_dir = None

        if state.is_hyper_param_search:
            trial_name = state.trial_name
            if trial_name is not None:
                log_dir = os.path.join(args.logging_dir, trial_name)

        if self.tb_writers is None:
            self._init_summary_writer(args, log_dir)

        for k, tbw in self.tb_writers.items():
            tbw.add_text("args", args.to_json_string())
            if "model" in kwargs:
                model = kwargs["model"]
                if hasattr(model, "config") and model.config is not None:
                    model_config_json = model.config.to_json_string()
                    tbw.add_text("model_config", model_config_json)
            # Version of TensorBoard coming from tensorboardX does not have this method.
            if hasattr(tbw, "add_hparams"):
                tbw.add_hparams(args.to_sanitized_dict(), metric_dict={})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if self.tb_writers is None:
            self._init_summary_writer(args)

        for tbk, tbw in self.tb_writers.items():
            logs_new = custom_rewrite_logs(logs, mode=tbk)
            for k, v in logs_new.items():
                if isinstance(v, (int, float)):
                    tbw.add_scalar(k, v, state.global_step)
                else:
                    logger.warning(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            tbw.flush()

    def on_train_end(self, args, state, control, **kwargs):
        for tbw in self.tb_writers.values():
            tbw.close()
        self.tb_writers = None

# Train the model
def train_model(model, tokenizer, train_dataset, val_dataset):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch_size = 8
    args = TrainingArguments(
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-3,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        warmup_ratio=0.5,
        weight_decay=0.0001,
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        output_dir="./tmp",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[CombinedTensorBoardCallback]
    )
    start_time = time.time()
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model training time: {elapsed_time} seconds")
    return trainer

# Predict using the model
def predict_model(trainer, dataset):
    predictions = trainer.predict(dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = [data_point["labels"] for data_point in dataset]
    return true_labels, predicted_labels

# Plot ROC curve
def plot_roc_curve(true_labels, predicted_scores):
    fpr, tpr, _ = roc_curve(true_labels, predicted_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.xlim(-0.05, 1.05)
    plt.show()

def kfold_cross_validation(model, tokenizer, train_data, val_data, k=5):
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data["data"], train_data["label"]), 1):
        print(f"Fold {fold}")

        train_data_fold = train_data.iloc[train_idx]
        val_data_fold = val_data.iloc[val_idx]

        # Tokenize train and validation data
        X_train_tokenized = tokenize_data(tokenizer, train_data_fold["data"])
        X_val_tokenized = tokenize_data(tokenizer, val_data_fold["data"])

        train_dataset = Dataset(X_train_tokenized, train_data_fold["label"])
        val_dataset = Dataset(X_val_tokenized, val_data_fold["label"])

        # Train model
        trainer = train_model(model, tokenizer, train_dataset, val_dataset)

        # Predict using the trained model
        true_labels, predicted_labels = predict_model(trainer, val_dataset)

        # Compute evaluation metrics
        val_metrics = compute_metrics(true_labels, predicted_labels)
        metrics_list.append(val_metrics)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    return metrics_list

# Fold function
def fold():
    # Load data
    poz = load_data(poz_path)
    non = load_data(non_path)
    val_poz = load_data(val_poz_path)
    val_non = load_data(val_non_path)

    # Concatenate data
    train_data = pd.concat([poz, non])
    val_data = pd.concat([val_poz, val_non])

    # Shuffle data
    train_data = shuffle(train_data)
    val_data = shuffle(val_data)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    # Perform k-fold cross-validation
    metrics_list = kfold_cross_validation(model, tokenizer, train_data, val_data, k=5)
    
    # Print average metrics across folds
    avg_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0]}
    print("Average Metrics Across Folds:")
    print(avg_metrics)

    # Plot ROC curve using the last fold's true labels and predicted scores
    last_fold_true_labels = []
    last_fold_predicted_scores = []
    for fold_metrics in metrics_list:
        last_fold_true_labels.extend(fold_metrics['true_labels'])
        last_fold_predicted_scores.extend(fold_metrics['predicted_scores'])

    plot_roc_curve(last_fold_true_labels, last_fold_predicted_scores)

# Main function
def main():
    # Load data
    poz = load_data(poz_path)
    non = load_data(non_path)
    val_poz = load_data(val_poz_path)
    val_non = load_data(val_non_path)

    # Concatenate data
    train_data = pd.concat([poz, non])
    val_data = pd.concat([val_poz, val_non])

    # Shuffle data
    train_data = shuffle(train_data)
    val_data = shuffle(val_data)

    # Extract features and labels
    X_train = list(train_data["data"])
    X_val = list(val_data["data"])
    y_train = list(train_data["label"])
    y_val = list(val_data["label"])

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)

    # Tokenize data
    X_train_tokenized = tokenize_data(tokenizer, X_train)
    X_val_tokenized = tokenize_data(tokenizer, X_val)

    # Create datasets
    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    # Train the model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)

    # Predict using the trained model
    true_labels, predicted_labels = predict_model(trainer, val_dataset)

    # Print confusion matrix and classification report
    confusion_matrix_result = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(confusion_matrix_result)
    report = classification_report(true_labels, predicted_labels)
    print("Classification Report:")
    print(report)

    # Plot ROC curve
    predicted_scores_tensor = torch.from_numpy(trainer.predict(val_dataset).predictions)
    softmax_scores = torch.softmax(predicted_scores_tensor, dim=1)
    predicted_scores = softmax_scores[:, 1].tolist()
    plot_roc_curve(true_labels, predicted_scores)


if __name__ == "__main__":
    #main()
    fold()
