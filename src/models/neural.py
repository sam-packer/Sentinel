"""
Deep learning model: Fine-tuned BERTweet for tweet classification.

BERTweet (vinai/bertweet-base) is a RoBERTa-base model pre-trained
on 850M English tweets. We fine-tune all layers with a classification
head for binary exaggerated/accurate prediction.

Hyperparameters are tuned with Optuna using stratified 3-fold CV
and macro F1 as the objective — same evaluation framework as the
classical model for fair comparison.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import optuna
try:
    import torch
    import transformers
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    raise ImportError(
        "PyTorch and transformers are required for the neural model. "
        "Install with: uv sync --extra neural"
    ) from exc
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from . import BaseModel

logger = logging.getLogger("sentinel.models.neural")

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Suppress noisy HuggingFace/transformers logging during training
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

# Reproducibility
SEED = 42
N_SPLITS = 3
CV_SEED = 42
MODEL_NAME = "vinai/bertweet-base"
MAX_LENGTH = 128

# Label mapping (alphabetical, consistent with classical model)
LABEL2ID = {"accurate": 0, "exaggerated": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _seed_everything(seed: int = SEED) -> None:
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_device() -> torch.device:
    """Detect best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TweetDataset(Dataset):
    """Tokenized tweets for PyTorch DataLoader."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = MAX_LENGTH):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def _compute_class_weights(labels: list[int], device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss."""
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_fold(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    params: dict,
    device: torch.device,
    tokenizer,
) -> float:
    """Train on one CV fold, return validation macro F1."""
    _seed_everything()

    train_ds = TweetDataset(train_texts, train_labels, tokenizer)
    val_ds = TweetDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"], shuffle=False
    )

    # Fresh model for each fold (local_files_only=True to skip HTTP checks)
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=params["dropout"],
        classifier_dropout=params["dropout"],
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config, local_files_only=True
    )
    model.to(device)

    class_weights = _compute_class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    # Linear warmup + decay scheduler
    total_steps = len(train_loader) * params["num_epochs"]
    warmup_steps = int(total_steps * params["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    patience = 2

    for _ in range(params["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            epoch_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        val_f1 = f1_score(all_labels, all_preds, average="macro")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Clean up GPU memory between folds
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_f1


def _tune_neural(
    texts: list[str],
    labels: list[int],
    n_trials: int,
    device: torch.device,
    tokenizer,
) -> dict:
    """Tune BERTweet hyperparameters with Optuna."""

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-4, 0.1, log=True),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "num_epochs": trial.suggest_int("num_epochs", 3, 8),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
        }

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]

            f1 = _train_one_fold(
                train_texts, train_labels, val_texts, val_labels,
                params, device, tokenizer,
            )
            fold_scores.append(f1)

            # Prune unpromising trials early after first fold
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        study_name="neural_tuning",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    logger.info(
        f"Neural tuning complete: best macro F1 = {best.value:.4f} "
        f"(trial {best.number}/{n_trials})"
    )
    logger.info(f"Best params: {best.params}")

    return {
        "best_params": best.params,
        "best_score": round(best.value, 4),
        "n_trials": n_trials,
        "best_trial_number": best.number,
    }


class NeuralModel(BaseModel):
    """Fine-tuned BERTweet for tweet exaggeration classification."""

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device: torch.device | None = None
        self._training_meta: dict = {}

    @property
    def name(self) -> str:
        return "neural"

    def train(
        self,
        texts: list[str],
        labels: list[str],
        *,
        n_trials: int = 50,
        saved_params: dict | None = None,
    ) -> dict:
        self._device = _get_device()
        logger.info(f"Device: {self._device}")

        # Encode string labels to integers
        y = [LABEL2ID[label] for label in labels]

        # Download model and tokenizer once (cached for all trials)
        logger.info(f"Downloading {MODEL_NAME} (if not cached)...")
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Pre-download model weights so folds can use local_files_only=True
        _pre = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        # Diagnostic: verify pretrained weights loaded
        _emb = _pre.roberta.embeddings.word_embeddings.weight[0, :5].tolist()
        logger.info(f"Pretrained embedding sample: {_emb}")
        del _pre
        torch.cuda.empty_cache()
        logger.info("Model cached.")

        # Tune or reuse hyperparameters
        if saved_params is not None:
            tuning = saved_params
            logger.info(f"Using saved params: {tuning['best_params']}")
        else:
            logger.info(
                f"Tuning BERTweet ({n_trials} trials, {N_SPLITS}-fold CV, macro F1)..."
            )
            tuning = _tune_neural(texts, y, n_trials, self._device, self._tokenizer)

        # Train final model on full training set with best params.
        # Hold out 10% for monitoring — NOT for model selection, just to
        # verify the model is learning and apply early stopping.
        logger.info("Training final model with best params on full training set...")
        best_params = tuning["best_params"]

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        # 90/10 split for train/monitor
        n_monitor = max(int(len(texts) * 0.1), 1)
        rng = np.random.default_rng(SEED)
        indices = rng.permutation(len(texts))
        monitor_idx = indices[:n_monitor]
        train_idx = indices[n_monitor:]

        train_texts_final = [texts[i] for i in train_idx]
        train_labels_final = [y[i] for i in train_idx]
        monitor_texts = [texts[i] for i in monitor_idx]
        monitor_labels = [y[i] for i in monitor_idx]

        config = AutoConfig.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            hidden_dropout_prob=best_params["dropout"],
            classifier_dropout=best_params["dropout"],
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=config
        )
        self._model.to(self._device)

        # Verify pretrained weights loaded (not random init)
        sample_weight = self._model.roberta.embeddings.word_embeddings.weight[0, :5]
        logger.info(f"Embedding sample (should NOT be near-zero): {sample_weight.tolist()}")

        train_ds = TweetDataset(train_texts_final, train_labels_final, self._tokenizer)
        monitor_ds = TweetDataset(monitor_texts, monitor_labels, self._tokenizer)
        train_loader = DataLoader(
            train_ds, batch_size=best_params["batch_size"], shuffle=True
        )
        monitor_loader = DataLoader(
            monitor_ds, batch_size=best_params["batch_size"], shuffle=False
        )

        class_weights = _compute_class_weights(train_labels_final, self._device)
        logger.info(f"Class weights: {class_weights.tolist()}")
        logger.info(f"Label distribution: {np.bincount(train_labels_final).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
        )

        total_steps = len(train_loader) * best_params["num_epochs"]
        warmup_steps = int(total_steps * best_params["warmup_ratio"])

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Training with early stopping (same as CV folds)
        best_state = None
        best_monitor_f1 = 0.0
        patience_counter = 0
        patience = 2

        for epoch in range(best_params["num_epochs"]):
            self._model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                batch_labels = batch["labels"].to(self._device)

                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = criterion(outputs.logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # Monitor validation
            self._model.eval()
            all_preds = []
            all_labels_monitor = []
            with torch.no_grad():
                for batch in monitor_loader:
                    input_ids = batch["input_ids"].to(self._device)
                    attention_mask = batch["attention_mask"].to(self._device)
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels_monitor.extend(batch["labels"].numpy())

            monitor_f1 = f1_score(all_labels_monitor, all_preds, average="macro")
            logger.info(
                f"Epoch {epoch + 1}/{best_params['num_epochs']}: "
                f"loss={avg_loss:.4f}, monitor_f1={monitor_f1:.4f}"
            )

            if monitor_f1 > best_monitor_f1:
                best_monitor_f1 = monitor_f1
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self._device)
            logger.info(f"Restored best model (monitor F1={best_monitor_f1:.4f})")

        # Compute train accuracy on full training set
        train_preds = self.predict_batch(texts)
        train_acc = sum(p == l for p, l in zip(train_preds, labels)) / len(labels)
        logger.info(f"Train accuracy: {train_acc:.3f}")

        self._training_meta = {
            "train_size": len(texts),
            "model_name": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "tuning": {
                "n_trials": n_trials,
                "cv_folds": N_SPLITS,
                "objective": "f1_macro",
                **tuning,
            },
            "train_accuracy": round(train_acc, 4),
        }
        return self._training_meta

    def predict(self, text: str) -> str:
        if self._model is None:
            raise RuntimeError(
                "Model has not been trained. Run 'uv run train neural' first."
            )
        preds = self.predict_batch([text])
        return preds[0]

    def predict_batch(self, texts: list[str]) -> list[str]:
        if self._model is None:
            raise RuntimeError(
                "Model has not been trained. Run 'uv run train neural' first."
            )

        self._model.eval()
        encodings = self._tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        all_preds = []
        batch_size = 128

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                input_ids = encodings["input_ids"][i:i + batch_size].to(self._device)
                attention_mask = encodings["attention_mask"][i:i + batch_size].to(self._device)

                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(ID2LABEL[p] for p in preds)

        return all_preds

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer using HuggingFace's native format
        model_dir = directory / "model"
        tokenizer_dir = directory / "tokenizer"
        self._model.save_pretrained(str(model_dir))
        self._tokenizer.save_pretrained(str(tokenizer_dir))

        # Save metadata
        meta = {
            "classes": list(LABEL2ID.keys()),
            "training_meta": self._training_meta,
        }
        with open(directory / "model.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save best hyperparameters separately for --skip-tuning
        if "tuning" in self._training_meta:
            with open(directory / "best_params.json", "w") as f:
                json.dump(self._training_meta["tuning"], f, indent=2)

        logger.info(f"Saved to {directory}")

    def load(self, directory: Path) -> None:
        self._device = _get_device()

        model_dir = directory / "model"
        tokenizer_dir = directory / "tokenizer"

        self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir)
        )
        self._model.to(self._device)
        self._model.eval()

        with open(directory / "model.json") as f:
            meta = json.load(f)
        self._training_meta = meta.get("training_meta", {})

        logger.info(f"Loaded from {directory} (device: {self._device})")
