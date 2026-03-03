"""
Multi-input fusion neural model for Sentinel.

Architecture:
  Tweet text -> ProsusAI/finbert -> [CLS] (768-dim)
  Headlines  -> all-MiniLM-L6-v2 (frozen) -> 384-dim embedding
  Scalar features -> 20-dim
  Concatenate -> Linear(1172, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 3)

Training:
  AdamW with differential LR (2e-5 for BERT, 1e-3 for head),
  linear warmup 10%, early stopping patience=3 on val macro F1.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger("sentinel.models.neural")

LABEL_TO_IDX = {"exaggerated": 0, "accurate": 1, "understated": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class ClaimDataset(Dataset):
    """Dataset for defense stock claims."""

    def __init__(
        self,
        texts: list[str],
        headline_embeddings: np.ndarray,
        scalar_features: np.ndarray,
        labels: list[str] | None = None,
        tokenizer=None,
        max_length: int = 128,
    ):
        self.texts = texts
        self.headline_embeddings = headline_embeddings
        self.scalar_features = scalar_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "headline_embedding": torch.tensor(
                self.headline_embeddings[idx], dtype=torch.float32
            ),
            "scalar_features": torch.tensor(
                self.scalar_features[idx], dtype=torch.float32
            ),
        }

        if self.labels is not None:
            item["label"] = torch.tensor(LABEL_TO_IDX[self.labels[idx]], dtype=torch.long)

        return item


class FusionHead(nn.Module):
    """Classification head that fuses BERT, headline, and scalar features."""

    def __init__(self, bert_dim: int = 768, headline_dim: int = 384, scalar_dim: int = 20):
        super().__init__()
        total_dim = bert_dim + headline_dim + scalar_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )

    def forward(self, bert_output, headline_embedding, scalar_features):
        combined = torch.cat([bert_output, headline_embedding, scalar_features], dim=1)
        return self.classifier(combined)


class NeuralClassifier:
    """Multi-input fusion model using FinBERT + MiniLM + scalar features."""

    def __init__(
        self,
        base_model: str = "ProsusAI/finbert",
        fallback_model: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 16,
        lr_bert: float = 2e-5,
        lr_head: float = 1e-3,
        max_epochs: int = 10,
        patience: int = 3,
        scalar_dim: int = 20,
        device: str | None = None,
    ):
        self.base_model_name = base_model
        self.fallback_model_name = fallback_model
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr_bert = lr_bert
        self.lr_head = lr_head
        self.max_epochs = max_epochs
        self.patience = patience
        self.scalar_dim = scalar_dim

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bert_model = None
        self.tokenizer = None
        self.fusion_head = None
        self._loaded = False

    def _load_bert(self):
        """Load BERT model and tokenizer, falling back if needed."""
        if self._loaded:
            return

        from transformers import AutoModel, AutoTokenizer

        try:
            logger.info(f"Loading {self.base_model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.bert_model = AutoModel.from_pretrained(self.base_model_name)
        except Exception as e:
            logger.warning(f"Failed to load {self.base_model_name}: {e}")
            logger.info(f"Falling back to {self.fallback_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name)
            self.bert_model = AutoModel.from_pretrained(self.fallback_model_name)

        bert_dim = self.bert_model.config.hidden_size
        self.fusion_head = FusionHead(
            bert_dim=bert_dim, headline_dim=384, scalar_dim=self.scalar_dim,
        )

        self.bert_model.to(self.device)
        self.fusion_head.to(self.device)
        self._loaded = True

    def fit(
        self,
        texts: list[str],
        headline_embeddings: np.ndarray,
        scalar_features: np.ndarray,
        labels: list[str],
        val_texts: list[str] | None = None,
        val_headline_embeddings: np.ndarray | None = None,
        val_scalar_features: np.ndarray | None = None,
        val_labels: list[str] | None = None,
    ) -> dict:
        """Train the fusion model.

        Args:
            texts: Training tweet texts.
            headline_embeddings: (n, 384) headline embeddings.
            scalar_features: (n, scalar_dim) engineered features.
            labels: Training labels.
            val_*: Optional validation set for early stopping.

        Returns:
            Dict with training metrics (epochs, best_val_f1, etc.)
        """
        self._load_bert()
        from sklearn.metrics import f1_score

        train_dataset = ClaimDataset(
            texts, headline_embeddings, scalar_features, labels,
            self.tokenizer, self.max_length,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )

        # Differential learning rates
        optimizer = torch.optim.AdamW([
            {"params": self.bert_model.parameters(), "lr": self.lr_bert},
            {"params": self.fusion_head.parameters(), "lr": self.lr_head},
        ])

        # Linear warmup
        total_steps = len(train_loader) * self.max_epochs
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = 0.0
        patience_counter = 0
        metrics = {"epochs_trained": 0, "best_val_f1": 0.0}

        for epoch in range(self.max_epochs):
            self.bert_model.train()
            self.fusion_head.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                headline_emb = batch["headline_embedding"].to(self.device)
                scalar_feat = batch["scalar_features"].to(self.device)
                label_ids = batch["label"].to(self.device)

                outputs = self.bert_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

                logits = self.fusion_head(cls_output, headline_emb, scalar_feat)
                loss = criterion(logits, label_ids)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs}, loss={avg_loss:.4f}")

            # Validation
            if val_texts is not None and val_labels is not None:
                val_preds = self._predict_internal(
                    val_texts, val_headline_embeddings, val_scalar_features,
                )
                val_f1 = f1_score(val_labels, val_preds, average="macro")
                logger.info(f"  val macro F1={val_f1:.4f}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

            metrics["epochs_trained"] = epoch + 1
            metrics["best_val_f1"] = best_val_f1

        return metrics

    def predict(
        self,
        texts: list[str],
        headline_embeddings: np.ndarray,
        scalar_features: np.ndarray,
    ) -> list[str]:
        """Predict labels.

        Args:
            texts: Tweet texts.
            headline_embeddings: (n, 384) headline embeddings.
            scalar_features: (n, scalar_dim) features.

        Returns:
            List of label strings.
        """
        self._load_bert()
        return self._predict_internal(texts, headline_embeddings, scalar_features)

    def predict_proba(
        self,
        texts: list[str],
        headline_embeddings: np.ndarray,
        scalar_features: np.ndarray,
    ) -> np.ndarray:
        """Predict class probabilities.

        Returns:
            (n, 3) array of probabilities [exaggerated, accurate, understated].
        """
        self._load_bert()
        dataset = ClaimDataset(
            texts, headline_embeddings, scalar_features,
            tokenizer=self.tokenizer, max_length=self.max_length,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.bert_model.eval()
        self.fusion_head.eval()
        all_probs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                headline_emb = batch["headline_embedding"].to(self.device)
                scalar_feat = batch["scalar_features"].to(self.device)

                outputs = self.bert_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                cls_output = outputs.last_hidden_state[:, 0, :]
                logits = self.fusion_head(cls_output, headline_emb, scalar_feat)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def _predict_internal(
        self,
        texts: list[str],
        headline_embeddings: np.ndarray,
        scalar_features: np.ndarray,
    ) -> list[str]:
        """Internal prediction method."""
        probs = self.predict_proba(texts, headline_embeddings, scalar_features)
        indices = probs.argmax(axis=1)
        return [IDX_TO_LABEL[i] for i in indices]

    def save(self, directory: str) -> None:
        """Save model weights to directory."""
        if not self._loaded:
            raise RuntimeError("No model to save.")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        self.bert_model.save_pretrained(str(path / "bert"))
        self.tokenizer.save_pretrained(str(path / "bert"))
        torch.save(self.fusion_head.state_dict(), str(path / "fusion_head.pt"))
        logger.info(f"Neural model saved to {directory}")

    def load(self, directory: str) -> "NeuralClassifier":
        """Load model weights from directory."""
        from transformers import AutoModel, AutoTokenizer

        path = Path(directory)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path / "bert"))
        self.bert_model = AutoModel.from_pretrained(str(path / "bert"))

        bert_dim = self.bert_model.config.hidden_size
        self.fusion_head = FusionHead(
            bert_dim=bert_dim, headline_dim=384, scalar_dim=self.scalar_dim,
        )
        self.fusion_head.load_state_dict(
            torch.load(str(path / "fusion_head.pt"), map_location=self.device, weights_only=True)
        )

        self.bert_model.to(self.device)
        self.fusion_head.to(self.device)
        self._loaded = True
        logger.info(f"Neural model loaded from {directory}")
        return self
