import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BERTForSentimentAnalysisCustom(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.bert = base_model
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False, #True
        )

        # last_four_layers = outputs.hidden_states[-4:]
        # cls_from_each = [layer[:, 0, :] for layer in last_four_layers]
        # combined_output = torch.cat(cls_from_each, dim=-1)

        # pooled_output = self.dropout(combined_output)

        last_hidden_state = outputs[0]
        cls_token = last_hidden_state[:, 0, :]

        pooled_output = self.dropout(cls_token)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_classes), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------------
# GloVe BiLSTM — lightweight alternative to the transformer encoder
# ---------------------------------------------------------------------------

class GloveBaselineForSentimentAnalysisCustom(nn.Module):
    """
    Shallow BiLSTM with additive attention over GloVe embeddings.

    Architecture:
        GloVe token embedding (optionally frozen)
        → Dropout
        → BiLSTM  (single layer, hidden_dim per direction)
        → Additive attention pooling  (learns which tokens matter)
        → Dropout → Linear classifier

    Why this over mean-pooling:
        - BiLSTM reads left-to-right and right-to-left, so "not bad" and
          "bad" produce different hidden states — mean-pooling cannot do this.
        - Additive attention learns to focus on sentiment-bearing tokens
          (adjectives, negations) and ignore filler words.

    Why this over the transformer:
        - ~2M parameters vs ~14M — trains in minutes on CPU.
        - LSTMs are sequential so the batch size can stay large without
          the O(seq²) memory cost of self-attention.
    """

    def __init__(
        self,
        embedding_matrix,
        num_classes: int,
        padding_idx: int = 0,
        dropout: float = 0.3,
        freeze_embeddings: bool = True,
        hidden_dim: int = 128,   # per-direction; BiLSTM output = hidden_dim * 2
        # Unused kwargs accepted so the factory signature stays compatible
        # with any callers that still pass transformer-specific keys.
        **kwargs,
    ):
        super().__init__()
        embedding_matrix = torch.as_tensor(embedding_matrix, dtype=torch.float32)
        embed_dim = embedding_matrix.shape[1]

        self.num_classes = num_classes
        self.padding_idx = padding_idx
        self.hidden_size = hidden_dim * 2   # exposed for compatibility

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )
        self.embed_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Additive (Bahdanau-style) attention: a single linear layer that
        # scores each token's hidden state, then softmax over real tokens.
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is None:
            attention_mask = (input_ids != self.padding_idx).long()

        # Embed
        x = self.embed_dropout(self.embedding(input_ids))   # (B, S, E)

        # Pack so the LSTM ignores padding — faster and avoids polluting
        # hidden states with pad-token gradients.
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # hidden: (B, S, hidden_dim * 2)

        # Additive attention — mask out padding before softmax
        scores = self.attention(hidden).squeeze(-1)          # (B, S)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)     # (B, S, 1)

        # Weighted sum of hidden states
        context = (hidden * weights).sum(dim=1)              # (B, hidden_dim*2)

        logits = self.classifier(self.dropout(context))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(hidden,),
            attentions=weights.squeeze(-1),
        )


# ---------------------------------------------------------------------------
# GloVe Simple — mean-pooling baseline (fastest, CPU-friendly)
# ---------------------------------------------------------------------------

class GloveBaselineForSentimentAnalysisCustomSimple(nn.Module):
    """
    Mean-pooling over frozen GloVe embeddings followed by a linear classifier.
    No sequential processing — every token is treated independently.
    Use this as the lowest-cost baseline before trying the BiLSTM.
    """

    def __init__(
        self,
        embedding_matrix,
        num_classes,
        padding_idx=0,
        dropout=0.2,
        freeze_embeddings=True,
    ):
        super().__init__()
        embedding_matrix = torch.as_tensor(embedding_matrix, dtype=torch.float32)

        self.num_classes = num_classes
        self.hidden_size = embedding_matrix.shape[1]
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        embedded_tokens = self.embedding(input_ids)

        if attention_mask is None:
            attention_mask = (input_ids != self.padding_idx).long()

        mask = attention_mask.unsqueeze(-1).type_as(embedded_tokens)
        masked_embeddings = embedded_tokens * mask

        token_counts = mask.sum(dim=1).clamp(min=1.0)
        pooled_output = masked_embeddings.sum(dim=1) / token_counts
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_function = nn.CrossEntropyLoss()
            loss = loss_function(logits.view(-1, self.num_classes), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(embedded_tokens,),
            attentions=None,
        )


def create_model(model_name, num_classes):
    base_bert_model = AutoModel.from_pretrained(model_name)
    return BERTForSentimentAnalysisCustom(base_bert_model, num_classes=num_classes)


def create_glove_baseline_model(
    embedding_matrix,
    num_classes,
    padding_idx=0,
    dropout=0.3,
    freeze_embeddings=True,
    hidden_dim=128,
    **kwargs,   # absorb any leftover transformer kwargs without breaking callers
):
    return GloveBaselineForSentimentAnalysisCustom(
        embedding_matrix=embedding_matrix,
        num_classes=num_classes,
        padding_idx=padding_idx,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
        hidden_dim=hidden_dim,
    )


def create_glove_simple_model(
    embedding_matrix,
    num_classes,
    padding_idx=0,
    dropout=0.2,
    freeze_embeddings=True,
    **kwargs,
):
    return GloveBaselineForSentimentAnalysisCustomSimple(
        embedding_matrix=embedding_matrix,
        num_classes=num_classes,
        padding_idx=padding_idx,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )