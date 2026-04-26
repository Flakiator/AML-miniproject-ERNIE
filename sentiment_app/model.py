import torch
import torch.nn as nn
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

        ### Try freezing of weights. Try to freeze all 12 then try to freeze first 8 haha

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


class GloveBaselineForSentimentAnalysisCustom(nn.Module):
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
    dropout=0.2,
    freeze_embeddings=True,
):
    return GloveBaselineForSentimentAnalysisCustom(
        embedding_matrix=embedding_matrix,
        num_classes=num_classes,
        padding_idx=padding_idx,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )
