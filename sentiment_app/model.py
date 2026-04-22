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
        self.classifier = nn.Linear(self.hidden_size * 4, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        last_four_layers = outputs.hidden_states[-4:]
        cls_from_each = [layer[:, 0, :] for layer in last_four_layers]
        combined_output = torch.cat(cls_from_each, dim=-1)

        pooled_output = self.dropout(combined_output)
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


def create_model(model_name, num_classes):
    base_bert_model = AutoModel.from_pretrained(model_name)
    return BERTForSentimentAnalysisCustom(base_bert_model, num_classes=num_classes)
