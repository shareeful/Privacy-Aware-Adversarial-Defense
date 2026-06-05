import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple


class DeBERTaClassifier(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-large", num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.config.output_attentions = True
        self.config.output_hidden_states = True

        self.deberta = AutoModel.from_pretrained(model_name, config=self.config)

        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Dict:
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=True,
            output_hidden_states=True,
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        probabilities = self.sigmoid(logits).squeeze(-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "cls_embedding": cls_embedding,
        }

        if return_attentions:
            result["attentions"] = outputs.attentions
            result["hidden_states"] = outputs.hidden_states

        if labels is not None:
            if class_weights is not None:
                sample_weights = class_weights[labels]
                bce_loss = nn.BCELoss(reduction="none")
                loss = (bce_loss(probabilities, labels.float()) * sample_weights).mean()
            else:
                loss_fct = nn.BCELoss()
                loss = loss_fct(probabilities, labels.float())
            result["loss"] = loss

        return result

    def get_cls_embeddings(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        return outputs.last_hidden_state[:, 0, :]

    def get_final_layer_attentions(
        self, input_ids, attention_mask, token_type_ids=None
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.deberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True,
            )
        return outputs.attentions[-1]

    def predict(
        self, input_ids, attention_mask, token_type_ids=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            result = self.forward(input_ids, attention_mask, token_type_ids)
        probs = result["probabilities"]
        preds = (probs > 0.5).long()
        return preds, probs
