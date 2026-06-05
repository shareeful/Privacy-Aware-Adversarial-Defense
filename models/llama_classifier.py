import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Optional, Tuple


class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: int = 32, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.original_weight = None
        self.original_bias = None

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = nn.functional.linear(x, self.original_weight, self.original_bias)
        lora_output = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_output + lora_output * self.scaling


class LLaMAClassifierWithLoRA(nn.Module):
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B",
        num_labels: int = 2,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        self.target_modules = target_modules

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_attentions = True
        self.config.output_hidden_states = True

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.lora_layers = nn.ModuleDict()
        self._inject_lora_adapters(lora_rank, lora_alpha, lora_dropout)

        hidden_size = self.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def _inject_lora_adapters(self, rank: int, alpha: int, dropout: float):
        for name, module in self.backbone.named_modules():
            for target in self.target_modules:
                if name.endswith(target) and isinstance(module, nn.Linear):
                    lora_key = name.replace(".", "_")
                    lora_layer = LoRALinear(
                        module.in_features, module.out_features,
                        rank=rank, alpha=alpha, dropout=dropout
                    )
                    lora_layer.original_weight = module.weight
                    lora_layer.original_bias = module.bias
                    lora_layer.lora_A.requires_grad = True
                    lora_layer.lora_B.requires_grad = True
                    self.lora_layers[lora_key] = lora_layer

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params = []
        for layer in self.lora_layers.values():
            params.extend([layer.lora_A, layer.lora_B])
        params.extend(self.classifier.parameters())
        return params

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
        )

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden = outputs.hidden_states[-1]
        final_token_hidden = last_hidden[
            torch.arange(batch_size, device=input_ids.device), sequence_lengths
        ]

        logits = self.classifier(final_token_hidden.float())
        probabilities = self.sigmoid(logits).squeeze(-1)

        result = {
            "logits": logits,
            "probabilities": probabilities,
            "cls_embedding": final_token_hidden.float(),
        }

        if return_attentions:
            result["attentions"] = outputs.attentions
            result["hidden_states"] = outputs.hidden_states
            result["sequence_lengths"] = sequence_lengths

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
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden = outputs.hidden_states[-1]
        return last_hidden[
            torch.arange(batch_size, device=input_ids.device), sequence_lengths
        ].float()

    def get_final_layer_attentions(
        self, input_ids, attention_mask, token_type_ids=None
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
