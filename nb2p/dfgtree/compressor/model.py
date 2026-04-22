import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features[:, 0, :]
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        encoder,
        config,
        tokenizer,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super(Model, self).__init__()
        self.encoder = encoder.to(device)
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.query = 0

    def forward(self, code_inputs, attn_mask, position_idx):
        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(code_inputs)
        nodes_to_token_mask = (
            nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        )
        nodes_to_token_mask = (
            nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        )
        avg_embeddings = torch.einsum(
            "abc,acd->abd", nodes_to_token_mask, inputs_embeddings
        )
        inputs_embeddings = (
            inputs_embeddings * (~nodes_mask)[:, :, None]
            + avg_embeddings * nodes_mask[:, :, None]
        )

        outputs = self.encoder.roberta(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
        )[0]
        
        return outputs.mean(axis=1)
