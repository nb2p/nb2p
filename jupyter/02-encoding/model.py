import torch
from torch import nn


class Model(nn.Module):
    def __init__(
        self,
        encoder,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super(Model, self).__init__()
        self.encoder = encoder.to(device)

    def forward(
        self,
        code_inputs: torch.Tensor,
        attn_mask: torch.Tensor,
        position_idx: torch.Tensor,
    ):

        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)

        nodes_mask_ext = nodes_mask[:, :, None]

        inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)

        nodes_to_token_mask = nodes_mask_ext & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = (
            nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        )

        avg_embeddings = torch.einsum(
            "abc,acd->abd", nodes_to_token_mask, inputs_embeddings
        )
        inputs_embeddings = (
            inputs_embeddings * ~nodes_mask_ext + avg_embeddings * nodes_mask_ext
        )
        return self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attn_mask,
            position_ids=position_idx,
        )[1]
