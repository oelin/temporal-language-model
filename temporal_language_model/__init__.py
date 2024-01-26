import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat


class TemporalAttention(nn.Module):
    """Temporal attention.

    Example
    -------
    >>> module = TemporalAttention(
    ...     embedding_dimension=256,
    ...     heads=16,
    ... )
    >>> x = torch.randn((1, 10, 256))
    >>> t = torch.randn((1, 256))
    >>> x = module(x, t, mask=None)
    """

    def __init__(self, *, embedding_dimension: int, heads: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_dimension : int
            The embedding dimension.
        heads : int
            The number of attention heads.
        """

        super().__init__()

        self.heads = heads

        self.linear_q = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_k = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_v = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_t = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

        self.linear_o = nn.Linear(
            in_features=embedding_dimension,
            out_features=embedding_dimension,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        t : torch.Tensor
            The time embedding, of shape (batch_size, embedding_dimension).
        mask : torch.Tensor
            The attention mask (e.g. a causal mask).

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        q = rearrange(self.linear_q(x), '... t (h e) -> ... h t e', h=self.heads)
        k = rearrange(self.linear_k(x), '... t (h e) -> ... h t e', h=self.heads)
        v = rearrange(self.linear_v(x), '... t (h e) -> ... h t e', h=self.heads)

        # Batched (Q @ T) @ Q.T.

        t = rearrange(self.linear_t(t), '... (h e) ->  ... h e', h=self.heads)
        t = repeat(t, '... h e -> ... h t e', t=q.size(-2))
        t = torch.einsum('...hte,...hse->...hts', t, q)
        t = torch.einsum('...hts,...hes->...hte', t, q.transpose(-1, -2))

        x = F.scaled_dot_product_attention(t, k, v, attn_mask=mask)
        x = self.linear_o(rearrange(x, '... h t e -> ... t (h e)'))

        return x
