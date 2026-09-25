import copy
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from modules.config import (
    HIDDEN_DIM, MAX_NEURONS,
    EMBED_MODEL_NAME, EMBED_DIM, PROJ_DIM,
    EMA_DECAY, DEVICE, MODEL_INPUT_DIM,
    VOCAB_SIZE, D_MODEL, NHEAD, N_LAYERS, DIM_FF, DROPOUT,
)


class NeuralBlock(nn.Module):
    """Single feedforward block - easy to stack or swap later."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.act    = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.linear(x)))


class NumericTokenizer(nn.Module):
    """
    Encodes a float vector into a sequence of soft vocabulary
    tokens so the transformer sees discrete-ish symbols rather
    than raw scalars.

    Each input dimension is quantised into VOCAB_SIZE bins via
    a learned soft-max over a set of prototype values, then
    looked up in a standard nn.Embedding table.  The result is
    a (B, D, D_MODEL) sequence - one token per input feature -
    that the TransformerEncoder can attend over.
    """

    def __init__(
        self,
        in_dim:     int = MODEL_INPUT_DIM,
        vocab_size: int = VOCAB_SIZE,
        d_model:    int = D_MODEL,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.prototypes = nn.Parameter(
            torch.linspace(-3.0, 3.0, vocab_size)
        )
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, D) or (D,)
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        # Soft assignment: distance of each feature to each prototype
        # -> (B, D, V)
        dists    = (x.unsqueeze(-1) - self.prototypes) ** 2
        weights  = torch.softmax(-dists, dim=-1)

        # Weighted sum over embedding rows -> (B, D, d_model)
        emb = weights @ self.embedding.weight

        return emb.squeeze(0) if was_1d else emb


class EmbeddingProjector(nn.Module):
    """
    Projects a frozen pretrained sentence-transformer
    embedding down to PROJ_DIM so it can be concatenated
    with the raw input without drowning it out.
    The ST encoder itself stays frozen - we only train
    the tiny linear projection on top.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        proj_dim:  int = PROJ_DIM,
    ) -> None:
        super().__init__()
        self._st = SentenceTransformer(EMBED_MODEL_NAME)
        for p in self._st.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(embed_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)
        # ST encode is the most expensive call in the model's forward
        # path and the same embed_ctx string is reused ~15x per epoch
        # (MC samples, MAML inner steps, verification). Cache the last
        # raw ST output keyed on the string so re-encoding only fires
        # when the string actually changes.
        self._last_text: str | None     = None
        self._last_raw:  torch.Tensor | None = None

    def encode_text(self, text: str) -> torch.Tensor:
        if text != self._last_text or self._last_raw is None:
            self._last_raw  = self._st.encode(
                text,
                convert_to_tensor=True,
                device=DEVICE,
            )
            self._last_text = text
        # ST encode runs under inference_mode internally so the
        # returned tensor can't enter the autograd graph directly
        # - clone() pulls it out into a normal tracked tensor
        return self.norm(self.proj(self._last_raw.clone()))

    def forward(self, text: str) -> torch.Tensor:
        return self.encode_text(text)

    def __deepcopy__(self, memo):
        # MAML deepcopies LarkosModel every inner update; the frozen
        # 22M-param MiniLM should be SHARED with the original, not
        # copied. Copying it is most of the per-epoch deepcopy cost.
        new = self.__class__.__new__(self.__class__)
        nn.Module.__init__(new)
        new._st        = self._st
        new.proj       = copy.deepcopy(self.proj, memo)
        new.norm       = copy.deepcopy(self.norm, memo)
        new._last_text = self._last_text
        new._last_raw  = self._last_raw
        return new


class LarkosModel(nn.Module):
    """
    Transformer-based model - NeuralBlock MLP replaced with a
    TransformerEncoder so feature interactions are learned via
    attention rather than fixed layer order.

    Numeric input is first tokenised by NumericTokenizer into a
    (D, d_model) sequence; the context embedding (if supplied)
    is projected to d_model and prepended as a CLS-style token.
    The encoder output at the CLS position drives the head.

    Input  : MODEL_INPUT_DIM floats (+PROJ_DIM if embed_ctx given)
    Output : MAX_NEURONS floats (mirrors neuron prediction shape)
    """

    def __init__(
        self,
        input_dim:  int  = MODEL_INPUT_DIM,
        hidden_dim: int  = HIDDEN_DIM,
        output_dim: int  = MAX_NEURONS,
        use_embed:  bool = True,
        d_model:    int  = D_MODEL,
        nhead:      int  = NHEAD,
        n_layers:   int  = N_LAYERS,
        dim_ff:     int  = DIM_FF,
        dropout:    float = DROPOUT,
    ) -> None:
        super().__init__()
        self.use_embed  = use_embed
        self.embedder   = EmbeddingProjector() if use_embed else None
        self.output_dim = output_dim
        self.d_model    = d_model

        self.tokenizer  = NumericTokenizer(input_dim, VOCAB_SIZE, d_model)

        # Project the ST context embedding to d_model so it can act
        # as a CLS token prepended to the numeric token sequence
        if use_embed:
            self.ctx_proj = nn.Linear(PROJ_DIM, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward = dim_ff,
            dropout        = dropout,
            batch_first    = True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Linear(d_model, output_dim)

    def forward(
        self,
        x:         torch.Tensor,
        embed_ctx: str | None = None,
    ) -> torch.Tensor:
        was_1d = x.dim() == 1
        if was_1d:
            x = x.unsqueeze(0)

        tokens = self.tokenizer(x)

        if self.use_embed and embed_ctx is not None:
            ctx_raw = self.embedder(embed_ctx)
            # (d_model,) -> (B, 1, d_model) CLS prepend
            cls = self.ctx_proj(ctx_raw).unsqueeze(0).unsqueeze(0)
            cls = cls.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        # (B, seq, d_model) -> use CLS position (index 0) as summary
        enc_out = self.encoder(tokens)
        summary = enc_out[:, 0, :]

        out = self.head(summary)
        return out.squeeze(0) if was_1d else out


class EMAWrapper:
    """
    Exponential moving average over model weights.
    Call .update() after each optimizer step, then
    .apply_shadow() / .restore() around inference to
    get the smoother shadow weights without touching
    the training graph.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = EMA_DECAY,
    ) -> None:
        self.decay  = decay
        self.shadow = copy.deepcopy(model.state_dict())

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = (
                    self.decay * self.shadow[k]
                    + (1.0 - self.decay) * v
                )
            else:
                self.shadow[k] = v

    def apply_shadow(self, model: nn.Module) -> None:
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self._backup)
