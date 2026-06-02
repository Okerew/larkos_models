import numpy as np
import torch
import torch.nn as nn
from collections import deque
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from modules.backend_state import BackendState
from modules.config import (
    BASE_LR, NUM_REGIONS, COSINE_T_MAX, DEVICE,
    EMOTION_HATE, EMOTION_SURPRISE, EMOTION_LOVE,
    MAX_NEURONS, INPUT_SIZE,
    MEM_WEIGHT_RATIO_BASE, MEM_WEIGHT_RATIO_RANGE,
    FOURIER_ENCODINGS, TEMPORAL_WINDOW, FOURIER_OUT_DIM,
    FUSION_DIM, MC_DROPOUT_T, EXPLORE_THRESHOLD,
    INTERNAL_DIM, FUSE_NHEAD, FUSE_N_LAYER, FUSE_DIM_FF,
    GPT2_HIDDEN, TEXT_MAX_NEW,
    VERIFY_INTERVAL, TARGET_FREEZE_INTERVAL,
    EMOTION_LOG_INTERVAL, N_PREFIX, BAND_M, BAND_Q, BAND_N
)
from modules.model import LarkosModel, EMAWrapper
from modules.strategies import (
    build_neuron_prediction,
    derive_lr, derive_alpha_from_context,
    derive_alpha_from_params, update_optimizer_lr,
    maml_inner_update,
)
from modules.fusion_mechanism.fusion import cognitive_fuse
from modules.logging_utils import (
    log_epoch, log_context, log_history, log_memory,
)

from modules.data_pipeline import TextDataPipeline

from modules.verify import run_verification, LearningPatternTracker

from modules.checkpoint import save_checkpoint, load_checkpoint

def _build_embed_ctx(ctx: dict) -> str:
    """
    Serialises the backend context dict into a short natural-
    language string so the pretrained ST model gets something
    meaningful to embed rather than raw JSON noise.
    """
    nodes = ctx.get("total_nodes",  "?")
    decay = ctx.get("decay_rate",   "?")
    vec   = ctx.get("global_context_vector", [])
    mag   = (
        round(float(sum(abs(v) for v in vec) / max(len(vec), 1)), 4)
        if vec else "?"
    )
    return (
        f"network with {nodes} context nodes, "
        f"decay {decay}, mean activation magnitude {mag}"
    )


def _mc_samples(
    model:            LarkosModel,
    x:                torch.Tensor,
    ctx:              str,
    exploration_rate: float = 0.0,
) -> list[torch.Tensor]:
    """
    Runs T stochastic forward passes to estimate per-output
    uncertainty used by fuse_uncertainty_weighted.
    The model stays in train mode so dropout (if any) fires;
    on a model without dropout all samples are identical and
    the variance collapses to zero -> equal weighting fallback.
    no_grad here because these are variance probes only - we
    don't want their graph nodes polluting the training pass.
    Gaussian noise is injected when exploration is above threshold
    so uncertainty estimates stay meaningful even without dropout.
    """
    model.train()
    with torch.no_grad():
        samples = []
        for _ in range(MC_DROPOUT_T):
            xi = x
            if exploration_rate > EXPLORE_THRESHOLD:
                xi = x + exploration_rate * torch.randn_like(x) * 0.05
            samples.append(model(xi, ctx).detach())
        return samples


def _derive_novelty(
    loss_history: list[float],
    current_loss: float,
) -> float:
    if not loss_history:
        return 1.0
    mean = sum(loss_history) / len(loss_history)
    return float(min(abs(current_loss - mean) / (mean + 1e-8), 1.0))


def _derive_satisfaction(loss_val: float, prev_loss: float) -> float:
    # Satisfaction is high when loss is low and dropping, clamped
    # to [0, 1] so it maps cleanly onto the emotional trigger scale
    improvement = max(prev_loss - loss_val, 0.0)
    base        = max(1.0 - loss_val, 0.0)
    return float(min(base + improvement, 1.0))


def _log_emotional_snapshot(
    epoch:    int,
    emo:      dict,
    aff:      dict,
    mask:     dict,
) -> None:
    cur   = aff.get("current_state", {})
    bonds = aff.get("bonds", [])

    valence   = cur.get("valence",   "?")
    arousal   = cur.get("arousal",   "?")
    stability = cur.get("stability", "?")

    intensities = [
        round(e.get("intensity", 0.0), 4)
        for e in emo.get("emotions", [])
    ]
    bond_summary = [
        f"  bond[{b.get('entity_id')}] "
        f"strength={b.get('attachment_strength', '?'):.4f} "
        f"trust={b.get('trust', '?'):.4f} "
        f"resonance={b.get('emotional_resonance', '?'):.4f}"
        for b in bonds
    ]

    print(f"  [epoch {epoch}] emotional snapshot:")
    print(
        f"    affective — valence={valence:.4f}  "
        f"arousal={arousal:.4f}  stability={stability:.4f}"
    )
    print(f"    intensities (by type): {intensities}")
    print(
        f"    cognitive_impact="
        f"{emo.get('cognitive_impact', '?'):.4f}  "
        f"regulation={emo.get('emotional_regulation', '?'):.4f}"
    )
    print(
        f"    mask_intensity={mask.get('mask_intensity', '?'):.4f}"
    )
    if bond_summary:
        print("    attachment bonds:")
        for line in bond_summary:
            print(line)
    else:
        print("    attachment bonds: none yet")


def _pick_emotion_type(
    loss_val:  float,
    prev_loss: float,
    novelty:   float,
) -> tuple[int, float]:
    # delta > 0 means loss fell (improvement), < 0 means it rose.
    # We key the type on the sign of delta with a neutral deadband
    # so small wiggles do not slam the emotion to an extreme, and
    # the returned strength is bounded well below 1.0 so the C side
    # does not saturate intensity to the [1.0, 1.0, ...] corner the
    # logs kept hitting.
    delta = prev_loss - loss_val

    if novelty > 0.7:
        return EMOTION_SURPRISE, min(novelty * 0.6, 0.6)
    if delta > 0.03:
        return EMOTION_LOVE, min(delta * 3.0, 0.6)
    if delta < -0.03:
        return EMOTION_HATE, min(abs(delta) * 3.0, 0.6)
    # Deadband: nothing notable happened, emit a weak neutral-ish
    # surprise so no single emotion accumulates unopposed.
    return EMOTION_SURPRISE, 0.1


def _derive_reflection_signal(
    reflect: dict,
) -> tuple[float, float, float, float]:
    # serialize_metrics emits *_score keys plus a confabulation flag;
    # there is no direct "drift" field so we derive one. Low
    # consistency means the backend's responses are diverging from
    # prior ones, and a raised confabulation flag is the strongest
    # drift signal the reflection produces, so we floor drift at 0.5
    # whenever it fires. novelty_score and coherence_score are passed
    # through directly as the backend's own independent reads. Missing
    # keys fall back to neutral so a sparse dict never destabilises the
    # values downstream.
    confidence  = float(reflect.get("confidence_score", 0.5))
    consistency = float(reflect.get("consistency_score", 1.0))
    novelty     = float(reflect.get("novelty_score", 0.0))
    coherence   = float(reflect.get("coherence_score", 0.5))
    confab      = bool(reflect.get("potentially_confabulated", False))

    drift = 1.0 - consistency
    if confab:
        drift = max(drift, 0.5)

    confidence = min(max(confidence, 0.0), 1.0)
    drift      = min(max(drift, 0.0), 1.0)
    novelty    = min(max(novelty, 0.0), 1.0)
    coherence  = min(max(coherence, 0.0), 1.0)
    return confidence, drift, novelty, coherence


def fourier_encode(
    x:             torch.Tensor,
    num_encodings: int = FOURIER_ENCODINGS,
) -> torch.Tensor:
    freqs = (
        2.0 ** torch.arange(num_encodings, dtype=torch.float32)
        * torch.pi
    ).to(x.device)
    # Ensure at least 2-D so flatten(-2) always has two dims to work with
    was_1d = x.dim() == 1
    if was_1d:
        x = x.unsqueeze(0)
    # x is (B, D); freqs is (E,) -> broadcast to (B, D, E)
    xf   = x.unsqueeze(-1) * freqs
    sins = torch.sin(xf).flatten(-2)
    coss = torch.cos(xf).flatten(-2)
    out  = torch.cat([sins, coss], dim=-1)
    return out.squeeze(0) if was_1d else out


class _EmbedWeightNet(nn.Module):
    """
    Small MLP that produces a per-epoch gate over the context
    embedding so the model decides how much the embedding matters
    based on the raw input values.
    """
    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _InputCrossAttention(nn.Module):
    """
    Treats the D-dim input as D separate 1-dim tokens and the
    E-dim embedding as a single token; input tokens act as queries,
    the embedding as key/value. Much more expressive than concat.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        heads:     int = 4,
    ) -> None:
        super().__init__()
        # Project each scalar input token to head_dim space
        self.q_proj = nn.Linear(1,         embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.attn   = nn.MultiheadAttention(
            embed_dim, heads, batch_first=True
        )
        self.out_proj = nn.Linear(input_dim * embed_dim, embed_dim)

    def forward(
        self,
        x:     torch.Tensor,
        embed: torch.Tensor,
    ) -> torch.Tensor:
        # x     : (D,)  -> (1, D, 1) -> (1, D, E)  [query tokens]
        # embed : (E,)  -> (1, 1, E)                [key/value]
        q = self.q_proj(x.unsqueeze(0).unsqueeze(-1))
        k = self.k_proj(embed.unsqueeze(0).unsqueeze(0))
        v = self.v_proj(embed.unsqueeze(0).unsqueeze(0))
        out, _ = self.attn(q, k, v)
        return self.out_proj(out.flatten(1)).squeeze(0)


class _OnlineMinMax:
    """
    Tracks per-dimension running min/max via EMA so early outliers
    don't permanently warp the normalization range. Unlike a hard
    min/max tracker, the EMA range slowly forgets extreme values and
    re-centers around the actual distribution the backend produces,
    output is always clamped to [-3, 3] as a final safety net.
    """
    def __init__(self, dim: int, momentum: float = 0.02) -> None:
        self.min      = torch.zeros(dim)
        self.max      = torch.ones(dim)
        self._seen    = torch.zeros(dim, dtype=torch.bool)
        self._mom     = momentum

    def update(self, x: torch.Tensor) -> None:
        x_cpu = x.detach().cpu()
        # Hard-init on first sight, EMA blend after that
        self.min = torch.where(
            ~self._seen, x_cpu,
            self.min + self._mom * (x_cpu - self.min),
        )
        self.max = torch.where(
            ~self._seen, x_cpu,
            self.max + self._mom * (x_cpu - self.max),
        )
        self._seen[:] = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mn = self.min.to(x.device)
        mx = self.max.to(x.device)
        scaled = (x - mn) / (mx - mn + 1e-8) * 2.0 - 1.0
        return scaled.clamp(-3.0, 3.0)


class _FusionTransformerHead(nn.Module):
    """
    Sits on top of cognitive_fuse. The C side writes three distinct
    contiguous bands into the FUSION_DIM vector:

        [0           : BAND_Q)            -> llm query stream
        [BAND_Q      : BAND_Q+BAND_N)     -> neuron stream
        [BAND_Q+BAND_N : FUSION_DIM)      -> memory stream

    We treat those bands as a length-3 token sequence so the encoder's
    self-attention can learn how much each stream should attend to the
    others. 

    Bands are unequal length (22/21/21) but a transformer needs a
    uniform d_model per token, so each band gets its own learned input
    projection into d_model, plus a learned stream-type embedding so
    attention can tell which token is which stream.

    LayerNorm + input dropout + noise injection prevent the encoder
    from memorising biased C-side fusion outputs; these are skipped on
    a frozen-input window (see forward) where the input is identical
    across the whole window and noise would only turn a fixed
    learnable target into an unlearnable one.
    """

    _BAND_Q = BAND_Q
    _BAND_N = BAND_N
    _BAND_M = BAND_M

    def __init__(
        self,
        d_model:    int = FUSION_DIM,
        nhead:      int = FUSE_NHEAD,
        n_layers:   int = FUSE_N_LAYER,
        dim_ff:     int = FUSE_DIM_FF,
        output_dim: int = MAX_NEURONS,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        assert (
            self._BAND_Q + self._BAND_N + self._BAND_M == FUSION_DIM
        ), "band split must tile FUSION_DIM"

        # One learned projection per stream maps its band into d_model.
        # Separate projections (not a shared one) let each stream learn
        # its own mapping since the bands carry different information.
        self.proj_q = nn.Linear(self._BAND_Q, d_model)
        self.proj_n = nn.Linear(self._BAND_N, d_model)
        self.proj_m = nn.Linear(self._BAND_M, d_model)

        # Learned stream-type embedding (3 tokens, one per stream) so
        # the attention can distinguish query / neuron / memory tokens.
        # This is the sequence analogue of positional encoding but
        # keyed on stream identity rather than position.
        self.stream_embed = nn.Parameter(
            torch.zeros(3, d_model)
        )
        nn.init.normal_(self.stream_embed, std=0.02)

        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers
        )
        self.linear_out = nn.Linear(d_model, output_dim)

    def _split_bands(
        self,
        fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # fused : (B, FUSION_DIM) -> three band slices matching fusion_mechanism.c
        q_end = self._BAND_Q
        n_end = self._BAND_Q + self._BAND_N
        return (
            fused[:, :q_end],
            fused[:, q_end:n_end],
            fused[:, n_end:],
        )

    def forward(
        self,
        fused:        torch.Tensor,
        frozen_input: bool = False,
    ) -> torch.Tensor:
        band_q, band_n, band_m = self._split_bands(fused)

        # Each band -> d_model, then stack into a length-3 sequence.
        # tok : (B, 3, d_model)
        tok_q = self.proj_q(band_q)
        tok_n = self.proj_n(band_n)
        tok_m = self.proj_m(band_m)
        tok   = torch.stack([tok_q, tok_n, tok_m], dim=1)

        # Add the learned stream-type embedding so attention knows
        # which token is which stream.
        tok = tok + self.stream_embed.unsqueeze(0)

        tok = self.input_norm(tok)

        # On a frozen-input window the input is identical across the
        # whole window, so dropout + noise just turn a fixed learnable
        # target into a noisy unlearnable one. We skip both there and
        # only regularise when the input is actually moving.
        if self.training and not frozen_input:
            tok = self.input_dropout(tok)
            tok = tok + torch.randn_like(tok) * 0.005

        # Self-attention now runs over 3 real tokens, so it actually
        # mixes the streams instead of collapsing to an MLP.
        enc = self.encoder(tok)

        # Mean-pool the three stream tokens back to one vector. Pooling
        # rather than taking a single token keeps all three streams in
        # the output and lets the encoder decide their balance via the
        # attention weights it learns.
        pooled = enc.mean(dim=1)
        return self.linear_out(pooled)


class _TextCodec:
    """
    Decodes the fused cognitive vector into text as a READOUT only.

      decode : fused numeric vector (MAX_NEURONS dims)
               -> _num_to_prefix -> (N_PREFIX, GPT2_HIDDEN) prefix
               -> used as an N_PREFIX-token soft-prompt prefix
               -> GPT-2 autoregressively continues from it

    The bridge is NOT trained. Seven runs established that the prefix
    cannot carry enough information to steer a frozen GPT-2 from an
    8-to-64-dim cognitive vector the LM loss never fell and the gate
    never opened, while the text objective fought base_loss for fused.
    So _num_to_prefix is now a fixed random projection: the text is a
    pure window into fused, not a learned objective. encode() is kept
    because text_encoding still feeds fused as a state driver via the
    C-side cognitive_fuse injection and the in-graph text_proj term.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token

        self.lm = GPT2LMHeadModel.from_pretrained("gpt2")
        self.lm.eval()
        self.lm.to(device)
        for p in self.lm.parameters():
            p.requires_grad_(False)

        self._num_to_prefix = nn.Linear(
            MAX_NEURONS, N_PREFIX * GPT2_HIDDEN
        ).to(device)
        for p in self._num_to_prefix.parameters():
            p.requires_grad_(False)

    def encode(self, text: str) -> torch.Tensor:
        enc = self.tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )
        ids  = enc.input_ids.to(self.device)
        mask = enc.attention_mask.to(self.device)

        with torch.no_grad():
            hidden = self.lm.transformer(
                ids, attention_mask=mask
            ).last_hidden_state
        return hidden.mean(dim=1).squeeze(0)

    def _prefix_from_numeric(
        self,
        numeric_vec: torch.Tensor,
    ) -> torch.Tensor:
        flat = self._num_to_prefix(numeric_vec)
        return flat.view(N_PREFIX, GPT2_HIDDEN)

    def decode(self, numeric_vec: torch.Tensor) -> str:
        prefix = self._prefix_from_numeric(numeric_vec).unsqueeze(0)

        if not torch.isfinite(prefix).all():
            return ""

        bos_id  = self.tok.bos_token_id
        bos_emb = self.lm.transformer.wte(
            torch.tensor([[bos_id]], device=self.device)
        )
        inputs_embeds = torch.cat([prefix, bos_emb], dim=1)

        attn_mask = torch.ones(
            1, inputs_embeds.shape[1],
            dtype=torch.long,
            device=self.device,
        )
        try:
            with torch.no_grad():
                out_ids = self.lm.generate(
                    inputs_embeds    = inputs_embeds,
                    attention_mask   = attn_mask,
                    max_new_tokens   = TEXT_MAX_NEW,
                    do_sample        = True,
                    temperature      = 0.8,
                    pad_token_id     = self.tok.eos_token_id,
                )
        except (torch.AcceleratorError, RuntimeError):
            return ""

        return self.tok.decode(out_ids[0], skip_special_tokens=True)


class _OnlineMeanStd:
    """
    Tracks per-dimension running mean and std via EMA so the
    fusion transformer input can be re-centered each epoch
    without hard-coding any scale assumptions about what
    cognitive_fuse returns from the C side.

    The std floor is deliberately not tiny: a near-constant fused
    dimension divided by ~0 std blows up to the clamp every step and
    shows up as one index always dominating the transformer input.
    A larger floor keeps low-variance dimensions genuinely quiet.
    """
    def __init__(
        self,
        dim:      int,
        momentum: float = 0.02,
        std_floor: float = 0.1,
    ) -> None:
        self.mean   = torch.zeros(dim)
        self.var    = torch.ones(dim)
        self._seen  = False
        self._mom   = momentum
        self._floor = std_floor

    def update(self, x: torch.Tensor) -> None:
        x_cpu = x.detach().cpu()
        if not self._seen:
            self.mean   = x_cpu.clone()
            self.var    = torch.ones_like(x_cpu)
            self._seen  = True
            return
        delta      = x_cpu - self.mean
        self.mean  = self.mean  + self._mom * delta
        self.var   = (1.0 - self._mom) * (
            self.var + self._mom * delta ** 2
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(x.device)
        std  = self.var.to(x.device).sqrt().clamp(min=self._floor)
        return ((x - mean) / std).clamp(-3.0, 3.0)


class TrainingLoop:
    """
    Wraps one full training run.  The public interface is just .run().

    Internally the per-epoch work is split into three clearly-named
    helpers so the gradient lifecycle is explicit:

        _forward  — all computation that builds tensors we
                    differentiate through; returns a ForwardResult
        _backward — takes a ForwardResult, computes the scalar loss,
                    calls .backward(), clips grads, steps optimizer/
                    scheduler/EMA; returns loss_val float
        _side_effects — everything that fires after the gradient step:
                    C backend triggers, logging, memory, emotional pipeline
    """

    # Small named container so _forward can hand many tensors to
    # _backward without a fragile positional tuple
    class _Fwd:
        __slots__ = (
            "model_pred", "fused", "maml_pred",
            "target", "mc_variance",
            "llm_embed_ca",
        )
        def __init__(
            self,
            model_pred, fused, maml_pred,
            target, mc_variance,
            llm_embed_ca,
        ):
            self.model_pred   = model_pred
            self.fused        = fused
            self.maml_pred    = maml_pred
            self.target       = target
            self.mc_variance  = mc_variance
            self.llm_embed_ca = llm_embed_ca

    def __init__(
        self,
        backend: "BackendState",
        epochs: int = 5,
        initial_alpha: float | None = None,
        data_dir=None,
        resume_from: str | None = None,
        resume_use_ema: bool = False,
    ) -> None:
        self.backend       = backend
        self.epochs        = epochs
        # Caller-supplied alpha used as a starting override on epoch 1;
        # after that, per-epoch context derivation takes over.
        self.initial_alpha = initial_alpha

        neurons      = backend.get_neurons()
        input_tensor = backend.get_input_tensor()
        dyn_params   = backend.get_dynamic_params()
        _ = neurons
        _ = dyn_params

        self.model     = LarkosModel().to(DEVICE)
        self.ema       = EMAWrapper(self.model)

        self.fusion_transformer = _FusionTransformerHead(
            output_dim=self.model.output_dim
        ).to(DEVICE)

        self.embed_weight_net = _EmbedWeightNet(
            INPUT_SIZE, INTERNAL_DIM
        ).to(DEVICE)
        self.cross_attn = _InputCrossAttention(
            INPUT_SIZE, INTERNAL_DIM
        ).to(DEVICE)
        self.online_norm     = _OnlineMinMax(INPUT_SIZE)
        # Tracks the running distribution of cognitive_fuse output
        # so we can re-center it before the fusion transformer sees
        # it prevents C-side bias from getting baked into weights
        self.fused_cog_norm  = _OnlineMeanStd(FUSION_DIM)
        self.text_codec      = _TextCodec(DEVICE)

        self._data_pipeline = (
            TextDataPipeline(data_dir) if data_dir is not None
            else TextDataPipeline()
        )

        self._sample_pool: list[str] = []
        self._sample_pool_idx = 0
        self._sample_pool_size = 8

        # Seed the first sample; updated each epoch via _data_pipeline
        _first = self._data_pipeline.next_sample()
        self.text_encoding = (
            self.text_codec.encode(_first).detach()
        )
        self._current_text_input = _first

        self.text_proj = nn.Linear(
            GPT2_HIDDEN, INTERNAL_DIM
        ).to(DEVICE)

        # Small head that maps the cross-attention embedding down to
        # the neuron-target dim so aux_loss is a real minimisable
        # objective rather than the old broadcast-expand hack which
        # pinned every embedding dim to the 8-dim target and sat at a
        # permanent ~2.0 noise floor dominating the total loss.
        self.aux_proj = nn.Linear(
            INTERNAL_DIM, self.model.output_dim
        ).to(DEVICE)

        # _num_to_prefix is deliberately absent here: the text bridge
        # is a frozen readout, not a trained head.
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters())
            + list(self.fusion_transformer.parameters())
            + list(self.embed_weight_net.parameters())
            + list(self.cross_attn.parameters())
            + list(self.text_proj.parameters())
            + list(self.aux_proj.parameters()),
            lr=BASE_LR,
            weight_decay=1e-4,
            eps=1e-6,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=COSINE_T_MAX, eta_min=1e-5
        )

        # SmoothL1Loss for prediction losses (replaces Cosine+L1 which
        # fought itself).  beta=0.1 gives L2 near zero, L1 far from zero.
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.1)
        # MSELoss for the MAML inner loop where same-space vectors are
        # compared and MSE scale is natural and well-behaved.
        self.criterion = nn.MSELoss()

        self._pattern_tracker  = LearningPatternTracker()
        self._mc_blend_ema: float = 0.5

        # Smoothed fusion_transformer grad norm used as the balance
        # reference so the rescale doesn't react to per-epoch pulses
        self._ft_norm_ema: float = 0.0

        self.loss_history: list[float]          = []
        self.prev_loss:    float                = 0.0
        self.input_history: deque[torch.Tensor] = deque(
            maxlen=TEMPORAL_WINDOW
        )

        # Cached target for frozen training windows populated on
        # first epoch and refreshed every TARGET_FREEZE_INTERVAL
        self._cached_target: np.ndarray | None = None
        self._target_epoch: int = 0

        # The transformer input must be frozen on the SAME schedule as
        # the target. If the input moves while the target is pinned the
        # transformer chases a moving input toward a fixed point, which
        # is the "loss that makes no sense" behaviour. We cache the raw
        # C-side fused vector and reuse it across the freeze window.
        self._cached_fused_cog: torch.Tensor | None = None

        # Below this loss we stop stepping a frozen (input, target) pair
        # so we don't memorise it into a grad pulse. Tuned to sit just
        # under the typical converged base/pred loss seen in logs.
        self._frozen_skip_floor: float = 0.15

        # Optionally pick up an existing checkpoint and keep training
        # from it. load_checkpoint restores onto the already-built loop,
        # so it has to run last once all the modules it overwrites exist.
        # use_ema stays off by default so the live trained weights carry
        # on rather than being clobbered by their EMA shadow, which is
        # the inference behaviour, not the resume one.
        if resume_from is not None:
            result = load_checkpoint(
                self, resume_from, use_ema=resume_use_ema
            )
            print(f"  resumed from checkpoint : {result}")

    def _vec_loss(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.smooth_l1(pred, target)

    def _forward(
        self,
        x_temporal:   torch.Tensor,
        x_norm:       torch.Tensor,
        embed_ctx:    str,
        neurons:      dict,
        neuron_pred:  torch.Tensor,
        mem_state:    dict,
        default_weights: torch.Tensor,
        mem_weight_ratio: float,
        alpha:        float,
        exploration_rate: float,
        epoch:        int,
    ) -> "_Fwd":
        """
        Runs the full differentiable forward pass.  Every tensor built
        here that feeds into the loss keeps its grad_fn intact nothing
        is detached or wrapped in no_grad inside this method.
        """
        # MC probes are variance estimates only we do NOT want their
        # graph polluting the training pass so no_grad is correct there
        mc_samples = _mc_samples(
            self.model,
            x_temporal.unsqueeze(0),
            embed_ctx,
            exploration_rate,
        )
        mc_stack    = torch.stack(mc_samples, dim=0)
        mc_variance = mc_stack.var(dim=0).mean().item()

        # Primary model prediction grad ON, used in loss directly
        # so cross_attn / embed_weight_net / text_proj all receive
        # gradients through this tensor
        model_pred = self.model(x_temporal.unsqueeze(0), embed_ctx)

        # Sanity-check; reset and retry on NaN/inf
        if not torch.isfinite(model_pred).all():
            for layer in self.model.modules():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
            model_pred = self.model(
                x_temporal.unsqueeze(0), embed_ctx
            )

        # Derive the embed query from model_pred so the gradient
        # path from fusion back through cross_attn and embed_weight_net
        # stays connected we only detach before handing off to the
        # C-side cognitive_fuse which cannot carry a grad_fn anyway
        llm_embed_raw = model_pred.squeeze(0)
        llm_embed     = llm_embed_raw[:INTERNAL_DIM]

        # Input-dependent gate and cross-attention both stay in graph
        embed_gate   = self.embed_weight_net(x_norm)
        llm_embed_g  = llm_embed * embed_gate
        llm_embed_ca = self.cross_attn(x_norm, llm_embed_g)

        text_proj_out = self.text_proj(
            self.text_encoding.to(DEVICE)
        )
        # text_proj receives gradients here; text_encoding itself
        # stays frozen (detached at construction time)
        llm_embed_ca = (
            llm_embed_ca
            + text_proj_out[:llm_embed_ca.shape[-1]]
        )

        # C-side fusion cannot carry grad_fn detach the query here.
        # llm_embed_ca is returned in _Fwd so _backward can build an
        # aux loss that keeps cross_attn / embed_weight_net / text_proj
        # in the gradient graph without touching cognitive_fuse. The
        # sentence still drives fused via the text_embed injection,
        # which is the cognitive driver.
        fused_cog_raw = cognitive_fuse(
            llm_embed        = llm_embed_ca.detach(),
            neurons          = neurons,
            mem_state        = mem_state,
            default_weights  = default_weights,
            mem_weight_ratio = mem_weight_ratio,
            context_factor   = alpha,
            text_embed       = self.text_encoding.detach(),
        )

        # Freeze the transformer input on the same window as the target.
        # cognitive_fuse depends on live neurons / mem / llm_embed_ca so
        # it moves every epoch; pinning the target while the input drifts
        # is what made base_loss thrash. We pin both together and only
        # let the input move when the target is refreshed.
        frozen_input = (
            self._cached_fused_cog is not None
            and (epoch - self._target_epoch) < TARGET_FREEZE_INTERVAL
        )
        if frozen_input:
            fused_cog_for_tf = self._cached_fused_cog.to(DEVICE)
        else:
            self._cached_fused_cog = fused_cog_raw.detach()
            fused_cog_for_tf = fused_cog_raw

        # Re-center the C-side output before the fusion transformer
        # so a stuck bias in any dimension doesn't get memorised into
        # the transformer weights; the EMA norm update happens outside
        # _forward (in run()) on the detached value to keep this clean
        fused_cog = self.fused_cog_norm.normalize(fused_cog_for_tf)

        # fusion_transformer is fully in-graph
        fused = self.fusion_transformer(
            fused_cog.unsqueeze(0), frozen_input=frozen_input
        )

        target = torch.tensor(
            neuron_pred, dtype=torch.float32
        ).to(DEVICE).unsqueeze(0)

        # MAML inner loop on a throw-away clone
        adapted   = maml_inner_update(
            self.model,
            x_temporal.unsqueeze(0),
            target,
            self.criterion,
            embed_ctx,
        )
        maml_pred = adapted(x_temporal.unsqueeze(0), embed_ctx)
        # NOTE: we do NOT fuse maml_pred with neuron_pred (the target)
        # here doing so would let target information leak into the
        # outer loss, letting the model appear to improve by relying
        # on the fusion blend rather than actually learning.

        # Stash raw fused_cog so the verifier can probe the fusion
        # transformer path independently; already detached at the
        # C boundary so we just keep a reference here
        self._last_fused_cog = fused_cog_raw.detach()

        return self._Fwd(
            model_pred   = model_pred,
            fused        = fused,
            maml_pred    = maml_pred,
            target       = target,
            mc_variance  = mc_variance,
            llm_embed_ca = llm_embed_ca,
        )

    def _backward(self, fwd: "_Fwd", epoch: int) -> float:
        """
        Computes the scalar loss from a ForwardResult, calls .backward(),
        clips gradients, and steps optimizer + scheduler + EMA.
        Returns the float loss value for logging.

        On a frozen-input window the (input, target) pair is identical
        across the whole window, so once it is essentially learned we
        stop stepping on it. Continuing to step a memorised pair drives
        grads to ~1e-4 and wastes scheduler/EMA progress, then the next
        refresh delivers a shock — that pulse is the sawtooth seen in
        the loss logs.
        """
        raw_blend = float(
            max(0.5 - fwd.mc_variance * 0.5, 0.1)
        )
        self._mc_blend_ema = (
            0.9 * self._mc_blend_ema + 0.1 * raw_blend
        )
        mc_blend = self._mc_blend_ema

        # No tanh it saturates gradients when model outputs drift
        # beyond ~±2.  Targets are left unscaled so the model head
        # converges naturally over more epochs.
        model_pred = fwd.model_pred
        fused      = fwd.fused
        maml_pred  = fwd.maml_pred

        scaled_target = fwd.target
        outer_loss = self._vec_loss(maml_pred, scaled_target)
        base_loss  = self._vec_loss(fused, scaled_target)
        pred_loss  = self._vec_loss(model_pred, scaled_target)

        # cross_attn, embed_weight_net and text_proj sit behind
        # cognitive_fuse which breaks the graph (C boundary). aux_loss
        # carries the gradient signal for those modules. We project the
        # cross-attention embedding DOWN to the target dim and regress
        # it against the target a real objective that can actually fall 
        ca_pred  = self.aux_proj(fwd.llm_embed_ca.unsqueeze(0))
        aux_loss = self._vec_loss(ca_pred, fwd.target)

        print(
            f"outer={outer_loss.item():.4f} "
            f"base={base_loss.item():.4f} "
            f"pred={pred_loss.item():.4f} "
            f"aux={aux_loss.item():.4f} "
            f"mc_blend={mc_blend:.4f}"
        )

        loss = (
            mc_blend           * outer_loss
            + (1.0 - mc_blend) * 0.4 * base_loss
            + (1.0 - mc_blend) * 0.3 * pred_loss
            + 0.2              * aux_loss
        )

        if epoch == 1:
            print(
                f"  loss components — "
                f"outer={outer_loss.item():.4f}  "
                f"base={base_loss.item():.4f}  "
                f"pred={pred_loss.item():.4f}  "
                f"aux={aux_loss.item():.4f}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

        loss_val = float(loss.item())

        # Skip the step on an already-learned frozen pair. We are some
        # epochs into the freeze window (not the refresh epoch) and the
        # loss is below the floor, so stepping again only memorises
        # harder and produces the grad pulse / sawtooth artefact.
        in_frozen_window = (epoch - self._target_epoch) > 0
        if in_frozen_window and loss_val < self._frozen_skip_floor:
            return loss_val

        loss.backward()

        # Gradient balancing: boost the weak embed_weight_net branch
        # BEFORE clip so the rescaling survives into the optimizer step.
        # We balance against an EMA of the fusion_transformer grad norm
        # rather than its instantaneous value the instantaneous norm
        # pulses hard on refresh epochs and balancing off that pulse.
        ft_params = [
            p for p in self.fusion_transformer.parameters()
            if p.grad is not None
        ]
        if ft_params:
            ft_norm_now = sum(
                p.grad.norm().item() for p in ft_params
            )
            self._ft_norm_ema = (
                0.9 * self._ft_norm_ema + 0.1 * ft_norm_now
                if self._ft_norm_ema > 0.0
                else ft_norm_now
            )
            ft_norm = self._ft_norm_ema
            for module in [self.embed_weight_net]:
                params = [
                    p for p in module.parameters()
                    if p.grad is not None
                ]
                if not params:
                    continue
                norm = sum(p.grad.norm().item() for p in params)
                if norm > 0 and ft_norm / norm > 10.0:
                    scale = ft_norm / (norm * 10.0)
                    scale = min(scale, 5.0)
                    for p in params:
                        p.grad.mul_(scale)

        # Global clip prevents any module from exploding after
        # balancing; runs last so both strong and boosted-weak
        # grads are scaled together.
        all_params = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
        ]
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)

        self.optimizer.step()
        self.ema.update(self.model)

        # Scheduler steps here and owns the LR from this point on.
        self.scheduler.step()

        return loss_val

    # ------------------------------------------------------------------
    # side-effects  (no grad, pure I/O + C backend triggers)
    # ------------------------------------------------------------------

    def _side_effects(
        self,
        epoch:        int,
        lr:           float,
        alpha:        float,
        loss_val:     float,
        fused_np:     "np.ndarray",
        fused_vec:    torch.Tensor,
        input_tensor: "np.ndarray",
        model_pred_np:"np.ndarray",
        neuron_pred:  "np.ndarray",
        region_scores: list,
    ) -> None:
        """
        All C backend triggers, logging and emotional pipeline updates.
        Nothing here touches tensors that need to be differentiated.
        """
        if not torch.isfinite(fused_vec).all():
            fused_vec = torch.nan_to_num(
                fused_vec, nan=0.0, posinf=1.0, neginf=-1.0
            )
        # decode(fused) is a pure readout now a window into fused,
        # not a trained objective. text_input is printed alongside it
        # for debugging the driver sentence that shaped fused.
        text_output = self.text_codec.decode(fused_vec)
        print(
            f"  [epoch {epoch}] "
            f"text_input  : {self._current_text_input}"
        )
        print(f"  [epoch {epoch}] text_output : {text_output}")

        self.backend.add_memory_step(fused_np.tolist())

        updated_meta = self.backend.update_meta(
            region_scores
        )
        _ = updated_meta

        reflect_metrics = self.backend.run_reflection()
        (reflect_conf, reflect_drift,
         reflect_novelty, reflect_coherence) = _derive_reflection_signal(
            reflect_metrics
        )
      
        # Re-gentlefy so the next epoch starts with unsaturated targets.
        self.backend.process_neurons(scaled_factor=0.6)
        self.backend.update_neuron_states(
            scaled_factor=0.6
        )

        novelty         = _derive_novelty(
            self.loss_history, loss_val
        )
        # Blend the backend's own novelty read with the loss-derived
        # one, and fold drift in on top, so a network that reports it
        # is drifting or sees novelty raises the signal even when the
        # loss alone looks flat. max() so reflection only adds urgency,
        # never masks a genuine loss-driven spike.
        novelty         = max(novelty, reflect_novelty, reflect_drift)
        perf_delta      = self.prev_loss - loss_val
        task_difficulty = float(min(loss_val, 1.0))
        self.backend.update_motivation(
            perf_delta, novelty, task_difficulty
        )

        # Low reflection confidence -> push more creative exploration.
        # An unsure backend explores harder without overriding the
        # perf-driven signal entirely.
        self.backend.update_imagination_creativity(
            perf_delta,
            max(novelty, 1.0 - reflect_conf),
        )

        self.backend.problem_solve_with_imagination(
            total_error=float(min(loss_val, 1.0)),
        )
        if epoch % 30 == 0:
            self.backend.store_best_imagination_to_memory()

        self.backend.update_identity(fused_np.tolist())

        identity_ok = self.backend.verify_identity()
        if not identity_ok.get("verified", True):
            print(
                f"  [epoch {epoch}] identity verification failed"
            )

        self.backend.detect_specializations(
            target_outputs=(
                neuron_pred.tolist()
                if hasattr(neuron_pred, "tolist")
                else [float(v) for v in neuron_pred]
            ),
        )
        self.backend.apply_specializations()
        self.backend.update_specialization_importance(
            network_performance=float(1.0 - min(loss_val, 1.0)),
            error_rate=float(min(loss_val, 1.0)),
        )
        spec_eval = self.backend.evaluate_specialization_effectiveness(
            network_performance=float(1.0 - min(loss_val, 1.0)),
        )
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  [epoch {epoch}] "
                f"spec effectiveness="
                f"{spec_eval.get('effectiveness', '?'):.4f}"
            )
            print(
                f"  [epoch {epoch}] "
                f"reflection — confidence={reflect_conf:.4f}  "
                f"drift={reflect_drift:.4f}  "
                f"novelty={reflect_novelty:.4f}  "
                f"coherence={reflect_coherence:.4f}"
            )

        satisfaction = _derive_satisfaction(loss_val, self.prev_loss)
        self.backend.detect_emotional_triggers(
            target_outputs=(
                neuron_pred.tolist()
                if hasattr(neuron_pred, "tolist")
                else [float(v) for v in neuron_pred]
            ),
            satisfaction=satisfaction,
        )

        emo_type, emo_strength = _pick_emotion_type(
            loss_val, self.prev_loss, novelty
        )
        self.backend.trigger_emotion(emo_type, emo_strength)

        # Plasticity is tied to the same deadband as the emotion type:
        # when nothing notable happened we feed near-zero plasticity so
        # arousal can decay instead of being held high every epoch,
        # which was part of what pinned the affective state to a corner.
        _emo_active = abs(self.prev_loss - loss_val) > 0.03
        self.backend.apply_emotional_processing(
            learning_rate=lr,
            plasticity=float(
                min(novelty * 0.3, 0.3) if _emo_active else 0.02
            ),
        )

        # Trust reflects task satisfaction and the backend's own
        # coherence read together  a satisfied run that the backend
        # also judges coherent earns more trust than satisfaction alone
        bond_trust      = float(
            min(max(0.5 * satisfaction + 0.5 * reflect_coherence,
                    0.0), 1.0)
        )
        # Attachment grows toward, not jumps to, a target so it does
        # not pin at the ceiling after a few good epochs.
        _attach_target  = float(min(1.0 - min(loss_val, 1.0), 1.0))
        bond_attachment = float(0.7 * 0.5 + 0.3 * _attach_target)
        # Valence keyed on the trend (perf_delta) so it swings both
        # ways instead of accumulating in one corner the way the old
        # satisfaction*2-1 did once loss settled.
        _trend = self.prev_loss - loss_val
        bond_valence    = float(
            max(min(_trend * 4.0, 1.0), -1.0)
        )
        self.backend.update_bond(
            attachment_strength=bond_attachment,
            trust=bond_trust,
            valence=bond_valence,
        )

        if epoch == 1 or epoch % EMOTION_LOG_INTERVAL == 0:
            emo  = self.backend.get_emotional_state()
            aff  = self.backend.get_affective_state()
            mask = self.backend.compute_mask_intensity()
            _log_emotional_snapshot(epoch, emo, aff, mask)

        log_epoch(
            epoch,
            input_tensor,
            model_pred_np,
            neuron_pred,
            fused_np,
            loss_val,
            lr,
            alpha,
        )
        log_context(self.backend.get_context_state())
        log_history(self.backend.get_network_history())
        log_memory(self.backend.get_memory_state())

        result = self.backend.receive_predictions(
            epoch, neuron_pred.tolist(), fused_np.tolist()
        )
        print(f"backend ack : {result}")

    def run(self) -> None:
        # Seed the optimizer LR once from the first meta read so
        # derive_lr()'s initial value is respected, then the scheduler
        # owns all subsequent adjustments no more per-epoch override
        _seed_meta = self.backend.get_meta_state()
        _seed_lr   = derive_lr(_seed_meta)
        update_optimizer_lr(self.optimizer, _seed_lr)

        for epoch in range(1, self.epochs + 1):

            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    print(name, p.grad.norm().item())

            # --- per-epoch backend state reads ---
            meta       = self.backend.get_meta_state()
            backend_lr = derive_lr(meta)

            # The actual training LR is managed by the scheduler
            # (CosineAnnealingLR stepped after each backward).
            # backend_lr is only used for emotional / identity
            # updates in the C backend completely separate.

            ctx   = self.backend.get_context_state()
            alpha = derive_alpha_from_context(ctx)
            alpha = derive_alpha_from_params(
                self.backend.get_dynamic_params(), alpha
            )

            # Caller's initial_alpha blends with the context-derived
            # value rather than overriding it outright avoids
            # shifting region_scores enough to destabilize backend state
            if self.initial_alpha is not None:
                alpha = (alpha + self.initial_alpha) * 0.5

            region_scores = [alpha] * NUM_REGIONS
            self.backend.run_decision_path(region_scores)
            self.backend.update_context()

            self.backend.process_neurons(scaled_factor=0.6)
            self.backend.update_neuron_states(
                scaled_factor=0.6
            )

            # Cycle through a fixed sample pool so the model sees
            # repeated texts across epochs the neuron targets still
            # evolve (C backend updates), but the consistent text input
            # lets the model isolate its own effect on the targets.
            if len(self._sample_pool) < self._sample_pool_size:
                _sample = self._data_pipeline.next_sample()
                self._sample_pool.append(_sample)
            else:
                _sample = self._sample_pool[
                    self._sample_pool_idx % self._sample_pool_size
                ]
                self._sample_pool_idx += 1
            self._current_text_input = _sample
            self.text_encoding = (
                self.text_codec.encode(_sample).detach()
            )

            step = epoch - 1
            rising_error = (
                len(self.loss_history) > 10
                and self.loss_history[-1] > self.loss_history[-10]
            )
            if step % 15 == 0 or rising_error:
                self.backend.activate_imagination_scenario(
                    divergence=(
                        0.2 + __import__("random").random() * 0.3
                    ),
                    task_description=self._current_text_input,
                    simulate_steps=10,
                )

            imag_state = self.backend.get_imagination_state()
            if imag_state.get("active", False):
                self.backend.apply_imagination_to_decision()
                self.backend.adjust_neurons_with_imagination(
                    outcome_index=0,
                    influence_factor=alpha,
                )

            self.backend.update_attractor_dynamics()
            self.backend.update_affective_complexity()
            self.backend.reshape_embeddings_with_emotion()

            neurons      = self.backend.get_neurons()
            input_tensor = self.backend.get_input_tensor()

            # --- input pipeline ---
            x_raw = torch.tensor(
                input_tensor, dtype=torch.float32
            ).to(DEVICE)
            self.online_norm.update(x_raw)
            x_norm    = self.online_norm.normalize(x_raw)
            x_fourier = fourier_encode(x_norm)
            self._last_x_fourier = x_fourier.detach()

            self.input_history.append(x_fourier.detach().cpu())
            padded = list(self.input_history)
            while len(padded) < TEMPORAL_WINDOW:
                padded.insert(0, torch.zeros(FOURIER_OUT_DIM))
            x_temporal = torch.cat(
                [t.to(DEVICE) for t in padded], dim=-1
            )

            embed_ctx = _build_embed_ctx(ctx)

            novelty          = _derive_novelty(
                self.loss_history, self.prev_loss
            )
            exploration_rate = float(novelty)

            mem_state = self.backend.get_memory_state()

            dw_state  = [
                float(
                    neurons.get(
                        f"neuron_{i}", {}
                    ).get("state", 0.0)
                )
                for i in range(MAX_NEURONS)
            ]
            dw_output = [
                float(
                    neurons.get(
                        f"neuron_{i}", {}
                    ).get("output", 0.0)
                )
                for i in range(MAX_NEURONS)
            ]
            dw_input  = input_tensor.tolist()
            default_weights = torch.tensor(
                dw_state + dw_output + dw_input,
                dtype=torch.float32,
            ).to(DEVICE)

            if not self.loss_history:
                mem_novelty = 1.0
            else:
                mean_loss = (
                    sum(self.loss_history)
                    / len(self.loss_history)
                )
                mem_novelty = float(
                    min(
                        abs(self.prev_loss - mean_loss)
                        / (mean_loss + 1e-8),
                        1.0,
                    )
                )
            mem_weight_ratio = (
                MEM_WEIGHT_RATIO_BASE
                + MEM_WEIGHT_RATIO_RANGE
                * float(
                    torch.sigmoid(
                        -torch.tensor(mem_novelty * 3.0)
                    ).item()
                )
            )

            # Build target from live backend state, but freeze it
            # for TARGET_FREEZE_INTERVAL epochs to prevent the model
            # from chasing a moving target driven by side-effects.
            # The live value is still passed to _side_effects so the
            # backend receives up-to-date neuron signals.
            neuron_pred_live = build_neuron_prediction(neurons)
            if (self._cached_target is None
                or (epoch - self._target_epoch) >= TARGET_FREEZE_INTERVAL):
                self._cached_target = neuron_pred_live.copy()
                self._target_epoch  = epoch
                # Drop the cached fused-cog input too so _forward takes
                # a fresh C-side reading on the same epoch the target
                # refreshes input and target move together, then both
                # stay pinned for the rest of the window.
                self._cached_fused_cog = None
            neuron_pred = self._cached_target

            # ---- forward ----
            self.model.train()
            self.optimizer.zero_grad()

            fwd = self._forward(
                x_temporal       = x_temporal,
                x_norm           = x_norm,
                embed_ctx        = embed_ctx,
                neurons          = neurons,
                neuron_pred      = neuron_pred,
                mem_state        = mem_state,
                default_weights  = default_weights,
                mem_weight_ratio = mem_weight_ratio,
                alpha            = alpha,
                exploration_rate = exploration_rate,
                epoch            = epoch,
            )

            # ---- backward ----
            loss_val = self._backward(fwd, epoch)

            # Only advance the fused_cog distribution stats when the
            # input is actually allowed to move (i.e. a refresh epoch).
            if (epoch - self._target_epoch) == 0:
                self.fused_cog_norm.update(self._last_fused_cog)

            if epoch == 1 or epoch % VERIFY_INTERVAL == 0:
                # Comprehensive verification loss: includes all module
                # paths so gradient checks reflect actual training
                # dynamics instead of stale leftover grads.  Zero grads
                # first for a clean measurement.
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    _vp    = self.model(
                        x_temporal.unsqueeze(0), embed_ctx
                    )
                    _llm_embed_raw = _vp.squeeze(0)
                    _llm_embed = _llm_embed_raw[:INTERNAL_DIM]
                    _embed_gate = self.embed_weight_net(x_norm)
                    _llm_embed_g = _llm_embed * _embed_gate
                    _llm_embed_ca_v = self.cross_attn(
                        x_norm, _llm_embed_g
                    )
                    _text_proj_out = self.text_proj(
                        self.text_encoding.to(DEVICE)
                    )
                    _llm_embed_ca_v = (
                        _llm_embed_ca_v
                        + _text_proj_out[
                            :_llm_embed_ca_v.shape[-1]
                        ]
                    )

                    _fused_cog_v = cognitive_fuse(
                        llm_embed        = _llm_embed_ca_v.detach(),
                        neurons          = neurons,
                        mem_state        = mem_state,
                        default_weights  = default_weights,
                        mem_weight_ratio = mem_weight_ratio,
                        context_factor   = alpha,
                        text_embed       = self.text_encoding.detach(),
                    )
                    # Mirror training: inside a freeze window the
                    # transformer was trained on the cached input, so
                    # the verifier must probe that same input or its
                    # reported loss describes a path training never took
                    _vfrozen = (
                        self._cached_fused_cog is not None
                        and (epoch - self._target_epoch)
                        < TARGET_FREEZE_INTERVAL
                    )
                    if _vfrozen:
                        _fused_cog_v = self._cached_fused_cog.to(DEVICE)
                    # Same re-centering as _forward so the verifier
                    # sees the identical normalised input
                    _fused_cog_v = self.fused_cog_norm.normalize(
                        _fused_cog_v
                    )
                    _vf = self.fusion_transformer(
                        _fused_cog_v.unsqueeze(0),
                        frozen_input=_vfrozen,
                    )

                    _neuron_pred_t = torch.tensor(
                        neuron_pred, dtype=torch.float32
                    ).to(DEVICE).unsqueeze(0)

                    _vpred = self.smooth_l1(
                        _vp, fwd.target
                    )
                    _vbase = self.smooth_l1(
                        _vf, _neuron_pred_t
                    )
                    _vaux_ca   = self.aux_proj(
                        _llm_embed_ca_v.unsqueeze(0)
                    )
                    _vaux = self.smooth_l1(_vaux_ca, fwd.target)
                    _vloss = (
                        0.4 * _vbase
                        + 0.3 * _vpred
                        + 0.2 * _vaux
                    )

                run_verification(
                    epoch              = epoch,
                    model              = self.model,
                    fusion_transformer = self.fusion_transformer,
                    embed_weight_net   = self.embed_weight_net,
                    cross_attn         = self.cross_attn,
                    text_proj          = self.text_proj,
                    optimizer          = self.optimizer,
                    criterion          = self.criterion,
                    x_temporal         = x_temporal,
                    x_norm             = x_norm,
                    x_fourier          = x_fourier,
                    model_pred         = _vp.detach(),
                    fused              = _vf.detach(),
                    fused_cog          = self._last_fused_cog,
                    target             = fwd.target,
                    loss               = _vloss,
                    loss_val           = loss_val,
                    loss_history       = self.loss_history,
                    lr                 = backend_lr,
                    embed_ctx          = embed_ctx,
                    pattern_tracker    = self._pattern_tracker,
                )

            # ---- side-effects (no grad) ----
            fused_np  = (
                fwd.fused.squeeze(0).detach().cpu().numpy()
            )
            fused_vec = fwd.fused.squeeze(0).detach()

            # Pass the LIVE neuron_pred (not the cached training target)
            # to side-effects so the backend receives up-to-date signals.
            self._side_effects(
                epoch         = epoch,
                lr            = backend_lr,
                alpha         = alpha,
                loss_val      = loss_val,
                fused_np      = fused_np,
                fused_vec     = fused_vec,
                input_tensor  = input_tensor,
                model_pred_np = (
                    fwd.model_pred.detach().cpu().numpy()
                ),
                neuron_pred   = neuron_pred_live,
                region_scores = region_scores,
            )

            self.loss_history.append(loss_val)
            self.prev_loss = loss_val

        # --- post-training ---
        self.backend.consolidate_memory()
        self.backend.save_memory("memory.bin")
        self.backend.save_network_states()

        # Torch-side state pairs with the C-side saves above; both
        # halves together are one checkpoint, neither is usable alone.
        save_result = save_checkpoint(self, "larkos_model.pt")
        print(f"  checkpoint : {save_result}")

        final_reflection = self.backend.get_identity_reflection()
        print(
            "Identity reflection:",
            final_reflection.get("reflection", ""),
        )
        print("=" * 48)
        print("Post-training saves complete.")


def training_loop(
    backend:  "BackendState",
    epochs:   int   = 5,
    alpha:    float = 0.5,
    data_dir=None,
    resume_from: str | None = None,
    resume_use_ema: bool = False,
) -> None:
    TrainingLoop(
        backend,
        epochs,
        initial_alpha=alpha,
        data_dir=data_dir,
        resume_from=resume_from,
        resume_use_ema=resume_use_ema,
    ).run()
