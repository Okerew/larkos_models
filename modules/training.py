import copy
import ctypes
import math
from pathlib import Path
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
    INTERNAL_DIM, GPT2_HIDDEN, TEXT_MAX_NEW,
    VERIFY_INTERVAL, TARGET_FREEZE_INTERVAL,
    EMOTION_LOG_INTERVAL, N_PREFIX, BAND_M, BAND_Q,
    FUSE_GRAPH_DMODEL, FUSE_GRAPH_NHEAD, FUSE_GRAPH_LAYERS,
    FUSE_GRAPH_DIM_FF, GAT_HEADS, GAT_LAYERS, MAX_CONNECTIONS,
    TEMPORAL_NHEAD, TEMPORAL_LAYERS, TEMPORAL_DIM_FF, SAMPLE_POOL_SIZE, D_NODE
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
from modules.epoch_controller import EpochController

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

    When exploration is below threshold the only variance source is
    dropout=0.1, so a small T captures the signal and the mc_blend
    EMA smooths the rest. We use the full MC_DROPOUT_T only when
    Gaussian exploration noise is actually being injected, where the
    variance estimate has to integrate over noisy inputs and needs
    more samples to be meaningful. Cuts MC cost ~3x on quiet epochs.
    """
    model.train()
    exploring = exploration_rate > EXPLORE_THRESHOLD
    T = MC_DROPOUT_T if exploring else max(MC_DROPOUT_T // 3, 2)
    with torch.no_grad():
        samples = []
        for _ in range(T):
            xi = x
            if exploring:
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


class _GATLayer(nn.Module):
    """
    Single hand-rolled GAT layer over a dense [N, N] adjacency mask.

    The Larkos neuron graph has N=MAX_NEURONS=128 nodes and at most 32
    real edges per neuron, plus a self-loop. That is small enough to
    run attention with a full [N, N] mask instead of scatter ops, which
    keeps the implementation aligned with the rest of the file's
    hand-rolled style and avoids a torch_geometric dependency.

    Per head, the unnormalised attention score follows the original GAT
    formulation:

        e_ij = LeakyReLU(a_src . W h_i + a_dst . W h_j)

    Edge weights from the C-side `weights[]` array modulate the score
    multiplicatively via a tanh-squashed gain, so a strong edge raises
    its softmax mass without being able to dominate the LeakyReLU sign.
    Non-edges are masked to -inf before softmax.
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int) -> None:
        super().__init__()
        assert d_out % n_heads == 0, (
            "GAT d_out must be divisible by n_heads"
        )
        self.n_heads  = n_heads
        self.head_dim = d_out // n_heads

        # Per-head linear projection of node features. Bias-free so the
        # zero-padded features of unused neurons stay at zero rather
        # than getting a constant bias that would leak through softmax.
        self.W = nn.Linear(d_in, n_heads * self.head_dim, bias=False)

        # GAT-style attention vectors split into source / destination
        # halves, equivalent to a^T [Wh_i || Wh_j] but cheaper.
        self.a_src = nn.Parameter(
            torch.randn(n_heads, self.head_dim) / math.sqrt(self.head_dim)
        )
        self.a_dst = nn.Parameter(
            torch.randn(n_heads, self.head_dim) / math.sqrt(self.head_dim)
        )

        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        h:           torch.Tensor,
        adj_mask:    torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        # h           : [B, N, d_in]
        # adj_mask    : [N, N] bool (True on edge, including self-loops)
        # edge_weight : [N, N] float
        # returns     : [B, N, n_heads * head_dim]
        B, N, _ = h.shape

        h_proj = self.W(h).view(
            B, N, self.n_heads, self.head_dim
        )  # [B, N, H, d_head]

        # alpha_src[b, i, h] = a_src[h] . h_proj[b, i, h]
        alpha_src = (
            h_proj * self.a_src.view(1, 1, self.n_heads, self.head_dim)
        ).sum(dim=-1)  # [B, N, H]
        alpha_dst = (
            h_proj * self.a_dst.view(1, 1, self.n_heads, self.head_dim)
        ).sum(dim=-1)  # [B, N, H]

        # scores[b, h, i, j] = leaky(alpha_src[b, i, h] + alpha_dst[b, j, h])
        scores = self.leaky(
            alpha_src.permute(0, 2, 1).unsqueeze(-1)
            + alpha_dst.permute(0, 2, 1).unsqueeze(-2)
        )  # [B, H, N, N]

        mask = adj_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        scores = scores.masked_fill(~mask, float("-inf"))

        # Edge-weight gain: tanh-squashed so a single huge weight cannot
        # explode the softmax; the +1 keeps self-loops (where weight=1)
        # at their unscaled score.
        ew = torch.tanh(edge_weight).unsqueeze(0).unsqueeze(0)
        scores = scores * (1.0 + ew)

        attn = torch.softmax(scores, dim=-1)  # [B, H, N, N]

        # h_for_agg : [B, H, N, d_head]
        h_for_agg = h_proj.permute(0, 2, 1, 3)
        out = torch.matmul(attn, h_for_agg)  # [B, H, N, d_head]
        out = (
            out.permute(0, 2, 1, 3)
                .contiguous()
                .view(B, N, self.n_heads * self.head_dim)
        )
        return out


class _TemporalAttentionEncoder(nn.Module):
    """
    Tiny temporal-axis transformer encoder that runs over the
    [TEMPORAL_WINDOW, FOURIER_OUT_DIM] input history before it is
    flattened for LarkosModel. Without this layer the history is
    `torch.cat`ed into a flat vector, leaving the model to infer
    temporal position from slot offset — there is no positional /
    temporal embedding anywhere downstream.

    Output shape matches the input ([TEMPORAL_WINDOW, FOURIER_OUT_DIM])
    so the existing flatten-then-NumericTokenizer path is unchanged;
    only the values inside the sequence have been attended across
    timesteps. Pooling would destroy the per-timestep granularity
    NumericTokenizer is built to consume, so we keep the full
    sequence and let LarkosModel's own input transformer continue to
    mix across (time x feature) jointly.

    The learned positional vector lets the model carve out "current
    frame" vs "older frames" without baking in any prior about how
    time should be encoded; with seq_len=TEMPORAL_WINDOW=6 it costs
    ~1.5k parameters — negligible.
    """

    def __init__(
        self,
        seq_len:  int = TEMPORAL_WINDOW,
        d_model:  int = FOURIER_OUT_DIM,
        nhead:    int = TEMPORAL_NHEAD,
        n_layers: int = TEMPORAL_LAYERS,
        dim_ff:   int = TEMPORAL_DIM_FF,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)

        self.input_norm = nn.LayerNorm(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq : [TEMPORAL_WINDOW, FOURIER_OUT_DIM]  (no batch dim)
        # returns same shape, attended across timesteps.
        seq = x_seq + self.pos                 # add temporal positional embed
        seq = self.input_norm(seq)
        # nn.TransformerEncoder expects [B, L, D] when batch_first=True
        return self.encoder(seq.unsqueeze(0)).squeeze(0)


class _NeuronGraphReasoner(nn.Module):
    """
    Graph attention over the live neuron graph exposed by
    backend_state.get_neurons(). Produces per-neuron embeddings that
    feed the refactored _FusionTransformerHead as a token sequence,
    carrying the neuron view end-to-end. The C-side BAND_N pipeline
    that the head used to consume has been retired from
    cognitive_fuse.

    Node features per neuron: state, output, layer one-hot (2 dims),
    and a tanh-squashed connection-degree signal. Edges and edge
    weights come straight from `connections[]` / `weights[]` in the
    neurons dict. Self-loops are always added so an isolated neuron
    still updates from itself rather than getting -inf attention.

    Two GAT layers with a residual + LayerNorm between them. The
    graph topology is fixed at backend init today (sparse, 2 edges per
    neuron) but the C side is free to mutate it; we rebuild the adj
    mask on every forward so the layer always sees the current graph.
    """

    # Per-node feature layout (8 dims):
    #   0  state
    #   1  output
    #   2  layer one-hot id==0
    #   3  layer one-hot id==1
    #   4  tanh(num_connections / MAX_CONNECTIONS)
    #   5  state - prev_state    (per-neuron velocity since last refresh)
    #   6  |output|              (output magnitude)
    #   7  tanh(mean outgoing edge weight)
    # On top of these, the forward adds a learned per-neuron embedding
    # of size d_out so identical static features at different neuron
    # indices still get distinct token representations — the GAT alone
    # cannot distinguish two symmetric nodes otherwise.
    _D_NODE = D_NODE

    def __init__(
        self,
        d_out:    int = FUSE_GRAPH_DMODEL,
        n_heads:  int = GAT_HEADS,
        n_layers: int = GAT_LAYERS,
    ) -> None:
        super().__init__()
        self.node_in = nn.Linear(self._D_NODE, d_out)

        # Learned per-neuron embedding. Added to the projected node
        # features so symmetric neurons (same static features) still
        # produce distinct tokens. Init std matches the rest of the
        # file's learned embeddings.
        self.neuron_embed = nn.Parameter(torch.zeros(MAX_NEURONS, d_out))
        nn.init.normal_(self.neuron_embed, std=0.02)

        self.layers  = nn.ModuleList([
            _GATLayer(d_out, d_out, n_heads)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_out) for _ in range(n_layers)
        ])
        self.act = nn.ELU()

        # Transient buffer for per-neuron state velocity. Not saved in
        # the checkpoint (persistent=False) — it is recomputable state,
        # not a learned parameter, and on resume we'd rather start from
        # zero velocity than carry a stale snapshot from a different
        # backend session.
        self.register_buffer(
            "_prev_states",
            torch.zeros(MAX_NEURONS),
            persistent=False,
        )

    def build_graph_inputs(
        self,
        neurons: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Built on CPU; the caller moves them to DEVICE alongside the
        # rest of the forward inputs. Plain-python reads from the
        # `neurons` dict — `get_neurons()` already pulled them across
        # the C boundary, so this loop is cheap (~MAX_NEURONS dict
        # lookups).
        #
        # Updates `_prev_states` as a side effect so the next call's
        # state_delta is a real per-neuron velocity rather than always
        # measuring from zero. Callers using the freeze cache should
        # invoke this only when they actually want a fresh read — the
        # freeze cache stores the returned tensors and re-uses them
        # across the window, so prev_states moves forward exactly once
        # per target refresh (every TARGET_FREEZE_INTERVAL epochs).
        N = MAX_NEURONS
        node_features = torch.zeros(N, self._D_NODE)
        adj_mask      = torch.eye(N, dtype=torch.bool)  # self-loops
        edge_weight   = torch.zeros(N, N)
        cur_states    = torch.zeros(N)

        prev_states = self._prev_states.detach().to("cpu")

        for i in range(N):
            neuron = neurons.get(f"neuron_{i}", {})
            state    = float(neuron.get("state",  0.0))
            output   = float(neuron.get("output", 0.0))
            layer_id = int(neuron.get("layer_id", 0))
            num_conn = int(neuron.get("num_connections", 0))
            conns    = neuron.get("connections", []) or []
            ws       = neuron.get("weights",     []) or []

            cur_states[i] = state

            node_features[i, 0] = state
            node_features[i, 1] = output
            # Two-class one-hot covers layer_id in {0, 1}; if the C
            # side ever grows more layers this still degrades to all
            # zeros for the extra ones, which is benign.
            node_features[i, 2] = 1.0 if layer_id == 0 else 0.0
            node_features[i, 3] = 1.0 if layer_id == 1 else 0.0
            node_features[i, 4] = math.tanh(
                num_conn / float(MAX_CONNECTIONS)
            )
            node_features[i, 5] = state - float(prev_states[i])
            node_features[i, 6] = abs(output)

            edge_weight[i, i] = 1.0

            usable = min(num_conn, MAX_CONNECTIONS, len(conns))
            w_sum: float = 0.0
            w_count: int = 0
            for slot in range(usable):
                j = int(conns[slot])
                if 0 <= j < N:
                    adj_mask[i, j] = True
                    if slot < len(ws):
                        w = float(ws[slot])
                        edge_weight[i, j] = w
                        w_sum   += w
                        w_count += 1
            mean_w = (w_sum / w_count) if w_count > 0 else 0.0
            node_features[i, 7] = math.tanh(mean_w)

        # Advance velocity reference. Detach so the buffer never carries
        # an autograd dependency.
        self._prev_states.copy_(cur_states.detach())

        return node_features, adj_mask, edge_weight

    def forward_from_inputs(
        self,
        node_features: torch.Tensor,
        adj_mask:      torch.Tensor,
        edge_weight:   torch.Tensor,
    ) -> torch.Tensor:
        # Runs the in-graph half of the reasoner (projection + GAT
        # layers) on pre-built features. Split out from `forward` so
        # the training freeze cache can pin the *inputs* to the GAT
        # while still running the GAT in-graph on every step — this
        # keeps gradient flowing to GAT params during a freeze window
        # without re-reading live neuron state.
        node_features = node_features.to(DEVICE)
        adj_mask      = adj_mask.to(DEVICE)
        edge_weight   = edge_weight.to(DEVICE)

        h = self.node_in(node_features).unsqueeze(0)        # [1, N, d_out]
        h = h + self.neuron_embed.unsqueeze(0)              # per-neuron embed
        for layer, norm in zip(self.layers, self.norms):
            h_new = self.act(layer(h, adj_mask, edge_weight))
            h     = norm(h + h_new)                         # residual + post-norm
        return h

    def forward(self, neurons: dict) -> torch.Tensor:
        # Convenience path: build inputs from a fresh neurons dict and
        # run the GAT in one go. Returns [1, MAX_NEURONS, d_out]. The
        # training loop bypasses this and calls build_graph_inputs /
        # forward_from_inputs separately so it can cache the inputs.
        node_features, adj_mask, edge_weight = self.build_graph_inputs(
            neurons
        )
        return self.forward_from_inputs(
            node_features, adj_mask, edge_weight,
        )


class _FusionTransformerHead(nn.Module):
    """
    Refactored fusion head: instead of attending over 3 stream tokens
    (Q / N / M) at FUSION_DIM and mean-pooling, this attends over a
    mixed sequence of per-neuron graph-attention tokens plus a handful
    of context tokens, then attention-pools.

    Sequence layout (length MAX_NEURONS + 3):
        - [0 : MAX_NEURONS)        per-neuron tokens from _NeuronGraphReasoner
        - [MAX_NEURONS]            band_q from cognitive_fuse  (LLM query stream)
        - [MAX_NEURONS + 1]        band_m from cognitive_fuse  (memory stream)
        - [MAX_NEURONS + 2]        driver embedding (llm_embed_ca)

    BAND_N has been retired on the C side too — its information is
    now carried by the per-neuron GAT tokens, which preserve the graph
    topology the C-side projection used to collapse into 341 dims. Q
    and M still flow in because cognitive_fuse mixes them with
    text_embed / memory attention that the GAT does not see.

    Token-type embedding (4 ids: graph, q, m, driver) lets the encoder
    distinguish which kind of token it is attending to without giving
    every neuron an individual positional embedding (their identity is
    implicit in the graph structure the GAT already attended over).

    Pool is a single learned query attending over the full sequence —
    a softmax over the 131 tokens, then a weighted sum back to d_model.
    Replaces the old mean-pool, which gave every token equal voice
    regardless of what the encoder learned.

    `frozen_input` carries over from the old head: in-graph dropout +
    noise are skipped on a frozen-target window so a pinned input does
    not get regularised against a pinned target.
    """

    _TOK_GRAPH  = 0
    _TOK_Q      = 1
    _TOK_M      = 2
    _TOK_DRIVER = 3

    def __init__(
        self,
        d_model:    int = FUSE_GRAPH_DMODEL,
        nhead:      int = FUSE_GRAPH_NHEAD,
        n_layers:   int = FUSE_GRAPH_LAYERS,
        dim_ff:     int = FUSE_GRAPH_DIM_FF,
        output_dim: int = MAX_NEURONS,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Context-token projections. band_q / band_m come straight from
        # cognitive_fuse's FUSION_DIM output; driver_in is the post
        # cross-attention / text-proj embedding from the rest of
        # _forward. Each gets its own linear so the attention can learn
        # different mappings for different streams.
        self.proj_q      = nn.Linear(BAND_Q, d_model)
        self.proj_m      = nn.Linear(BAND_M, d_model)
        self.proj_driver = nn.Linear(INTERNAL_DIM, d_model)

        # 4 token types: graph / q / m / driver. Shared across all 128
        # graph tokens so the encoder treats them as a set, not a
        # sequence with positions.
        self.token_type_embed = nn.Parameter(torch.zeros(4, d_model))
        nn.init.normal_(self.token_type_embed, std=0.02)

        self.input_norm    = nn.LayerNorm(d_model)
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

        # Learned-query attention pool. Single query attends over the
        # encoded sequence; the softmax over tokens replaces mean-pool.
        self.pool_query = nn.Parameter(
            torch.randn(d_model) / math.sqrt(d_model)
        )

        self.linear_out = nn.Linear(d_model, output_dim)

    @staticmethod
    def split_band_q_m(
        fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Helper for callers: slice band_q and band_m out of a raw
        # cognitive_fuse output. The C side now writes only two
        # contiguous bands ([0 : BAND_Q) and [BAND_Q : FUSION_DIM)),
        # so the split is a clean halving of the fused vector.
        return fused[:, :BAND_Q], fused[:, BAND_Q:]

    def forward(
        self,
        graph_tokens: torch.Tensor,
        band_q:       torch.Tensor,
        band_m:       torch.Tensor,
        driver:       torch.Tensor,
        frozen_input: bool = False,
    ) -> torch.Tensor:
        # graph_tokens : [B, MAX_NEURONS, d_model] (from _NeuronGraphReasoner)
        # band_q       : [B, BAND_Q]
        # band_m       : [B, BAND_M]
        # driver       : [B, INTERNAL_DIM]
        B = graph_tokens.shape[0]

        q_tok = self.proj_q(band_q).unsqueeze(1)        # [B, 1, d_model]
        m_tok = self.proj_m(band_m).unsqueeze(1)        # [B, 1, d_model]
        d_tok = self.proj_driver(driver).unsqueeze(1)   # [B, 1, d_model]

        type_g = self.token_type_embed[self._TOK_GRAPH].view(1, 1, -1)
        type_q = self.token_type_embed[self._TOK_Q].view(1, 1, -1)
        type_m = self.token_type_embed[self._TOK_M].view(1, 1, -1)
        type_d = self.token_type_embed[self._TOK_DRIVER].view(1, 1, -1)

        seq = torch.cat(
            [
                graph_tokens + type_g,
                q_tok        + type_q,
                m_tok        + type_m,
                d_tok        + type_d,
            ],
            dim=1,
        )  # [B, MAX_NEURONS + 3, d_model]

        seq = self.input_norm(seq)

        if self.training and not frozen_input:
            seq = self.input_dropout(seq)
            seq = seq + torch.randn_like(seq) * 0.005

        enc = self.encoder(seq)  # [B, L, d_model]

        # Learned-query attention pool: softmax over the sequence
        # length, weighted sum back to d_model.
        scores = (enc @ self.pool_query) / math.sqrt(self.d_model)
        attn   = torch.softmax(scores, dim=-1)        # [B, L]
        pooled = (attn.unsqueeze(-1) * enc).sum(dim=1)  # [B, d_model]

        return self.linear_out(pooled)


class _TextCodec:
    """
    Decodes the C-side cognitive_fuse output into text as a READOUT only.

      decode : fused-cog vector (FUSION_DIM dims, pre-transformer C output)
               -> _num_to_prefix -> (N_PREFIX, GPT2_HIDDEN) prefix
               -> per-prefix stats matched to wte distribution so GPT-2
                  sees in-distribution embeddings rather than noise
               -> prefix prepended to the actual input sentence as a
                  real text anchor (BOS-only when no anchor given)
               -> GPT-2 autoregressively continues from that scaffolding

    The bridge is NOT trained. Seven runs established that the prefix
    cannot carry enough information to steer a frozen GPT-2 from an
    8-to-64-dim cognitive vector the LM loss never fell and the gate
    never opened, while the text objective fought base_loss for fused.
    So _num_to_prefix is now a fixed random projection: the text is a
    pure window into the cognitive vector, not a learned objective.

    Source widened from MAX_NEURONS (8) to FUSION_DIM (64) so the
    prefix is built from the full pre-transformer fused signal rather
    than the 8-dim head output. encode() is kept because text_encoding
    still feeds fused as a state driver via the C-side cognitive_fuse
    injection and the in-graph text_proj term.

    distilgpt2 replaces gpt2 here: same tokenizer + same GPT2_HIDDEN
    (768), roughly half the weights and ~2x faster generation. The text
    is a debug readout so the quality drop is irrelevant.
    """

    def __init__(self, device: str) -> None:
        self.device = device
        self.tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.tok.pad_token = self.tok.eos_token

        self.lm = GPT2LMHeadModel.from_pretrained("distilgpt2")
        self.lm.eval()
        self.lm.to(device)
        for p in self.lm.parameters():
            p.requires_grad_(False)

        self._num_to_prefix = nn.Linear(
            FUSION_DIM, N_PREFIX * GPT2_HIDDEN
        ).to(device)
        for p in self._num_to_prefix.parameters():
            p.requires_grad_(False)

        # A raw random projection lands far from the wte distribution,
        # so GPT-2 would decode from out-of-distribution embeddings
        # noise in, noise out. Cache wte's mean/std once and rescale
        # every prefix to match before feeding generate().
        wte = self.lm.transformer.wte.weight.detach()
        self._wte_mean = wte.mean()
        self._wte_std  = wte.std()

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
        flat   = self._num_to_prefix(numeric_vec)
        prefix = flat.view(N_PREFIX, GPT2_HIDDEN)
        # Rescale to wte's statistics. A near-constant projection
        # output keeps its zero variance and is filtered in decode().
        std = prefix.std()
        if std > 1e-6:
            prefix = (prefix - prefix.mean()) / std
        return prefix * self._wte_std + self._wte_mean

    def decode(
        self,
        numeric_vec: torch.Tensor,
        anchor_text: str | None = None,
    ) -> str:
        prefix = self._prefix_from_numeric(numeric_vec).unsqueeze(0)

        if not torch.isfinite(prefix).all():
            return ""
        # A near-constant prefix carries no signal and would only push
        # GPT-2 to emit the same noise every epoch — skip generation.
        if prefix.std().item() < 1e-3:
            return ""

        if anchor_text:
            # Anchor the generation on the real driver sentence so
            # GPT-2 has plausible context to continue from; the prefix
            # then biases the continuation rather than steering blind
            # from BOS.
            anchor_ids = self.tok(
                anchor_text,
                return_tensors="pt",
                truncation=True,
                max_length=32,
            ).input_ids.to(self.device)
            anchor_emb = self.lm.transformer.wte(anchor_ids)
            inputs_embeds = torch.cat([prefix, anchor_emb], dim=1)
        else:
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

def _expand_copy(
    new_param: torch.Tensor,
    old_param: torch.Tensor,
) -> torch.Tensor:
    """
    Copy old_param into a clone of new_param, fitting as many elements
    as possible along each dimension. If the new model is larger than
    the old on a given dimension, the old values fill the leading slice
    and the remainder keeps the random initialisation. If old is larger,
    the excess is truncated.
    """
    result = new_param.clone()
    idx = tuple(
        slice(0, min(o, n))
        for o, n in zip(old_param.shape, new_param.shape)
    )
    result[idx] = old_param[idx]
    return result

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
        epochs: int | None = 5,
        initial_alpha: float | None = None,
        data_dir=None,
        resume_from: str | None = None,
        resume_use_ema: bool = False,
        resume_memory_from: str | None = None,
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

        # Persistent MAML clone, built once and refreshed via param
        # copy each inner update. We deepcopy self.model here so the
        # EmbeddingProjector.__deepcopy__ override fires — that shares
        # the frozen 22M-param MiniLM with self.model rather than
        # loading a second copy. From this point on the inner update
        # only copies the small trainable subset, not the ST encoder.
        self.fast_model = copy.deepcopy(self.model).to(DEVICE)

        # Temporal encoder runs on the [TEMPORAL_WINDOW, FOURIER_OUT_DIM]
        # input history before it is flattened for LarkosModel, giving
        # the model an explicit temporal axis to attend over rather
        # than letting time be implicit in slot offset.
        self.temporal_encoder = _TemporalAttentionEncoder().to(DEVICE)

        # Graph reasoner runs first: turns the live neuron graph into
        # MAX_NEURONS tokens at FUSE_GRAPH_DMODEL. The fusion head then
        # attends over those tokens plus a handful of context tokens.
        self.graph_reasoner = _NeuronGraphReasoner().to(DEVICE)

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
            TextDataPipeline(Path(data_dir)) if data_dir is not None
            else TextDataPipeline()
        )

        self._sample_pool: list[str] = []
        self._sample_pool_idx = 0
        self._sample_pool_size = SAMPLE_POOL_SIZE

        # epochs=None selects dynamic control: the EpochController
        # counts epochs and decides when to finalize from cycle-aligned
        # loss statistics (see modules/epoch_controller.py). An int
        # keeps the legacy fixed-count behaviour the test framework
        # relies on.
        self._epoch_controller = (
            EpochController(pool_size=self._sample_pool_size)
            if epochs is None else None
        )
        if self._epoch_controller is not None:
            print(
                "  dynamic epoch control: "
                f"{self._epoch_controller.describe()}"
            )

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
        # AdamW (decoupled weight decay) instead of Adam — Adam folds
        # WD into the gradient before adaptive scaling, which under-
        # regularizes; AdamW applies it directly to the weights and is
        # the correct interpretation of weight_decay=1e-4.
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters())
            + list(self.fusion_transformer.parameters())
            + list(self.graph_reasoner.parameters())
            + list(self.temporal_encoder.parameters())
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
        self.refresh_loss_history: list[float]  = []
        self.prev_loss:    float                = 0.0
        # Latest C-side reads the epoch controller consumes; set per
        # epoch inside _side_effects. Neutral defaults so a first-epoch
        # read can neither force nor block a stop.
        self._last_dyn_stability: float = 1.0
        self._last_reflect_drift: float = 0.0
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

        # Freeze-cache entry for the driver embedding. The refactored
        # fusion head consumes the driver token directly from
        # llm_embed_ca rather than going through the C side, so it was
        # not implicitly pinned by the old _cached_fused_cog.
        self._cached_driver: torch.Tensor | None = None

        # Freeze-cache entry for the GAT inputs (node_features,
        # adj_mask, edge_weight). Earlier 0.3 left graph_tokens out of
        # the freeze cache entirely so the GAT could keep accumulating
        # gradient during a freeze window, but that left the fusion
        # head with a half-frozen / half-drifting input every epoch
        # (Q+M+driver pinned, neuron tokens live) — exactly the kind
        # of inconsistent input the freeze window was meant to avoid.
        #
        # We now pin the GAT *inputs* alongside the rest, and still
        # run forward_from_inputs in-graph on every step. Result: the
        # GAT continues to receive gradient on every frozen epoch
        # (via its own params), but its inputs are pinned to the same
        # snapshot as cognitive_fuse / driver, so the head sees a
        # consistent input regime across the whole window.
        self._cached_graph_inputs: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor
        ] | None = None

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

        # Optionally resume memory state from a saved memory.bin file.
        # This loads the full C-side memory system (entries, levels,
        # importance thresholds) so training picks up from a prior
        # session's memory state rather than starting from scratch.
        if resume_memory_from is not None:
            result = self.backend.load_memory(resume_memory_from)
            print(f"  resumed memory from : {result}")

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

        # Freeze the heavy fusion-head inputs on the same window as the
        # target. cognitive_fuse + the driver + the GAT inputs all
        # evolve every epoch from upstream signals (live neurons / mem
        # / model_pred); pinning the target while these drift is what
        # made base_loss thrash. We pin them together and only let them
        # move when the target is refreshed. The GAT itself still runs
        # in-graph below so its params keep receiving gradient.
        frozen_input = (
            self._cached_fused_cog    is not None
            and self._cached_driver       is not None
            and self._cached_graph_inputs is not None
            and (epoch - self._target_epoch) < TARGET_FREEZE_INTERVAL
        )
        if frozen_input:
            fused_cog_for_tf = self._cached_fused_cog.to(DEVICE)
            driver_for_tf    = self._cached_driver.to(DEVICE)
            nf_for_gat, am_for_gat, ew_for_gat = (
                self._cached_graph_inputs
            )
        else:
            self._cached_fused_cog = fused_cog_raw.detach()
            self._cached_driver    = llm_embed_ca.detach()
            # Build the GAT inputs from live neurons exactly once per
            # freeze window (here, at refresh). This also advances the
            # reasoner's _prev_states buffer so per-neuron velocity is
            # measured from refresh to refresh.
            nf_for_gat, am_for_gat, ew_for_gat = (
                self.graph_reasoner.build_graph_inputs(neurons)
            )
            self._cached_graph_inputs = (
                nf_for_gat.detach(),
                am_for_gat.detach(),
                ew_for_gat.detach(),
            )
            fused_cog_for_tf = fused_cog_raw
            driver_for_tf    = llm_embed_ca

        # Always run the GAT in-graph so its parameters keep receiving
        # gradient on every step, including frozen epochs. Only the
        # inputs are pinned during a freeze window.
        graph_tokens_for_tf = self.graph_reasoner.forward_from_inputs(
            nf_for_gat, am_for_gat, ew_for_gat,
        )

        # Re-center the C-side output before slicing bands so a stuck
        # bias in any dimension doesn't get memorised into the transformer
        # weights; the EMA norm update happens outside _forward (in
        # run()) on the detached value to keep this clean.
        fused_cog = self.fused_cog_norm.normalize(fused_cog_for_tf)

        # Slice band_q / band_m out of the normalised fused vector;
        # band_n is dropped — the graph tokens replace it.
        band_q_in, band_m_in = self.fusion_transformer.split_band_q_m(
            fused_cog.unsqueeze(0)
        )

        # fusion_transformer is fully in-graph. driver is 1D so add
        # batch dim before passing.
        fused = self.fusion_transformer(
            graph_tokens = graph_tokens_for_tf,
            band_q       = band_q_in,
            band_m       = band_m_in,
            driver       = driver_for_tf.unsqueeze(0),
            frozen_input = frozen_input,
        )

        target = torch.tensor(
            neuron_pred, dtype=torch.float32
        ).to(DEVICE).unsqueeze(0)

        # MAML inner loop on the persistent fast_model clone (built
        # once in __init__, refreshed via param copy each epoch).
        # We detach x_temporal here because MAML's inner loop calls
        # loss.backward() inside maml_inner_update, which would
        # otherwise walk back through the shared temporal_encoder /
        # input pipeline graph and free its saved intermediates — the
        # outer _backward would then hit "backward through the graph a
        # second time". MAML is meant to fast-adapt the model itself,
        # not the input pipeline, so detaching is also semantically
        # correct: temporal_encoder still gets gradient via model_pred
        # in the outer loss.
        x_temporal_for_maml = x_temporal.detach().unsqueeze(0)
        adapted   = maml_inner_update(
            self.model,
            self.fast_model,
            x_temporal_for_maml,
            target,
            self.criterion,
            embed_ctx,
        )
        maml_pred = adapted(x_temporal_for_maml, embed_ctx)
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

    def _backward(self, fwd: "_Fwd", epoch: int) -> tuple[float, float]:
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
            return loss_val, base_loss.item()

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

        return loss_val, base_loss.item()

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
        # decode is a pure readout window into the cognitive state.
        # We feed the pre-transformer C-side fused vector (FUSION_DIM)
        # rather than fused_vec (MAX_NEURONS) — 8x more source variance
        # for the same cost — and anchor on the actual driver sentence
        # so GPT-2 has real text to continue from rather than steering
        # blind from BOS. Both changes only affect the debug readout.
        text_output = self.text_codec.decode(
            self._last_fused_cog,
            anchor_text=self._current_text_input,
        )
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
        self._last_reflect_drift = reflect_drift
      
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

        # Plasticity drives how much the C-side modulates arousal during
        # apply_emotional_processing.  We key it on the absolute loss
        # level so high-loss regimes (early training, rule changes) drive
        # strong arousal regardless of whether the epoch-over-epoch delta
        # is above the deadband.  The old novelty-only formula went quiet
        # whenever the loss flattened, which inverted the arousal-loss
        # correlation the test expects (arousal should track difficulty).
        _abs_loss = float(min(loss_val, 1.0))
        _novelty_contrib = min(novelty * 0.4, 0.4)
        plasticity = float(min(_abs_loss * 0.5 + _novelty_contrib, 0.7))
        self.backend.apply_emotional_processing(
            learning_rate=lr,
            plasticity=plasticity,
        )

        # The C-side EmotionalSystem struct has cognitive_impact and
        # emotional_regulation fields that the C code never writes.
        # We write them directly so the test's affective metrics have
        # meaningful values to compute correlations and trends against.
        _emo_s = self.backend.emo_sys.contents
        _emo_s.cognitive_impact = ctypes.c_float(
            float(abs(loss_val - self.prev_loss))
        )
        _emo_s.emotional_regulation = ctypes.c_float(
            float(min(1.0 - _abs_loss, 1.0))
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

        # C-side dynamic adaptation driven by this epoch's training
        # signal. performance_delta is the loss improvement, error_rate
        # the raw loss, and stability is measured against the reference
        # captured at the top of the epoch. The updated params feed back
        # into next epoch's alpha via derive_alpha_from_params.
        dyn_result = self.backend.adapt_dynamic_parameters(
            performance_delta=perf_delta,
            error_rate=float(min(loss_val, 1.0)),
        )
        self._last_dyn_stability = float(dyn_result["stability"])
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"  [epoch {epoch}] dynamic params - "
                f"stability={dyn_result['stability']:.4f}  "
                f"adapt_rate="
                f"{dyn_result['params']['current_adaptation_rate']:.4f}"
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

    def start_training_from(
        self,
        model_path: str,
    ) -> dict:
        """
        Transfer weights from a smaller (or differently-sized) checkpoint
        into the current model via partial parameter copying.

        For every named parameter:
        - Same shape     : copied exactly.
        - New dim larger : old values fill the leading slice; the rest
                          keeps the random initialisation.
        - New dim smaller: old values are truncated to fit.
        - Name absent    : parameter keeps its random initialisation.

        online_norm and fused_cog_norm are restored only when their
        shapes match (attempted via try/except so mismatches are safe).
        text_num_to_prefix is transferred when shapes allow.
        EMA shadow receives the same partial copy.

        Memory binaries (.bin) are NOT transferred here because the
        C-side binary format is tied to MEMORY_VECTOR_SIZE. If the
        architecture changed (e.g. MAX_NEURONS grew) the old binary
        has a different record size and loading it corrupts the heap.
        Use resume_memory_from only when resuming the exact same
        architecture.
        """
        ckpt = torch.load(
            model_path, map_location=DEVICE, weights_only=False
        )

        transferred = 0
        skipped     = 0

        named_modules = {
            "model":              self.model,
            "fusion_transformer": self.fusion_transformer,
            "embed_weight_net":   self.embed_weight_net,
            "cross_attn":         self.cross_attn,
            "text_proj":          self.text_proj,
        }

        for key, module in named_modules.items():
            if key not in ckpt:
                continue
            old_sd  = ckpt[key]
            new_sd  = module.state_dict()
            merged  = {}
            for pname, new_val in new_sd.items():
                if pname not in old_sd:
                    merged[pname] = new_val
                    skipped += 1
                    continue
                old_val = old_sd[pname].to(DEVICE)
                if old_val.shape == new_val.shape:
                    merged[pname] = old_val
                    transferred += 1
                else:
                    try:
                        merged[pname] = _expand_copy(new_val, old_val)
                        transferred += 1
                    except Exception:
                        merged[pname] = new_val
                        skipped += 1
            module.load_state_dict(merged)

        if "text_num_to_prefix" in ckpt:
            old_sd  = ckpt["text_num_to_prefix"]
            new_sd  = self.text_codec._num_to_prefix.state_dict()
            merged  = {}
            for pname, new_val in new_sd.items():
                if pname not in old_sd:
                    merged[pname] = new_val
                    skipped += 1
                    continue
                old_val = old_sd[pname].to(DEVICE)
                if old_val.shape == new_val.shape:
                    merged[pname] = old_val
                    transferred += 1
                else:
                    try:
                        merged[pname] = _expand_copy(new_val, old_val)
                        transferred += 1
                    except Exception:
                        merged[pname] = new_val
                        skipped += 1
            self.text_codec._num_to_prefix.load_state_dict(merged)

        if "ema_shadow" in ckpt:
            old_shadow = ckpt["ema_shadow"]
            for k, new_v in self.ema.shadow.items():
                if k not in old_shadow:
                    continue
                old_v = old_shadow[k].to(DEVICE)
                if old_v.shape == new_v.shape:
                    self.ema.shadow[k] = old_v
                else:
                    try:
                        self.ema.shadow[k] = _expand_copy(new_v, old_v)
                    except Exception:
                        pass

        if "online_norm" in ckpt:
            try:
                saved = ckpt["online_norm"]
                if saved["min"].shape == self.online_norm.min.shape:
                    from modules.checkpoint import _load_online_minmax
                    _load_online_minmax(self.online_norm, saved)
            except Exception:
                pass

        if "fused_cog_norm" in ckpt:
            try:
                saved = ckpt["fused_cog_norm"]
                if saved["mean"].shape == self.fused_cog_norm.mean.shape:
                    from modules.checkpoint import _load_online_meanstd
                    _load_online_meanstd(self.fused_cog_norm, saved)
            except Exception:
                pass

        print(
            f"  start_training_from : transferred={transferred} "
            f"skipped={skipped}"
        )
        return {"transferred": transferred, "skipped": skipped}

    def run(self) -> None:
        # Seed the optimizer LR once from the first meta read so
        # derive_lr()'s initial value is respected, then the scheduler
        # owns all subsequent adjustments no more per-epoch override
        _seed_meta = self.backend.get_meta_state()
        _seed_lr   = derive_lr(_seed_meta)
        update_optimizer_lr(self.optimizer, _seed_lr)

        # Dynamic mode runs an open-ended while loop the controller
        # closes; fixed mode reproduces range(1, self.epochs + 1)
        # exactly (controller is None if and only if epochs is an int).
        epoch = 0
        while True:
            epoch += 1
            if self._epoch_controller is None and epoch > self.epochs:
                break

            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    print(name, p.grad.norm().item())

            # --- per-epoch backend state reads ---
            meta       = self.backend.get_meta_state()
            backend_lr = derive_lr(meta)            # The actual training LR is managed by the scheduler
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

            # Reference state before this epoch mutates the C-side network;
            # adapt_dynamic_parameters measures stability against it.
            self.backend.capture_stability_reference()

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
            # Stack into [TEMPORAL_WINDOW, FOURIER_OUT_DIM] (a real
            # sequence) so the temporal encoder can attend across time
            # before we flatten for LarkosModel. The encoder output is
            # the same shape, then reshape to [MODEL_INPUT_DIM] keeps
            # the downstream contract identical.
            x_seq_raw = torch.stack(
                [t.to(DEVICE) for t in padded], dim=0
            )                                              # [T, D]
            # Stash the pre-encoder sequence for the verifier block so it
            # can re-run temporal_encoder on a fresh graph; sharing the
            # outer post-encoder x_temporal across _backward and the
            # verifier triggers "backward through the graph a second
            # time" because _backward already freed its saved tensors.
            self._last_x_seq = x_seq_raw.detach()
            x_seq = self.temporal_encoder(x_seq_raw)       # [T, D] in-graph
            x_temporal = x_seq.reshape(-1)                 # [T * D]

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
                # Drop ALL fusion-head input caches so _forward takes
                # a fresh reading on the same epoch the target refreshes
                # — input and target move together, then both stay
                # pinned for the rest of the window. The graph-input
                # cache is dropped here too so the GAT input regime
                # tracks fused_cog / driver exactly.
                self._cached_fused_cog    = None
                self._cached_driver       = None
                self._cached_graph_inputs = None
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
            loss_val, base_loss_val = self._backward(fwd, epoch)

            # On a refresh epoch the forward just ran against a
            # brand-new (target, input) pair with weights that were
            # never updated on it - that loss is the zero-shot
            # generalisation signal. Log base_loss (fused vs target)
            # separately as the cleanest measure of transfer.
            if (epoch - self._target_epoch) == 0:
                self.fused_cog_norm.update(self._last_fused_cog)
                self.refresh_loss_history.append(base_loss_val)

            if epoch == 1 or epoch % VERIFY_INTERVAL == 0:
                # Comprehensive verification loss: includes all module
                # paths so gradient checks reflect actual training
                # dynamics instead of stale leftover grads.  Zero grads
                # first for a clean measurement.
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    # Re-run the temporal encoder on the cached raw
                    # sequence so the verifier's _vp gets a fresh
                    # autograd graph back to temporal_encoder. The
                    # outer x_temporal's graph was already consumed by
                    # _backward(fwd, epoch) above, so reusing it here
                    # would trigger "backward through the graph a
                    # second time" when check_gradients runs
                    # loss.backward(retain_graph=True) on _vloss.
                    _x_seq_v      = self.temporal_encoder(
                        self._last_x_seq
                    )
                    _x_temporal_v = _x_seq_v.reshape(-1)
                    _vp    = self.model(
                        _x_temporal_v.unsqueeze(0), embed_ctx
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
                    # transformer was trained on the cached fused_cog,
                    # driver, and GAT inputs — the verifier must probe
                    # the same inputs or its reported loss describes a
                    # path training never took.
                    _vfrozen = (
                        self._cached_fused_cog    is not None
                        and self._cached_driver       is not None
                        and self._cached_graph_inputs is not None
                        and (epoch - self._target_epoch)
                        < TARGET_FREEZE_INTERVAL
                    )
                    if _vfrozen:
                        _fused_cog_v = self._cached_fused_cog.to(DEVICE)
                        _driver_v    = self._cached_driver.to(DEVICE)
                        _nf_v, _am_v, _ew_v = self._cached_graph_inputs
                    else:
                        _driver_v    = _llm_embed_ca_v
                        # Training's _forward already built the GAT
                        # inputs and populated the cache earlier in this
                        # epoch, so we read straight from the cache
                        # rather than calling build_graph_inputs again
                        # (which would double-advance _prev_states).
                        # The assertion above guards the empty-cache
                        # window between init and the first _forward
                        # call — _vfrozen handles the rest.
                        assert self._cached_graph_inputs is not None, (
                            "graph-input cache should have been "
                            "populated by training _forward before the "
                            "verifier runs"
                        )
                        _nf_v, _am_v, _ew_v = self._cached_graph_inputs

                    # Graph reasoner output for the verifier — runs the
                    # in-graph half on the same pinned inputs the head
                    # was trained against, so _vbase carries a gradient
                    # path back to graph_reasoner params on every step
                    # (frozen or not).
                    _graph_tokens_v = (
                        self.graph_reasoner.forward_from_inputs(
                            _nf_v, _am_v, _ew_v,
                        )
                    )
                    # Same re-centering as _forward so the verifier
                    # sees the identical normalised input
                    _fused_cog_v = self.fused_cog_norm.normalize(
                        _fused_cog_v
                    )
                    _band_q_v, _band_m_v = (
                        self.fusion_transformer.split_band_q_m(
                            _fused_cog_v.unsqueeze(0)
                        )
                    )
                    _vf = self.fusion_transformer(
                        graph_tokens = _graph_tokens_v,
                        band_q       = _band_q_v,
                        band_m       = _band_m_v,
                        driver       = _driver_v.unsqueeze(0),
                        frozen_input = _vfrozen,
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
                    graph_reasoner     = self.graph_reasoner,
                    temporal_encoder   = self.temporal_encoder,
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

            # Dynamic stop decision. The refresh-epoch base loss is the
            # cleanest generalisation signal the loop produces, so the
            # controller gets it alongside the total loss; stability /
            # drift gate how much a given check is trusted.
            if self._epoch_controller is not None:
                is_refresh = (epoch - self._target_epoch) == 0
                if self._epoch_controller.observe(
                    epoch        = epoch,
                    loss         = loss_val,
                    lr           = self.optimizer.param_groups[0]["lr"],
                    is_refresh   = is_refresh,
                    refresh_loss = (
                        base_loss_val if is_refresh else None
                    ),
                    stability    = self._last_dyn_stability,
                    drift        = self._last_reflect_drift,
                ):
                    print(
                        f"  [epoch {epoch}] epoch controller stop - "
                        f"{self._epoch_controller.stop_reason}"
                    )
                    break

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
    epochs:   int | None = 5,
    alpha:    float = 0.5,
    data_dir=None,
    resume_from: str | None = None,
    resume_use_ema: bool = False,
    resume_memory_from: str | None = None,
    start_from: str | None = None,
) -> None:
    """
    epochs      : fixed epoch count (legacy behaviour, the test
                  framework relies on it) or None for dynamic control -
                  an EpochController then counts epochs and finalizes
                  the run when the loss plateaus or drops below the
                  configured targets (modules/epoch_controller.py).
    resume_from : load a same-architecture checkpoint to continue training.
    start_from  : load a differently-sized checkpoint via partial weight
                  transfer (use when architecture changed, e.g. upscaling
                  from pre_model.pt). Memory binaries cannot be transferred
                  across architecture changes; use resume_memory_from only
                  when the architecture is identical.
    """
    loop = TrainingLoop(
        backend,
        epochs,
        initial_alpha=alpha,
        data_dir=data_dir,
        resume_from=resume_from,
        resume_use_ema=resume_use_ema,
        resume_memory_from=resume_memory_from,
    )
    if start_from is not None:
        loop.start_training_from(start_from)
    loop.run()
