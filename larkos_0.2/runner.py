import torch

from modules.backend_state import BackendState
from modules.config import (
    DEVICE, INPUT_SIZE, INTERNAL_DIM, MAX_NEURONS, FUSION_DIM,
    FOURIER_OUT_DIM, TEMPORAL_WINDOW
)
from collections import deque

from modules.model import LarkosModel, EMAWrapper
from modules.training import (
    _FusionTransformerHead, _EmbedWeightNet, _InputCrossAttention,
    _OnlineMinMax, _OnlineMeanStd, _TextCodec,
    _build_embed_ctx, fourier_encode,
)
from modules.fusion_mechanism.fusion import cognitive_fuse
from modules.checkpoint import load_checkpoint



class LarkosRunner:
    """
    Loads a saved model and runs forward-only inference. NOT the
    training loop it deliberately reconstructs only the modules on the
    forward path and exposes the same attribute names load_checkpoint
    writes into, so the checkpoint loader can target it unchanged.

    The C side still has to be live: cognitive_fuse reads neurons,
    mem_state and default_weights from a BackendState every step, so
    the runner loads the memory system and the saved network states into
    the backend before stepping. Without that the fusion input would be whatever
    empty backend ships with and the readout would not reflect the trained dynamics.
    """

    def __init__(
        self,
        backend:    "BackendState",
        ckpt_path:  str  = "larkos_model.pt",
        mem_path:   str  = "memory.bin",
        use_ema:    bool = True,
    ) -> None:
        self.backend = backend

        self.model = LarkosModel().to(DEVICE)
        self.ema   = EMAWrapper(self.model)

        self.fusion_transformer = _FusionTransformerHead(
            output_dim=self.model.output_dim
        ).to(DEVICE)
        self.embed_weight_net = _EmbedWeightNet(
            INPUT_SIZE, INTERNAL_DIM
        ).to(DEVICE)
        self.cross_attn = _InputCrossAttention(
            INPUT_SIZE, INTERNAL_DIM
        ).to(DEVICE)
        self.text_proj = torch.nn.Linear(
            self.text_proj_in_dim(), INTERNAL_DIM
        ).to(DEVICE)

        self.online_norm    = _OnlineMinMax(INPUT_SIZE)
        self.fused_cog_norm = _OnlineMeanStd(FUSION_DIM)
        self.text_codec     = _TextCodec(DEVICE)

        # Same attribute names the loop uses so load_checkpoint can
        # restore them onto us without knowing we are the runner.
        self.loss_history: list[float] = []
        self.input_history: deque[torch.Tensor] = deque(
            maxlen=TEMPORAL_WINDOW
        )
        self._cached_target = None
        self._target_epoch  = 0
        self._cached_fused_cog: torch.Tensor | None = None
        self._sample_pool: list[str] = []
        self._sample_pool_idx = 0
        self._current_text_input = ""
        self.text_encoding = torch.zeros(
            self.text_proj_in_dim()
        ).to(DEVICE)

        load_result = load_checkpoint(
            self, ckpt_path, use_ema=use_ema
        )
        print(f"  model load : {load_result}")

        # C-side state: memory first, then the saved network states so
        # cognitive_fuse sees the trained neuron / memory landscape.
        mem_result = self.backend.load_memory(mem_path)
        print(f"  memory load: {mem_result}")
        net_result = self.backend.load_network_states()
        print(f"  net load   : {net_result}")

        self._eval()

    def text_proj_in_dim(self) -> int:
        # text_proj maps the GPT-2 hidden readout into INTERNAL_DIM;
        # its input width is the codec's hidden size. Imported lazily
        # from config to avoid hard-coding the GPT2_HIDDEN constant
        # twice across files, function exists for backwards
        # compatibility and for readability.
        from modules.config import GPT2_HIDDEN
        return GPT2_HIDDEN

    def _eval(self) -> None:
        for module in (
            self.model, self.fusion_transformer,
            self.embed_weight_net, self.cross_attn, self.text_proj,
        ):
            module.eval()

    def _build_default_weights(
        self,
        neurons:      dict,
        input_tensor,
    ) -> torch.Tensor:
        # Same flattened state/output/input layout cognitive_fuse
        # expects in the training loop nothing inference-specific here,
        # we just rebuild it from the live backend each step.
        dw_state = [
            float(
                neurons.get(f"neuron_{i}", {}).get("state", 0.0)
            )
            for i in range(MAX_NEURONS)
        ]
        dw_output = [
            float(
                neurons.get(f"neuron_{i}", {}).get("output", 0.0)
            )
            for i in range(MAX_NEURONS)
        ]
        dw_input = input_tensor.tolist()
        return torch.tensor(
            dw_state + dw_output + dw_input,
            dtype=torch.float32,
        ).to(DEVICE)

    @torch.no_grad()
    def step(
        self,
        text_input:       str | None = None,
        mem_weight_ratio: float = 0.5,
        alpha:            float = 0.5,
    ) -> dict:
        """
        Single forward pass mirroring TrainingLoop._forward minus the
        MC probes, the MAML inner loop and aux_proj none of which sit
        on the output path. Returns the fused vector, the model's raw
        prediction and the decoded text readout.

        text_input lets the caller drive the model with a sentence;
        when None the restored current_text_input is reused so a fresh
        runner reproduces the last driver the checkpoint trained on.
        """
        if text_input is not None:
            self._current_text_input = text_input
            self.text_encoding = (
                self.text_codec.encode(text_input).detach()
            )

        ctx          = self.backend.get_context_state()
        embed_ctx    = _build_embed_ctx(ctx)
        neurons      = self.backend.get_neurons()
        mem_state    = self.backend.get_memory_state()
        input_tensor = self.backend.get_input_tensor()
        print(
            f"  step: input_tensor "
            f"min={float(input_tensor.min()):.4f} "
            f"max={float(input_tensor.max()):.4f} "
            f"mean={float(input_tensor.mean()):.4f}"
        )

        x_raw = torch.tensor(
            input_tensor, dtype=torch.float32
        ).to(DEVICE)
        self.online_norm.update(x_raw)
        x_norm    = self.online_norm.normalize(x_raw)
        x_fourier = fourier_encode(x_norm)

        self.input_history.append(x_fourier.detach().cpu())
        padded = list(self.input_history)
        while len(padded) < TEMPORAL_WINDOW:
            padded.insert(0, torch.zeros(FOURIER_OUT_DIM))
        x_temporal = torch.cat(
            [t.to(DEVICE) for t in padded], dim=-1
        )
        print(
            f"  step: x_temporal shape={tuple(x_temporal.shape)} "
            f"finite={bool(torch.isfinite(x_temporal).all())}"
        )

        default_weights = self._build_default_weights(
            neurons, input_tensor
        )

        model_pred = self.model(x_temporal.unsqueeze(0), embed_ctx)
        _mp = model_pred.detach()
        print(
            f"  step: model_pred "
            f"min={float(_mp.min()):.4f} "
            f"max={float(_mp.max()):.4f} "
            f"finite={bool(torch.isfinite(_mp).all())}"
        )

        llm_embed_raw = model_pred.squeeze(0)
        llm_embed     = llm_embed_raw[:INTERNAL_DIM]
        embed_gate    = self.embed_weight_net(x_norm)
        llm_embed_g   = llm_embed * embed_gate
        llm_embed_ca  = self.cross_attn(x_norm, llm_embed_g)

        text_proj_out = self.text_proj(
            self.text_encoding.to(DEVICE)
        )
        llm_embed_ca = (
            llm_embed_ca
            + text_proj_out[:llm_embed_ca.shape[-1]]
        )

        print(f"  step: calling cognitive_fuse alpha={alpha:.4f}")
        fused_cog_raw = cognitive_fuse(
            llm_embed        = llm_embed_ca.detach(),
            neurons          = neurons,
            mem_state        = mem_state,
            default_weights  = default_weights,
            mem_weight_ratio = mem_weight_ratio,
            context_factor   = alpha,
            text_embed       = self.text_encoding.detach(),
        )
        print(
            f"  step: fused_cog_raw "
            f"min={float(fused_cog_raw.min()):.4f} "
            f"max={float(fused_cog_raw.max()):.4f} "
            f"finite={bool(torch.isfinite(fused_cog_raw).all())}"
        )

        # The transformer was trained with its input frozen across a
        # window; at inference there is no training schedule, so we
        # always treat the input as live (frozen_input=False) and just
        # normalise the fresh C-side reading the way training did on a
        # refresh epoch.
        fused_cog = self.fused_cog_norm.normalize(fused_cog_raw)
        fused = self.fusion_transformer(
            fused_cog.unsqueeze(0), frozen_input=False
        )

        fused_vec = fused.squeeze(0).detach()
        _nonfinite = int((~torch.isfinite(fused_vec)).sum())
        print(
            f"  step: fused_vec "
            f"min={float(fused_vec.min()):.4f} "
            f"max={float(fused_vec.max()):.4f} "
            f"nonfinite={_nonfinite}"
        )
        if not torch.isfinite(fused_vec).all():
            fused_vec = torch.nan_to_num(
                fused_vec, nan=0.0, posinf=1.0, neginf=-1.0
            )
        # Decode reads from the C-side pre-transformer fused vector
        # (FUSION_DIM) — same widening as the training loop after the
        # bridge source dim moved from MAX_NEURONS to FUSION_DIM. The
        # current driver sentence anchors the generation so GPT-2 has
        # real context to continue from.
        fused_cog_for_decode = fused_cog_raw.detach()
        if not torch.isfinite(fused_cog_for_decode).all():
            fused_cog_for_decode = torch.nan_to_num(
                fused_cog_for_decode,
                nan=0.0, posinf=1.0, neginf=-1.0,
            )
        text_output = self.text_codec.decode(
            fused_cog_for_decode,
            anchor_text=self._current_text_input or None,
        )

        return {
            "fused":       fused_vec.cpu().numpy(),
            "model_pred":  model_pred.squeeze(0).cpu().numpy(),
            "text_input":  self._current_text_input,
            "text_output": text_output,
        }


def run_model(
    backend:   "BackendState",
    ckpt_path: str  = "larkos_model.pt",
    mem_path:  str  = "memory.bin",
    use_ema:   bool = True,
    steps:     int  = 1,
    text_input: str | None = None,
) -> list[dict]:
    runner  = LarkosRunner(
        backend, ckpt_path, mem_path, use_ema=use_ema
    )
    results = []
    for _ in range(steps):
        out = runner.step(text_input=text_input)
        print(f"  text_output: {out['text_output']}")
        results.append(out)
    return results
