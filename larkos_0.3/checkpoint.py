import torch

from modules.config import DEVICE, CHECKPOINT_VERSION, FUSION_DIM


def _serialise_online_minmax(norm) -> dict:
    return {
        "min":  norm.min.clone(),
        "max":  norm.max.clone(),
        "seen": norm._seen.clone(),
        "mom":  norm._mom,
    }


def _load_online_minmax(norm, state: dict) -> None:
    # Force CPU: update() keeps its running stats on CPU (it does
    # x.detach().cpu() every call), but torch.load's map_location can
    # land these on CUDA, which then mismatches inside update(). We
    # restore the CPU invariant the class was built around.
    norm.min   = state["min"].clone().cpu()
    norm.max   = state["max"].clone().cpu()
    norm._seen = state["seen"].clone().cpu()
    norm._mom  = state["mom"]


def _serialise_online_meanstd(norm) -> dict:
    return {
        "mean":  norm.mean.clone(),
        "var":   norm.var.clone(),
        "seen":  norm._seen,
        "mom":   norm._mom,
        "floor": norm._floor,
    }


def _load_online_meanstd(norm, state: dict) -> None:
    # Same CPU invariant as _load_online_minmax: update() runs on CPU,
    # so the restored mean/var must be CPU too or normalize()/update()
    # hit a cross-device op.
    norm.mean   = state["mean"].clone().cpu()
    norm.var    = state["var"].clone().cpu()
    norm._seen  = state["seen"]
    norm._mom   = state["mom"]
    norm._floor = state["floor"]


def _maybe_tensor(x):
    # The freeze-window cache holds either a detached tensor or None.
    # Kept on CPU in the checkpoint so a GPU-trained model reloads on
    # a CPU box without a device mismatch; rehydrated to DEVICE on load.
    if x is None:
        return None
    return x.detach().cpu()


def save_checkpoint(loop, filename: str = "larkos_model.pt") -> dict:
    """
    Inference-scoped checkpoint: only the modules and running state the
    forward path actually touches. The optimizer, scheduler, aux_proj
    and the training-only scalars are intentionally NOT saved here they
    play no part in a forward pass and would only bloat the file.

    Forward path is:
        model -> embed_weight_net -> cross_attn -> text_proj
              -> cognitive_fuse (C side, no weights)
              -> fused_cog_norm -> fusion_transformer

    Both EMA and live model weights are stored so the runner can pick
    either via use_ema EMA is only a shadow of model, so model must be
    present regardless; ema is the extra copy, not a replacement.

    The C-side memory system and network states are NOT saved here
    they go through trigger_save_memory / trigger_save_network_states.
    This file is purely the torch-side state.
    """
    ckpt = {
        "version":     CHECKPOINT_VERSION,
        "epochs_done": len(loop.loss_history),

        "model":              loop.model.state_dict(),
        "fusion_transformer": loop.fusion_transformer.state_dict(),
        "graph_reasoner":     loop.graph_reasoner.state_dict(),
        "temporal_encoder":   loop.temporal_encoder.state_dict(),
        "embed_weight_net":   loop.embed_weight_net.state_dict(),
        "cross_attn":         loop.cross_attn.state_dict(),
        "text_proj":          loop.text_proj.state_dict(),

        # shadow is already a full state_dict deepcopy keyed exactly
        # like model.state_dict(), so we save it as-is moving the
        # tensors to CPU so a GPU-trained shadow reloads on a CPU box.
        # wv shadow tag
        "ema_shadow": {
            k: v.detach().cpu()
            for k, v in loop.ema.shadow.items()
        },

        # _num_to_prefix is a fixed RANDOM projection never optimised,
        # never reproducible across runs so it must be persisted or a
        # reloaded model decodes through a different random readout than
        # it trained against.
        "text_num_to_prefix": (
            loop.text_codec._num_to_prefix.state_dict()
        ),

        "online_norm":    _serialise_online_minmax(loop.online_norm),
        "fused_cog_norm": _serialise_online_meanstd(
            loop.fused_cog_norm
        ),

        # Freeze-window bookkeeping the forward path reads these to
        # decide whether the transformer input is pinned, so the runner
        # must start from the same window the save left off in.
        "cached_target": (
            None if loop._cached_target is None
            else loop._cached_target.copy()
        ),
        "target_epoch":     loop._target_epoch,
        "cached_fused_cog": _maybe_tensor(loop._cached_fused_cog),
        # New freeze-cache entry introduced by the graph-reasoning
        # fusion head — same shape contract as cached_fused_cog (None
        # or a detached CPU tensor) so old checkpoints loaded under the
        # new code see None and start a fresh freeze window.
        # graph_tokens are NOT cached: the GAT needs gradient through
        # the freeze window or it dies (see TrainingLoop.__init__).
        "cached_driver": _maybe_tensor(
            getattr(loop, "_cached_driver", None)
        ),

        # Temporal input window feeds x_temporal; without it the first
        # few forward passes run on zero-padded history instead of the
        # real tail the model last saw.
        "input_history": [t.clone() for t in loop.input_history],

        # Text driver state so the runner continues from the same
        # sample rather than re-seeding the pool from scratch.
        "sample_pool":        list(loop._sample_pool),
        "sample_pool_idx":    loop._sample_pool_idx,
        "current_text_input": loop._current_text_input,
        "text_encoding":      loop.text_encoding.detach().cpu(),
    }

    torch.save(ckpt, filename)
    return {
        "status":      "saved",
        "file":        filename,
        "epochs_done": ckpt["epochs_done"],
    }


def load_checkpoint(
    loop,
    filename: str = "larkos_model.pt",
    use_ema:  bool = True,
) -> dict:
    """
    Restores an inference checkpoint onto an already-built loop-shaped
    object (TrainingLoop or LarkosRunner) that exposes the same module
    attributes. With use_ema the EMA shadow is overlaid onto model
    after loading, so the live forward pass runs on the smoothed
    weights; model is loaded first either way since EMA only shadows it.
    """
    # weights_only=False because the checkpoint holds a numpy
    # cached_target plus python lists / scalars, not just tensors. The
    # file is self-produced locally so the trusted-source caveat the
    # torch.load default warns about does not apply here.
    ckpt = torch.load(
        filename, map_location=DEVICE, weights_only=False
    )

    # strict=False so an older checkpoint whose module structure has
    # since changed (e.g. removed NumericTokenizer.in_proj) still loads
    # — the obsolete keys are dropped and any new keys keep their fresh
    # init. The missing/unexpected sets are surfaced for visibility.
    _missing, _unexpected = loop.model.load_state_dict(
        ckpt["model"], strict=False
    )
    if _missing or _unexpected:
        print(
            f"  model load: missing={list(_missing)} "
            f"unexpected={list(_unexpected)}"
        )
    # strict=False so an old pre-graph-head checkpoint whose
    # fusion_transformer carried proj_q/proj_n/proj_m/stream_embed (the
    # 3-stream layout) loads cleanly onto the new mixed-token head —
    # the obsolete keys are dropped and the new layers (proj_driver,
    # token_type_embed, pool_query, etc.) keep their fresh init. Same
    # idea downstream for graph_reasoner: an old checkpoint won't have
    # the key at all, so the freshly init'd reasoner runs as-is.
    _ft_missing, _ft_unexpected = loop.fusion_transformer.load_state_dict(
        ckpt["fusion_transformer"], strict=False
    )
    if _ft_missing or _ft_unexpected:
        print(
            f"  fusion_transformer load: missing={list(_ft_missing)} "
            f"unexpected={list(_ft_unexpected)}"
        )
    _gr_state = ckpt.get("graph_reasoner")
    if _gr_state is not None:
        # strict=False so a checkpoint from before _D_NODE grew (5→8)
        # or before neuron_embed existed still loads — the new tensors
        # stay at their fresh init, which is the safest default when
        # the shape contract changes.
        _gr_missing, _gr_unexpected = loop.graph_reasoner.load_state_dict(
            _gr_state, strict=False
        )
        if _gr_missing or _gr_unexpected:
            print(
                f"  graph_reasoner load: missing={list(_gr_missing)} "
                f"unexpected={list(_gr_unexpected)}"
            )
    else:
        print(
            "  graph_reasoner load: no entry in checkpoint — "
            "keeping fresh init"
        )
    # temporal_encoder added when TEMPORAL_WINDOW grew beyond 2. Old
    # checkpoints predate the encoder entirely, and any checkpoint
    # saved with a different TEMPORAL_WINDOW will have a positional
    # embedding tensor of a different shape. strict=False tolerates
    # both — log what was missing / unexpected so a config change is
    # visible at load time.
    _te_state = ckpt.get("temporal_encoder")
    if _te_state is not None:
        _te_missing, _te_unexpected = loop.temporal_encoder.load_state_dict(
            _te_state, strict=False
        )
        if _te_missing or _te_unexpected:
            print(
                f"  temporal_encoder load: missing={list(_te_missing)} "
                f"unexpected={list(_te_unexpected)}"
            )
    else:
        print(
            "  temporal_encoder load: no entry in checkpoint — "
            "keeping fresh init"
        )
    loop.embed_weight_net.load_state_dict(ckpt["embed_weight_net"])
    loop.cross_attn.load_state_dict(ckpt["cross_attn"])
    loop.text_proj.load_state_dict(ckpt["text_proj"])

    # shadow is a plain state_dict, not a module assign the restored
    # dict straight back, moving each tensor onto DEVICE. Filter to keys
    # the current model actually exposes so dropped keys (e.g. tokenizer
    # in_proj) don't poison apply_shadow's load_state_dict downstream.
    _live_keys = set(loop.model.state_dict().keys())
    loop.ema.shadow = {
        k: v.to(DEVICE)
        for k, v in ckpt["ema_shadow"].items()
        if k in _live_keys
    }
    # _num_to_prefix is a frozen random projection. If its source dim
    # changed (e.g. MAX_NEURONS=8 -> FUSION_DIM=64) the saved weights
    # can't load and the new random init is the only valid value
    # anyway — there's no meaningful transfer for a non-trained head.
    try:
        loop.text_codec._num_to_prefix.load_state_dict(
            ckpt["text_num_to_prefix"]
        )
    except RuntimeError as e:
        print(
            f"  text_num_to_prefix load skipped ({e.__class__.__name__}): "
            f"keeping fresh random projection"
        )

    _load_online_minmax(loop.online_norm, ckpt["online_norm"])
    # FUSION_DIM shrank from 1024 to 683 when BAND_N was retired from
    # cognitive_fuse, so a checkpoint saved under the old layout has
    # fused_cog_norm.mean/var sized to the old dim. The running stats
    # are EMA estimates that will rebuild from the next refresh-epoch's
    # _last_fused_cog anyway, so dropping the stale tensors is safe —
    # the freshly init'd _OnlineMeanStd starts seeded and gets seen=False.
    _fcn_state = ckpt["fused_cog_norm"]
    if _fcn_state["mean"].shape == loop.fused_cog_norm.mean.shape:
        _load_online_meanstd(loop.fused_cog_norm, _fcn_state)
    else:
        print(
            "  fused_cog_norm load skipped: dim mismatch "
            f"(ckpt {tuple(_fcn_state['mean'].shape)} vs live "
            f"{tuple(loop.fused_cog_norm.mean.shape)}) — "
            "stats will re-seed from live data"
        )

    loop._cached_target = ckpt["cached_target"]
    loop._target_epoch  = ckpt["target_epoch"]
    _cfc = ckpt["cached_fused_cog"]
    # FUSION_DIM may have changed since this checkpoint was saved (the
    # post-BAND_N-retirement layout is 683-d, the old layout was 1024).
    # The cache feeds the fusion head's split_band_q_m which slices
    # FUSION_DIM-shaped vectors, so reusing an old-dim cache would
    # mis-slice. Drop the stale cache and let the next refresh epoch
    # rebuild it under the new layout.
    if _cfc is not None and _cfc.shape[-1] != FUSION_DIM:
        print(
            "  cached_fused_cog skipped: dim mismatch "
            f"(ckpt {tuple(_cfc.shape)} vs FUSION_DIM={FUSION_DIM}) — "
            "rebuilding on next refresh epoch"
        )
        _cfc = None
    loop._cached_fused_cog = (
        None if _cfc is None else _cfc.to(DEVICE)
    )
    # Old checkpoints don't carry cached_driver — .get() returns None
    # and the next refresh epoch will rebuild it. Either way the loop /
    # runner already declared the attribute in __init__. The legacy
    # cached_graph_tokens key (written by an earlier version of this
    # head) is intentionally ignored: graph_tokens are no longer cached
    # because doing so cut gradient to the GAT every frozen epoch.
    _cdr = ckpt.get("cached_driver")
    loop._cached_driver = (
        None if _cdr is None else _cdr.to(DEVICE)
    )

    loop.input_history.clear()
    for t in ckpt["input_history"]:
        loop.input_history.append(t)

    # Don't restore the sample pool / text input from the old run —
    # the user may have changed data_dir, and a stale pool of old-domain
    # samples would prevent the new pipeline from ever being consulted.
    # The caller must re-seed these from the new data pipeline.
    loop._sample_pool.clear()
    loop._sample_pool_idx = 0

    if use_ema:
        # apply_shadow loads the shadow dict onto model (and stashes a
        # backup internally). The runner never trains, so we do not
        # restore() the keys match model exactly so this is a clean
        # full overlay onto the freshly loaded weights.
        loop.ema.apply_shadow(loop.model)

    for module in (
        loop.model, loop.temporal_encoder, loop.graph_reasoner,
        loop.fusion_transformer, loop.embed_weight_net,
        loop.cross_attn, loop.text_proj,
    ):
        module.to(DEVICE)

    return {
        "status":      "loaded",
        "file":        filename,
        "version":     ckpt.get("version", "?"),
        "epochs_done": ckpt.get("epochs_done", "?"),
        "use_ema":     use_ema,
    }
