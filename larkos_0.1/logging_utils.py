import numpy as np


def log_epoch(
    epoch:     int,
    input_t:   np.ndarray,
    model_p:   np.ndarray,
    neuron_p:  np.ndarray,
    fused_p:   np.ndarray,
    loss:      float,
    lr:        float,
    alpha:     float,
) -> None:
    sep = "-" * 48
    print(sep)
    print(f"Epoch {epoch:>4}")
    print(f"  input   : {np.round(input_t,  4)}")
    print(f"  model   : {np.round(model_p,  4)}")
    print(f"  neuron  : {np.round(neuron_p, 4)}")
    print(f"  fused   : {np.round(fused_p,  4)}")
    print(f"  loss    : {loss:.6f}")
    print(f"  lr      : {lr:.6f}")
    print(f"  alpha   : {alpha:.4f}")


def log_context(ctx: dict) -> None:
    nodes    = ctx.get("total_nodes", "n/a")
    vec_size = ctx.get("vector_size", "n/a")
    decay    = ctx.get("decay_rate",  "n/a")
    print(
        f"  context nodes={nodes}  vec_size={vec_size}"
        f"  decay={decay}"
    )


def log_history(history: list) -> None:
    _wv_len = len(history)
    print(
        f"  network history steps captured : {_wv_len}"
    )


def log_memory(mem: dict) -> None:
    used = mem.get("size",     "n/a")
    cap  = mem.get("capacity", "n/a")
    st   = mem.get("short_term",  {}).get("size", "n/a")
    mt   = mem.get("medium_term", {}).get("size", "n/a")
    lt   = mem.get("long_term",   {}).get("size", "n/a")
    print(
        f"  memory : {used}/{cap} slots"
        f"  (st={st} mt={mt} lt={lt})"
    )
