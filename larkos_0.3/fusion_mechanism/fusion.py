import ctypes
import pathlib
import torch

from ctypes import (
    CDLL, POINTER,
    c_float, c_int, c_uint,
)

from modules.config import DEVICE, MAX_CONNECTIONS, MEMORY_VECTOR_SIZE

_LIB_PATH = pathlib.Path(__file__).parent / "libfusion.so"
_lib: CDLL | None = None


def _load() -> CDLL:
    global _lib
    if _lib is not None:
        return _lib
    lib = CDLL(str(_LIB_PATH))

    lib.cognitive_fuse.restype  = None
    lib.cognitive_fuse.argtypes = [
        POINTER(c_float),   # llm_embed
        c_int,              # llm_dim
        ctypes.c_void_p,    # neurons (opaque ptr, C knows layout)
        c_int,              # n_neurons
        ctypes.c_void_p,    # mem_entries (opaque ptr)
        c_int,              # n_mem
        POINTER(c_float),   # default_weights (MEMORY_VECTOR_SIZE)
        c_float,            # mem_weight_ratio
        c_float,            # context_factor
        POINTER(c_float),   # text_embed (text_dim, query-band inject)
        c_int,              # text_dim
        POINTER(c_float),   # out
    ]

    lib.fusion_dim.restype  = c_int
    lib.fusion_dim.argtypes = []

    _lib = lib
    return lib


class _CNeuron(ctypes.Structure):
    _fields_ = [
        ("state",           c_float),
        ("output",          c_float),
        ("num_connections", c_uint),
        ("layer_id",        c_uint),
        ("connections",     c_uint  * MAX_CONNECTIONS),
        ("weights",         c_float * MAX_CONNECTIONS),
    ]


class _CMemEntry(ctypes.Structure):
    _fields_ = [
        ("vector",     c_float * MEMORY_VECTOR_SIZE),
        ("importance", c_float),
        ("timestamp",  c_uint),
    ]


def _neurons_to_c(neurons: dict) -> tuple[ctypes.Array, int]:
    entries = list(neurons.values())
    n       = len(entries)
    arr     = (_CNeuron * max(n, 1))()
    for i, nd in enumerate(entries):
        arr[i].state           = float(nd.get("state",           0.0))
        arr[i].output          = float(nd.get("output",          0.0))
        arr[i].num_connections = int  (nd.get("num_connections",   0))
        arr[i].layer_id        = int  (nd.get("layer_id",          0))
        conns   = nd.get("connections", [])
        weights = nd.get("weights",     [])
        for c in range(MAX_CONNECTIONS):
            arr[i].connections[c] = (
                int(conns[c]) if c < len(conns) else 0
            )
            arr[i].weights[c] = (
                float(weights[c]) if c < len(weights) else 0.0
            )
    return arr, n


def _mem_entries_to_c(
    mem_state: dict,
) -> tuple[ctypes.Array, int]:
    raw: list[dict] = []
    for tier in ("short_term", "medium_term", "long_term"):
        raw.extend(mem_state.get(tier, {}).get("entries", []))

    n   = len(raw)
    arr = (_CMemEntry * max(n, 1))()
    for i, e in enumerate(raw):
        vec = e.get("vector", [])
        for j in range(MEMORY_VECTOR_SIZE):
            arr[i].vector[j] = float(vec[j]) if j < len(vec) else 0.0
        arr[i].importance = float(e.get("importance", 0.0))
        arr[i].timestamp  = int  (e.get("timestamp",  0))
    return arr, n


def cognitive_fuse(
    llm_embed:       torch.Tensor,
    neurons:         dict,
    mem_state:       dict,
    default_weights: torch.Tensor | None = None,
    mem_weight_ratio: float = 1.0,
    context_factor:  float = 0.5,
    text_embed:      torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Projects the LLM embedding, neuron states (including graph
    topology and edge weights) and hierarchical memory into a shared
    FUSION_DIM space, runs attention so the LLM queries its own
    cognitive system, applies adaptive sigmoid gating and
    layer-normalises the result.

    default_weights : 1-D float32 tensor of MEMORY_VECTOR_SIZE (22)
                      representing the current system state used as a
                      prior when blending with stored memory vectors.
                      If None, a zero vector is used (fallback).
    mem_weight_ratio: blend factor in [0,1]. 0 = pure default prior,
                      1 = pure stored memory (original behaviour).
    text_embed      : 1-D float32 tensor of the input-sentence
                      embedding, injected ungated into the query band
                      so the sentence reaches fused regardless of
                      mem_weight_ratio. If None, no text is injected.

    Returns a 1-D float32 tensor of length FUSION_DIM on DEVICE.
    """
    lib = _load()
    dim = lib.fusion_dim()

    embed_np = llm_embed.detach().cpu().float().numpy()
    embed_c  = embed_np.ctypes.data_as(POINTER(c_float))

    n_arr, n_count = _neurons_to_c(neurons)
    m_arr, m_count = _mem_entries_to_c(mem_state)

    if default_weights is not None:
        dw_np = default_weights.detach().cpu().float().numpy()
        dw_c  = dw_np.ctypes.data_as(POINTER(c_float))
    else:
        dw_arr = (c_float * MEMORY_VECTOR_SIZE)()
        dw_c   = dw_arr

    if text_embed is not None:
        txt_np  = text_embed.detach().cpu().float().numpy()
        txt_c   = txt_np.ctypes.data_as(POINTER(c_float))
        txt_dim = len(txt_np)
    else:
        txt_c   = None
        txt_dim = 0

    out_buf = (c_float * dim)()

    lib.cognitive_fuse(
        embed_c,
        c_int(len(embed_np)),
        ctypes.cast(n_arr, ctypes.c_void_p),
        c_int(n_count),
        ctypes.cast(m_arr, ctypes.c_void_p),
        c_int(m_count),
        dw_c,
        c_float(float(mem_weight_ratio)),
        c_float(float(context_factor)),
        txt_c,
        c_int(txt_dim),
        out_buf,
    )

    return torch.tensor(
        list(out_buf), dtype=torch.float32
    ).to(DEVICE)
