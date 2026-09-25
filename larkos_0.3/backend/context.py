from ctypes import (
    CDLL, Structure,
    c_float,  c_char_p,
    c_uint32,
    POINTER, c_void_p,
)


class GlobalContextManager(Structure):
    pass


GlobalContextManager._fields_ = [
    ("root",                  c_void_p),
    ("total_nodes",           c_uint32),
    ("global_context_vector", POINTER(c_float)),
    ("vector_size",           c_uint32),
    ("decay_rate",            c_float),
    ("update_threshold",      c_float),
    ("max_depth",             c_uint32),
    ("max_children_per_node", c_uint32),
]


def bind(lib: CDLL):
    lib.initializeGlobalContextManager.argtypes = [c_uint32]
    lib.initializeGlobalContextManager.restype  = \
        POINTER(GlobalContextManager)

    lib.addContextNode.argtypes = [
        POINTER(GlobalContextManager),
        c_char_p,
        c_char_p,
        POINTER(c_float),
    ]
    lib.addContextNode.restype = c_void_p

    lib.updateGlobalContext.argtypes = [
        POINTER(GlobalContextManager),
        POINTER(c_float),
        c_uint32,
        POINTER(c_float),
    ]
    lib.updateGlobalContext.restype = None

    lib.integrateGlobalContext.argtypes = [
        POINTER(GlobalContextManager),
        POINTER(c_float),
        c_uint32,
        POINTER(c_float),
        c_uint32,
    ]
    lib.integrateGlobalContext.restype = None

    lib.evaluateConstraintSatisfaction.argtypes = [
        c_void_p,
        POINTER(c_float),
        c_uint32,
    ]
    lib.evaluateConstraintSatisfaction.restype = c_float


def serialize_context_manager(ctx_mgr) -> dict:
    m = ctx_mgr.contents
    return {
        "total_nodes":      int(m.total_nodes),
        "vector_size":      int(m.vector_size),
        "decay_rate":       float(m.decay_rate),
        "update_threshold": float(m.update_threshold),
        "max_depth":        int(m.max_depth),
        "global_context_vector": [
            float(m.global_context_vector[i])
            for i in range(m.vector_size)
        ] if m.global_context_vector else [],
    }
