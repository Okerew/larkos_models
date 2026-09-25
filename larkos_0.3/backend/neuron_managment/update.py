from ctypes import CDLL, POINTER, c_float, c_int, c_void_p


def bind(lib: CDLL, max_neurons: int, max_connections: int) -> None:
    lib.updateNeuronStates.argtypes = [
        c_void_p,
        c_int,
        POINTER(c_float),
        c_float,
    ]
    lib.updateNeuronStates.restype = None

    lib.processNeurons.argtypes = [
        c_void_p,
        c_int,
        POINTER(c_float),
        POINTER(c_int),
        c_int,
        c_float,
    ]
    lib.processNeurons.restype = None

    global _max_neurons, _max_connections
    _max_neurons     = max_neurons
    _max_connections = max_connections


def call_update_neuron_states(
    lib:              CDLL,
    neurons_ptr,
    recurrent_weights_ptr,
    scaled_factor:    float,
    num_neurons:      int | None = None,
) -> None:
    n = num_neurons if num_neurons is not None else _max_neurons
    lib.updateNeuronStates(
        neurons_ptr,
        c_int(n),
        recurrent_weights_ptr,
        c_float(scaled_factor),
    )


def call_process_neurons(
    lib:             CDLL,
    neurons_ptr,
    weights_ptr,
    connections_ptr,
    scaled_factor:   float,
    num_neurons:     int | None = None,
    max_connections: int | None = None,
) -> None:
    n = num_neurons     if num_neurons     is not None else _max_neurons
    c = max_connections if max_connections is not None else _max_connections
    lib.processNeurons(
        neurons_ptr,
        c_int(n),
        weights_ptr,
        connections_ptr,
        c_int(c),
        c_float(scaled_factor),
    )
