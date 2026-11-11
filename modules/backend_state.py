from ctypes import (
    CDLL, Structure, c_float, c_uint, c_int,
    POINTER, cast, c_void_p, c_uint32, byref,
)
import math
import numpy as np
import os

import modules.backend.memory as memory
import modules.backend.meta as meta
import modules.backend.decision_path as decision_path
import modules.backend.neuron_managment.core as neuron_mgmt
import modules.backend.context as context
import modules.backend.network_state as network_state
import modules.backend.dynamic_paramaters as dynamic_params
import modules.backend.neuron_managment.update as neuron_update
import modules.backend.reflections as reflection
import modules.backend.self_identity as identity
import modules.backend.imagination as imagination
import modules.backend.neuron_managment.specialization as specialization
import modules.backend.emotional_affective as emotional
import modules.backend.workspace as workspace

from modules.config import (
    MAX_NEURONS, INPUT_SIZE, NUM_REGIONS,
    MAX_CONNECTIONS,     MEMORY_VECTOR_SIZE,
    MEMORY_DECAY_INTERVAL,
    WM_POOL_CAPACITY,
    WM_MAX_SLOTS,
    WM_LABEL_SIZE,
    WM_TEXT_SIZE,
    WM_HISTORY_SIZE,
    build_input_tensor,
)


class BackendState:
    def __init__(self):
        self.MAX_NEURONS = MAX_NEURONS
        self.MAX_CONNECTIONS = MAX_CONNECTIONS
        self.INPUT_SIZE = INPUT_SIZE
        self.NUM_REGIONS = NUM_REGIONS
        self.MAX_HISTORY = 128
        self.SCALED_FACTOR = 1.0
        self.IMAGINATION_CREATIVITY = 0.6
        self.IMAGINATION_COHERENCE = 0.5
        self.SPECIALIZATION_THRESHOLD = 0.65
        self.AFFECTIVE_EMBED_DIM = 32
        self.BOND_ENTITY_ID = 1
        self.BOND_ENTITY_NAME = b"training_target"

        class Neuron(Structure):
            _fields_ = [
                ("state", c_float),
                ("output", c_float),
                ("num_connections", c_uint),
                ("layer_id", c_uint),
            ]

        self.Neuron = Neuron

        lib_path = os.path.join(
            os.path.dirname(__file__), "..", "neural_web.so"
        )
        lib_path = os.path.abspath(lib_path)
        self.lib = CDLL(lib_path)

        lib = self.lib

        lib.initializeNeurons.argtypes = [
            POINTER(Neuron),
            POINTER(c_uint),
            POINTER(c_float),
            POINTER(c_float),
        ]
        lib.initializeNeurons.restype = None

        memory.bind(lib)
        meta.bind(lib, self.MAX_NEURONS)
        decision_path.bind(lib, self.MAX_NEURONS)
        neuron_mgmt.bind(lib, self.MAX_NEURONS)
        context.bind(lib)
        network_state.bind(lib)
        dynamic_params.bind(lib)
        neuron_update.bind(lib, self.MAX_NEURONS, self.MAX_CONNECTIONS)
        reflection.bind(lib, self.MAX_NEURONS, self.MAX_CONNECTIONS)
        identity.bind(lib)
        imagination.bind(lib)
        specialization.bind(lib, self.MAX_NEURONS, self.MAX_CONNECTIONS)
        emotional.bind(lib, self.MAX_NEURONS)
        workspace.bind(lib)

        self.neurons = (Neuron * self.MAX_NEURONS)()
        self.connections = (c_uint * (self.MAX_NEURONS * self.MAX_CONNECTIONS))()
        self.weights = (c_float * (self.MAX_NEURONS * self.MAX_CONNECTIONS))()

        # Input is a real observation of system state, NOT random noise.
        # We seed it to zeros and let the first build_input_tensor()
        # call populate it from actual backend signal. The old
        # np.random.rand init meant the model spent its whole life
        # learning to map noise onto an evolving target — there was no
        # learnable relationship, which is why pred never got good.
        self.input_tensor_np = np.zeros(
            self.INPUT_SIZE, dtype=np.float32
        )
        self.input_tensor_c = (c_float * self.INPUT_SIZE)(
            *self.input_tensor_np
        )

        lib.initializeNeurons(
            self.neurons, self.connections, self.weights, self.input_tensor_c
        )

        self.mem_sys = lib.createMemorySystem(memory.MEMORY_CAPACITY)
        self.work_mem = lib.createWorkingMemorySystem(memory.MEMORY_CAPACITY)
        self.wm_pool = lib.createWorkspacePool()
        self.feature_proj = memory.FeatureMatrix()

        self.meta_ctrl = lib.initializeMetaController(self.NUM_REGIONS)
        self.metacog = lib.initializeMetacognitionMetrics()
        self.meta_state = lib.initializeMetaLearningState(self.NUM_REGIONS)

        self.ctx_mgr = lib.initializeGlobalContextManager(
            memory.CONTEXT_VECTOR_SIZE
        )
        self.dyn_params = dynamic_params.make_default()

        self.reflect_hist = lib.initializeReflectionSystem()
        self.reflect_params = lib.initializeReflectionParameters()

        self.ID_NUM_VALUES = 8
        self.ID_NUM_BELIEFS = 16
        self.ID_NUM_MARKERS = 8
        self.ID_HISTORY_SIZE = 64
        self.ID_PATTERN_SIZE = identity.PATTERN_SIZE

        self.identity_sys = lib.initializeSelfIdentity(
            self.ID_NUM_VALUES,
            self.ID_NUM_BELIEFS,
            self.ID_NUM_MARKERS,
            self.ID_HISTORY_SIZE,
            self.ID_PATTERN_SIZE,
        )
        lib.initializeIdentityComponents(self.identity_sys)

        self.imag_sys = lib.initializeImaginationSystem(
            self.IMAGINATION_CREATIVITY,
            self.IMAGINATION_COHERENCE,
        )

        self.spec_sys = lib.initializeSpecializationSystem(
            c_float(self.SPECIALIZATION_THRESHOLD)
        )

        self.aff_sys = lib.initializeAffectiveSystem(
            c_uint32(self.AFFECTIVE_EMBED_DIM)
        )
        self.emo_sys = lib.initializeEmotionalSystem()

        _seed_bond = emotional.AttachmentBond()
        _seed_bond.entity_id = self.BOND_ENTITY_ID
        _seed_bond.entity_name = self.BOND_ENTITY_NAME
        lib.updateAttachmentBond(
            self.aff_sys,
            _seed_bond,
            c_float(0.5),
            c_float(0.5),
            c_float(0.0),
        )

        self.state_history = (
            network_state.NetworkStateSnapshot * self.MAX_HISTORY
        )()
        self._history_count = 0
        self._step_counter = 0

        self.neurons_export = {}
        raw_neurons = [
            {
                "state": self.neurons[i].state,
                "output": self.neurons[i].output,
                "num_connections": self.neurons[i].num_connections,
                "layer_id": self.neurons[i].layer_id,
            }
            for i in range(self.MAX_NEURONS)
        ]
        for i, n in enumerate(raw_neurons):
            n["id"] = i
            n["connections"] = [
                int(self.connections[i * self.MAX_CONNECTIONS + j])
                for j in range(self.MAX_CONNECTIONS)
            ]
            n["weights"] = [
                float(self.weights[i * self.MAX_CONNECTIONS + j])
                for j in range(self.MAX_CONNECTIONS)
            ]
            self.neurons_export[f"neuron_{i}"] = n

        self._prev_outputs = (c_float * self.MAX_NEURONS)(*([0.0] * self.MAX_NEURONS))
        self._prev_states = (c_float * self.MAX_NEURONS)(*([0.0] * self.MAX_NEURONS))
        self._stability_ref = (c_float * self.MAX_NEURONS)(*([0.0] * self.MAX_NEURONS))

    def _refresh_neurons_export(self):
        for i in range(self.MAX_NEURONS):
            state = self.neurons[i].state
            output = self.neurons[i].output
            if math.isnan(state) or math.isinf(state):
                state = 0.0
            if math.isnan(output) or math.isinf(output):
                output = 0.0
            self.neurons_export[f"neuron_{i}"]["state"] = state
            self.neurons_export[f"neuron_{i}"]["output"] = output

    def _snapshot_prev_neuron_state(self):
        for i in range(self.MAX_NEURONS):
            self._prev_outputs[i] = self.neurons[i].output
            self._prev_states[i] = self.neurons[i].state

    # --- fetch-like methods ---

    def get_neurons(self) -> dict:
        return self.neurons_export

    def get_neuron(self, index: int) -> dict:
        key = f"neuron_{index}"
        return self.neurons_export.get(key, {"error": "Neuron not found"})

    def build_input_tensor(self) -> np.ndarray:
        states = np.array(
            [self.neurons[i].state for i in range(self.MAX_NEURONS)],
            dtype=np.float32,
        )
        outputs = np.array(
            [self.neurons[i].output for i in range(self.MAX_NEURONS)],
            dtype=np.float32,
        )
        weights_flat = np.array(
            [self.weights[i]
             for i in range(self.MAX_NEURONS * self.MAX_CONNECTIONS)],
            dtype=np.float32,
        )
        return build_input_tensor(
            states, outputs, weights_flat,
            self._step_counter, self.get_memory_stats(),
            memory.MEMORY_CAPACITY, self.INPUT_SIZE,
        )

    def get_input_tensor(self) -> np.ndarray:
        # Rebuild from live state every call so the input actually
        # reflects the system at this timestep. Keep the C-side mirror
        # in sync so anything on the C boundary reading input_tensor_c
        # sees the same values. No more np.clip on a stale random buffer.
        self.input_tensor_np = self.build_input_tensor()
        for i in range(self.INPUT_SIZE):
            self.input_tensor_c[i] = float(self.input_tensor_np[i])
        return self.input_tensor_np.copy()

    def get_meta_state(self) -> dict:
        return {
            "controller": meta.serialize_controller(self.meta_ctrl),
            "metacognition": meta.serialize_metacog(self.metacog),
            "learning_state": meta.serialize_meta_state(self.meta_state),
        }

    def get_context_state(self) -> dict:
        return context.serialize_context_manager(self.ctx_mgr)

    def get_dynamic_params(self) -> dict:
        return dynamic_params.serialize_params(self.dyn_params)

    def get_memory_state(self) -> dict:
        return memory.serialize_state(self.mem_sys)

    def get_entry_meta(self) -> list:
        return memory.entry_meta(self.mem_sys)

    def get_entry_views(self) -> list:
        return memory.entry_views(self.mem_sys)

    def get_memory_stats(self) -> dict:
        # Sizes/capacities only - the per-step input builder and tier
        # counters use this so a 1M-entry memory never gets marshalled
        # entry-by-entry on the hot path
        return memory.serialize_stats(self.mem_sys)

    def get_network_history(self) -> list:
        return network_state.serialize_history(
            self.state_history, self._history_count
        )

    def get_reflection_state(self) -> dict:
        return {
            "history": reflection.serialize_history(self.reflect_hist),
            "params": reflection.serialize_params(self.reflect_params),
        }

    def get_identity_state(self) -> dict:
        return identity.serialize_identity(self.identity_sys)

    def get_identity_reflection(self) -> dict:
        raw = self.lib.generateIdentityReflection(self.identity_sys)
        return {"reflection": raw.decode() if raw else ""}

    def get_imagination_state(self) -> dict:
        return imagination.serialize_imagination(self.imag_sys)

    def get_specialization_state(self) -> dict:
        return specialization.serialize_system(self.spec_sys)

    def get_affective_state(self) -> dict:
        return emotional.serialize_affective(self.aff_sys)

    def get_emotional_state(self) -> dict:
        return emotional.serialize_emotional(self.emo_sys)

    # --- remote-like methods ---

    def receive_predictions(self, epoch, neuron_pred, fused):
        return {"status": "ok", "epoch": epoch}

    def add_memory_step(self, new_input: list[float] | None = None):
        if new_input is not None:
            cleaned = []
            for v in new_input:
                if math.isnan(v) or math.isinf(v):
                    cleaned.append(0.0)
                else:
                    cleaned.append(v)
            arr = np.array(cleaned, dtype=np.float32)[:self.INPUT_SIZE]
            inp = (c_float * self.INPUT_SIZE)(*arr)
        else:
            inp = self.input_tensor_c

        self.lib.addMemory(
            self.mem_sys, self.work_mem,
            cast(self.neurons, POINTER(c_uint)),
            inp, c_uint(self._step_counter), self.feature_proj,
        )
        self.lib.consolidateToLongTermMemory(
            self.work_mem, self.mem_sys, c_uint(self._step_counter)
        )
        if self._step_counter % MEMORY_DECAY_INTERVAL == 0:
            self.lib.decayMemorySystem(self.mem_sys)

        if self._history_count < self.MAX_HISTORY:
            snap = self.state_history[self._history_count]
            self.lib.captureNetworkState(
                cast(self.neurons, POINTER(c_float)),
                inp,
                snap,
                cast(self.weights, POINTER(c_float)),
                self._step_counter,
            )
            network_state.capture_current_memory(snap, self.mem_sys)
            self._history_count += 1

        self._step_counter += 1
        return {"status": "ok", "step": self._step_counter - 1}

    def consolidate_memory(self):
        self.lib.consolidateMemory(self.mem_sys)
        return {"status": "consolidated"}

    def _wm_pool_path(self, filename: str) -> str:
        base = os.path.splitext(filename)[0]
        return base + "_wm_pool.bin"

    def save_memory(self, filename: str = "memory.bin"):
        self.lib.saveMemorySystem(self.mem_sys, filename.encode())
        self.lib.saveWorkspacePool(
            self.wm_pool, self._wm_pool_path(filename).encode()
        )
        return {"status": "saved", "file": filename}

    def load_memory(self, filename: str = "memory.bin"):
        self.lib.freeMemorySystem(self.mem_sys)
        self.mem_sys = self.lib.loadMemorySystem(filename.encode())
        pool_path = self._wm_pool_path(filename)
        if os.path.exists(pool_path):
            restored = self.lib.loadWorkspacePool(pool_path.encode())
            if restored:
                self.lib.freeWorkspacePool(self.wm_pool)
                self.wm_pool = restored
        return {"status": "loaded", "file": filename}

    def save_network_states(self):
        self.lib.saveNetworkStates(self.state_history, self._history_count)
        return {"status": "saved", "steps": self._history_count}

    def load_network_states(self):
        rc = self.lib.loadNetworkStates(
            cast(self.neurons, POINTER(c_float)),
            cast(self.input_tensor_c, POINTER(c_float)),
        )
        self._refresh_neurons_export()
        self.input_tensor_np = np.array(
            [self.input_tensor_c[i] for i in range(self.INPUT_SIZE)],
            dtype=np.float32,
        )
        status = "loaded" if rc == 0 else "failed"
        return {"status": status, "file": "network_states.json"}

    def update_context(self):
        self.lib.updateGlobalContext(
            self.ctx_mgr,
            cast(self.neurons, POINTER(c_float)),
            self.MAX_NEURONS,
            self.input_tensor_c,
            c_uint32(self.INPUT_SIZE),
        )
        self.lib.integrateGlobalContext(
            self.ctx_mgr,
            cast(self.neurons, POINTER(c_float)),
            self.MAX_NEURONS,
            cast(self.weights, POINTER(c_float)),
            self.MAX_CONNECTIONS,
        )
        return self.get_context_state()

    def update_meta(self, region_scores: list[float]):
        perf = meta.make_performance_metrics(region_scores)
        self.lib.updateMetaControllerPriorities(
            self.meta_ctrl, perf, self.metacog
        )
        self.lib.applyMetaControllerAdaptations(
            cast(self.neurons, POINTER(c_float)),
            cast(self.weights, POINTER(c_float)),
            self.meta_ctrl,
            self.MAX_NEURONS,
        )
        return self.get_meta_state()

    def run_decision_path(self, region_scores: list[float] | None = None):
        self.lib.selectOptimalMetaDecisionPath(
            cast(self.neurons, POINTER(c_float)),
            cast(self.weights, POINTER(c_float)),
            cast(self.connections, POINTER(c_uint)),
            self.input_tensor_c,
            self.MAX_NEURONS,
            cast(self.meta_state, POINTER(c_float)),
            cast(self.metacog, POINTER(c_float)),
        )
        scores = region_scores or [0.5] * self.NUM_REGIONS
        return self.update_meta(scores)

    def update_neuron_states(self, scaled_factor: float | None = None):
        sf = scaled_factor if scaled_factor is not None else self.SCALED_FACTOR
        neuron_update.call_update_neuron_states(
            self.lib,
            cast(self.neurons, c_void_p),
            cast(self.weights, POINTER(c_float)),
            sf,
        )
        self._refresh_neurons_export()
        return {"status": "ok", "op": "update_neuron_states"}

    def process_neurons(self, scaled_factor: float | None = None):
        sf = scaled_factor if scaled_factor is not None else self.SCALED_FACTOR
        neuron_update.call_process_neurons(
            self.lib,
            cast(self.neurons, c_void_p),
            cast(self.weights, POINTER(c_float)),
            cast(self.connections, POINTER(c_int)),
            sf,
        )
        self._refresh_neurons_export()
        return {"status": "ok", "op": "process_neurons"}

    def run_reflection(self):
        # history contains _history_count populated entries (0 .. _history_count-1).
        # C helpers index into history[step-1] so step must never exceed
        # _history_count. _step_counter grows unboundedly across epochs while
        # _history_count caps at MAX_HISTORY, so we pass the smaller value.
        safe_step = min(self._step_counter, self._history_count)
        metrics = self.lib.performSelfReflection(
            cast(self.neurons, c_void_p),
            self.mem_sys,
            self.state_history,
            self.reflect_hist,
            c_int(safe_step),
        )
        return reflection.serialize_metrics(metrics)

    def integrate_reflection(self):
        self.lib.integrateReflectionSystem(
            cast(self.neurons, c_void_p),
            self.mem_sys,
            self.state_history,
            c_int(self._step_counter),
            cast(self.weights, POINTER(c_float)),
            cast(self.connections, POINTER(c_uint)),
            self.reflect_params,
        )
        return self.get_reflection_state()

    def capture_stability_reference(self):
        # Snapshots neuron states as the reference for measureNetworkStability
        # so the C-side adaptation can gauge how much the epoch moved them.
        for i in range(self.MAX_NEURONS):
            self._stability_ref[i] = self.neurons[i].state

    def adapt_dynamic_parameters(
        self, performance_delta: float, error_rate: float
    ) -> dict:
        # measure -> updateDynamicParameters -> adaptNetworkDynamic. Order
        # matters: stability must be read before the adaptation mutates the
        # neuron states, and updateDynamicParameters must run first so
        # adaptNetworkDynamic consumes the freshly-updated params.
        stability = float(self.lib.measureNetworkStability(
            cast(self.neurons, c_void_p),
            self._stability_ref,
        ))
        self.lib.updateDynamicParameters(
            self.dyn_params,
            c_float(performance_delta),
            c_float(stability),
            c_float(error_rate),
            cast(self.meta_ctrl, c_void_p),
            cast(self.metacog, c_void_p),
        )
        self.lib.adaptNetworkDynamic(
            cast(self.neurons, c_void_p),
            cast(self.weights, POINTER(c_float)),
            self.dyn_params,
            c_float(performance_delta),
            self.input_tensor_c,
        )
        self._refresh_neurons_export()
        return {
            "stability": stability,
            "params":    self.get_dynamic_params(),
        }

    def update_identity(self, new_input: list[float] | None = None):
        inp = (
            (c_float * self.INPUT_SIZE)(
                *np.array(new_input, dtype=np.float32)[:self.INPUT_SIZE]
            )
            if new_input is not None
            else self.input_tensor_c
        )
        self.lib.updateIdentity(
            self.identity_sys,
            cast(self.neurons, c_void_p),
            c_uint32(self.MAX_NEURONS),
            self.mem_sys,
            cast(inp, POINTER(c_float)),
        )
        return self.get_identity_state()

    def verify_identity(self) -> dict:
        verified = bool(self.lib.verifyIdentity(self.identity_sys))

        analysis_data = None
        restored      = False

        if not verified:
            print("Warning: Identity consistency check failed")
            analysis = self.lib.analyzeIdentitySystem(self.identity_sys)
            analysis_data = identity.serialize_analysis(analysis)
            print(
                f"Core Value Conflicts: "
                f"{analysis_data['core_value_conflicts']}"
            )
            print(
                f"Belief Contradictions: "
                f"{analysis_data['belief_contradictions']}"
            )
            print(
                f"Marker Deviations: "
                f"{analysis_data['marker_deviations']}"
            )
            print(
                f"Coherence Score: "
                f"{analysis_data['coherence_score']:.4f}"
            )
            print(
                f"Stability Index: "
                f"{analysis_data['stability_index']:.4f}"
            )
            print(
                f"Adaptation Needed: "
                f"{analysis_data['adaptation_needed']}"
            )

            backup = self.lib.createIdentityBackup(self.identity_sys)
            if backup:
                self.lib.restoreIdentityFromBackup(
                    self.identity_sys, backup
                )
                self.lib.freeIdentityBackup(backup)
                restored = True

        raw = self.lib.generateIdentityReflection(self.identity_sys)
        reflection = raw.decode() if raw else ""
        if reflection:
            print(reflection)

        return {
            "verified":    verified,
            "restored":    restored,
            "reflection":  reflection,
            "analysis":    analysis_data,
        }

    def apply_imagination_to_decision(self) -> dict:
        influence = self.lib.applyImaginationToDecision(
            self.imag_sys,
            cast(self.neurons, POINTER(c_float)),
            cast(self.input_tensor_c, POINTER(c_float)),
            self.MAX_NEURONS,
        )
        self._refresh_neurons_export()

        if self._step_counter % 5 == 0:
            print(
                f"Applied imagination with influence: "
                f"{float(influence) * 100.0:.2f}%"
            )

        sys = self.imag_sys.contents

        scenario = None
        if sys.num_scenarios > 0 and 0 <= sys.current_scenario < imagination.MAX_SCENARIOS:
            scenario = sys.scenarios[sys.current_scenario]
            history_idx = self._step_counter % imagination.DIVERGENCE_HISTORY_SIZE
            sys.divergence_history[history_idx] = scenario.divergence_factor
            sys.steps_simulated += 1

        deactivated = False
        if sys.steps_simulated > 20:
            count = int(sys.steps_simulated)
            sys.active = False
            sys.steps_simulated = 0
            print(f"Deactivating imagination after {count} steps")
            deactivated = True

        return {
            "status":               "ok",
            "influence":            float(influence),
            "steps_simulated":      int(sys.steps_simulated),
            "divergence_history_idx": history_idx if scenario is not None else -1,
            "divergence_factor":    float(scenario.divergence_factor) if scenario is not None else 0.0,
            "deactivated":          deactivated,
        }

    def activate_imagination_scenario(
        self,
        divergence:        float,
        task_description:  str   = "",
        simulate_steps:    int   = 10,
    ) -> dict:
        sys = self.imag_sys.contents
        sys.active = True

        new_scenario = self.lib.createScenario(
            cast(self.neurons, POINTER(c_float)),
            self.mem_sys,
            self.MAX_NEURONS,
            c_float(divergence),
        )

        label = (
            f"Scenario_{int(sys.total_scenarios_generated)}"
            f"_{task_description}"
        ).encode()[:imagination.SCENARIO_NAME_SIZE - 1]
        padded = label + b'\x00' * (
         imagination.SCENARIO_NAME_SIZE - len(label)
        )
        sys.current_scenario_name = padded
        sys.total_scenarios_generated += 1  
        self.lib.simulateScenario(
            new_scenario,
            cast(self.neurons, POINTER(c_float)),
            cast(self.input_tensor_c, POINTER(c_float)),
            self.MAX_NEURONS,
            simulate_steps,
            cast(self.weights, POINTER(c_float)),
            cast(self.connections, POINTER(c_uint)),
            self.MAX_CONNECTIONS,
        )
        self.lib.evaluateScenarioPlausibility(
            new_scenario,
            self.mem_sys,
        )

        slot = int(sys.num_scenarios)
        if slot < imagination.MAX_SCENARIOS:
            sys.scenarios[slot] = new_scenario
            sys.num_scenarios  += 1
            sys.current_scenario = slot
        else:
            worst_idx = 0
            worst_score = float("inf")
            for i in range(imagination.MAX_SCENARIOS):
                sc = sys.scenarios[i]
                n  = max(0, min(
                    sc.num_outcomes,
                    imagination.MAX_OUTCOMES_PER_SCENARIO,
                ))
                score = sum(
                    sc.outcomes[j].plausibility * sc.outcomes[j].confidence
                    for j in range(n)
                )
                if score < worst_score:
                    worst_score = score
                    worst_idx   = i
            sys.scenarios[worst_idx] = new_scenario
            sys.current_scenario     = worst_idx

        return self.get_imagination_state()


    def update_imagination_creativity(
        self, performance_delta: float, novelty: float
    ) -> dict:
        self.lib.updateImaginationCreativity(
            self.imag_sys,
            c_float(performance_delta),
            c_float(novelty),
        )
        return self.get_imagination_state()

    def adjust_neurons_with_imagination(
        self, outcome_index: int = 0, influence_factor: float = 1.0
    ) -> dict:
        s = self.imag_sys.contents
        scenario = s.scenarios[s.current_scenario]
        n = max(
            0,
            min(
                scenario.num_outcomes,
                imagination.MAX_OUTCOMES_PER_SCENARIO,
            ),
        )
        idx = max(0, min(outcome_index, n - 1 if n > 0 else 0))
        outcome = scenario.outcomes[idx]
        self.lib.adjustNeuronsWithImagination(
            cast(self.neurons, POINTER(c_float)),
            outcome,
            c_int(self.MAX_NEURONS),
            c_float(influence_factor),
        )
        self._refresh_neurons_export()
        return {
            "status": "ok",
            "op": "adjust_neurons_with_imagination",
            "outcome_index": idx,
            "influence_factor": influence_factor,
        }
    
    def problem_solve_with_imagination(
        self,
        total_error: float,
    ) -> dict:
        if total_error <= 0.5:
            return {"status": "skipped", "reason": "error below threshold"}
        problem_scenario = self.lib.createScenario(
            cast(self.neurons, POINTER(c_float)),
            self.mem_sys,
            self.MAX_NEURONS,
            c_float(0.6),
        )
        self.lib.simulateScenario(
            problem_scenario,
            cast(self.neurons, POINTER(c_float)),
            cast(self.input_tensor_c, POINTER(c_float)),
            self.MAX_NEURONS,
            15,
            cast(self.weights, POINTER(c_float)),
            cast(self.connections, POINTER(c_uint)),
            self.MAX_CONNECTIONS,
        )
        blended = (c_float * MEMORY_VECTOR_SIZE)(*([0.0] * MEMORY_VECTOR_SIZE))
        self.lib.blendImaginedOutcomes(
            problem_scenario.outcomes,
            problem_scenario.num_outcomes,
            blended,
        )
        # input_tensor_c can be shorter than MAX_NEURONS if update_meta
        # reinitialised it with a different size clamp to be safe
        tensor_len = min(
            self.MAX_NEURONS,
            MEMORY_VECTOR_SIZE,
            len(self.input_tensor_c),
        )
        for i in range(tensor_len):
            bv = float(blended[i])
            if math.isnan(bv) or math.isinf(bv):
                bv = 0.0
            self.neurons[i].state = (
                self.neurons[i].state * 0.7 + bv * 0.3
            )
            self.input_tensor_c[i] = (
                float(self.input_tensor_c[i]) * 0.8 + bv * 0.2
            )
            # Clamp input tensor to prevent unbounded growth from
            # compounding multiplicative updates in C reshape
            self.input_tensor_c[i] = max(-10.0, min(10.0, float(self.input_tensor_c[i])))
        self.input_tensor_np[:tensor_len] = [
            float(self.input_tensor_c[i])
            for i in range(tensor_len)
        ]
        self._refresh_neurons_export()
        return {
            "status":  "ok",
            "op":      "problem_solve_with_imagination",
            "blended": [float(blended[i]) for i in range(MEMORY_VECTOR_SIZE)],
        }
    def store_best_imagination_to_memory(self) -> dict:
        sys = self.imag_sys.contents
        n   = int(sys.num_scenarios)
        if n == 0:
            return {"status": "skipped", "reason": "no scenarios"}

        from modules.backend.memory import MemoryEntry
        best_idx   = -1
        best_score = -1.0
        for i in range(n):
            sc    = sys.scenarios[i]
            nout  = max(0, min(
                sc.num_outcomes, imagination.MAX_OUTCOMES_PER_SCENARIO
            ))
            if nout == 0:
                continue
            score = max(
                (
                    sc.outcomes[j].plausibility
                    * sc.outcomes[j].confidence
                    for j in range(nout)
                ),
                default=0.0,
            )
            if score > best_score:
                best_score = score
                best_idx   = i

        if best_idx < 0:
            return {"status": "skipped", "reason": "no scenarios with outcomes"}

        best_sc  = sys.scenarios[best_idx]
        nout     = max(0, min(
            best_sc.num_outcomes, imagination.MAX_OUTCOMES_PER_SCENARIO
        ))
        if nout == 0:
            return {"status": "skipped", "reason": "selected scenario has no outcomes"}
        entry    = MemoryEntry()
        for i in range(MEMORY_VECTOR_SIZE):
            val = float(best_sc.outcomes[0].vector[i])
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            entry.vector[i] = c_float(val)
        if math.isnan(best_score) or math.isinf(best_score):
            best_score = 0.0
        entry.importance = c_float(best_score)
        entry.timestamp  = c_uint(self._step_counter)

        self.lib.addToDirectMemory(self.mem_sys, byref(entry))
        return {
            "status":     "ok",
            "op":         "store_best_imagination_to_memory",
            "best_idx":   best_idx,
            "best_score": best_score,
        }

    def detect_specializations(
        self, target_outputs: list[float] | None = None
    ) -> dict:
        self._snapshot_prev_neuron_state()
        tgt = (
            (c_float * self.INPUT_SIZE)(
                *np.array(target_outputs, dtype=np.float32)[:self.INPUT_SIZE]
            )
            if target_outputs is not None
            else (c_float * self.INPUT_SIZE)(*([0.0] * self.INPUT_SIZE))
        )
        self.lib.detectSpecializations(
            self.spec_sys,
            cast(self.neurons, POINTER(c_float)),
            c_int(self.MAX_NEURONS),
            cast(self.input_tensor_c, POINTER(c_float)),
            cast(tgt, POINTER(c_float)),
            cast(self._prev_outputs, POINTER(c_float)),
            cast(self._prev_states, POINTER(c_float)),
        )
        return self.get_specialization_state()

    def apply_specializations(self) -> dict:
        self.lib.applySpecializations(
            self.spec_sys,
            cast(self.neurons, POINTER(c_float)),
            cast(self.weights, POINTER(c_float)),
            cast(self.connections, POINTER(c_int)),
            c_int(self.MAX_NEURONS),
            c_int(self.MAX_CONNECTIONS),
        )
        self._refresh_neurons_export()
        return {"status": "ok", "op": "apply_specializations"}

    def update_specialization_importance(
        self, network_performance: float, error_rate: float
    ) -> dict:
        self.lib.updateSpecializationImportance(
            self.spec_sys,
            c_float(network_performance),
            c_float(error_rate),
            cast(self.neurons, POINTER(c_float)),
        )
        return self.get_specialization_state()

    def evaluate_specialization_effectiveness(
        self, network_performance: float
    ) -> dict:
        score = self.lib.evaluateSpecializationEffectiveness(
            self.spec_sys,
            c_float(network_performance),
        )
        return {"effectiveness": float(score)}

    def detect_emotional_triggers(
        self,
        target_outputs: list[float] | None = None,
        satisfaction: float = 0.5,
    ) -> dict:
        tgt = (
            (c_float * self.MAX_NEURONS)(
                *np.array(target_outputs, dtype=np.float32)[:self.MAX_NEURONS]
            )
            if target_outputs is not None
            else (c_float * self.MAX_NEURONS)(*([0.0] * self.MAX_NEURONS))
        )
        self.lib.detectEmotionalTriggers(
            self.emo_sys,
            cast(self.neurons, c_void_p),
            cast(tgt, POINTER(c_float)),
            c_int(self.MAX_NEURONS),
            c_uint(self._step_counter),
            c_float(satisfaction),
            self.aff_sys,
            None,
        )
        return {
            "emotional": self.get_emotional_state(),
            "affective": self.get_affective_state(),
        }

    def apply_emotional_processing(
        self, learning_rate: float = 0.01, plasticity: float = 0.1
    ) -> dict:
        self.lib.applyEmotionalProcessing(
            self.emo_sys,
            cast(self.neurons, c_void_p),
            c_int(self.MAX_NEURONS),
            cast(self.input_tensor_c, POINTER(c_float)),
            c_float(learning_rate),
            c_float(plasticity),
            self.aff_sys,
        )
        self._refresh_neurons_export()
        return {
            "status": "ok",
            "op": "apply_emotional_processing",
            "emotional": self.get_emotional_state(),
            "affective": self.get_affective_state(),
        }

    def update_attractor_dynamics(self) -> dict:
        self.lib.updateAttractorDynamics(
            self.aff_sys,
            cast(self.input_tensor_c, POINTER(c_float)),
            c_uint32(self._step_counter),
        )
        return self.get_affective_state()

    def update_affective_complexity(self) -> dict:
        self.lib.updateAffectiveComplexity(
            self.aff_sys, c_uint32(self._step_counter)
        )
        return self.get_affective_state()

    def reshape_embeddings_with_emotion(self) -> dict:
        self.lib.reshapeEmbeddingsWithEmotion(
            self.aff_sys,
            cast(self.input_tensor_c, POINTER(c_float)),
            c_uint32(self.INPUT_SIZE),
        )
        return self.get_affective_state()

    def trigger_emotion(self, emotion_type: int, trigger_strength: float) -> dict:
        self.lib.triggerEmotion(
            self.emo_sys,
            c_int(emotion_type),
            c_float(trigger_strength),
            c_uint(self._step_counter),
        )
        return self.get_emotional_state()

    def compute_mask_intensity(self, person_id: int = 0) -> dict:
        intensity = self.lib.h_iga(
            None,
            self.aff_sys,
            self.emo_sys,
            c_int(person_id),
        )
        return {"mask_intensity": float(intensity)}

    def update_bond(
        self, attachment_strength: float, trust: float, valence: float
    ) -> dict:
        bond = emotional.AttachmentBond()
        bond.entity_id = self.BOND_ENTITY_ID
        bond.entity_name = self.BOND_ENTITY_NAME
        self.lib.updateAttachmentBond(
            self.aff_sys,
            bond,
            c_float(attachment_strength),
            c_float(trust),
            c_float(valence),
        )
        return self.get_affective_state()

    # --- WM runtime (workspace pool) ---

    def _wm_vec(self, values):
        arr = (c_float * MEMORY_VECTOR_SIZE)(
            *([0.0] * MEMORY_VECTOR_SIZE)
        )
        for i, v in enumerate(list(values)[:MEMORY_VECTOR_SIZE]):
            v = float(v)
            arr[i] = 0.0 if (math.isnan(v) or math.isinf(v)) else v
        return arr

    def _wm_find(self, ws_id: int):
        if not self.wm_pool:
            return None
        pool = self.wm_pool.contents
        for i in range(WM_POOL_CAPACITY):
            ws = pool.workspaces[i]
            if ws.active and int(ws.workspace_id) == int(ws_id):
                return ws
        return None

    @staticmethod
    def _wm_label(slot_field) -> str:
        return slot_field.decode(errors="replace").rstrip("\x00")

    def wm_save_pool(self, filename: str = "memory.bin") -> dict:
        path = self._wm_pool_path(filename)
        self.lib.saveWorkspacePool(self.wm_pool, path.encode())
        return {"status": "saved", "file": path}

    def wm_status(self) -> dict:
        return workspace.serialize_pool(self.wm_pool)

    def wm_allocate(self, kind: int, query_vec=None) -> dict:
        q = None
        if query_vec is not None:
            q = self._wm_vec(query_vec)
        rc = self.lib.createWorkspace(
            self.wm_pool,
            c_uint(int(kind)),
            cast(q, POINTER(c_float)) if q is not None else None,
            self.mem_sys,
            c_uint(self._step_counter),
        )
        return {"rc": int(rc)}

    def wm_destroy(self, ws_id: int) -> dict:
        rc = self.lib.destroyWorkspace(
            self.wm_pool, c_uint(int(ws_id))
        )
        return {"rc": int(rc)}

    def wm_slots(self, ws_id: int) -> list:
        ws = self._wm_find(ws_id)
        if ws is None:
            return []
        out = []
        for i in range(WM_MAX_SLOTS):
            s = ws.slots[i]
            if not s.used:
                continue
            out.append({
                "index": i,
                "label": self._wm_label(s.label),
                "text": self._wm_label(s.text),
                "importance": round(float(s.importance), 6),
                "step_tag": int(s.step_tag),
            })
        return out

    def wm_write(
        self, ws_id: int, label: str, text: str,
        vector, importance: float,
    ) -> dict:
        v = self._wm_vec(vector)
        rc = self.lib.writeSlot(
            self.wm_pool,
            c_uint(int(ws_id)),
            label.encode()[:WM_LABEL_SIZE - 1],
            (text or "").encode()[:WM_TEXT_SIZE - 1],
            cast(v, POINTER(c_float)),
            c_float(float(importance)),
            c_uint(self._step_counter),
        )
        return {"rc": int(rc)}

    def wm_transform(
        self, ws_id: int, label: str, target_vec, alpha: float,
    ) -> dict:
        t = self._wm_vec(target_vec)
        rc = self.lib.transformSlot(
            self.wm_pool,
            c_uint(int(ws_id)),
            label.encode()[:WM_LABEL_SIZE - 1],
            cast(t, POINTER(c_float)),
            c_float(float(alpha)),
            c_uint(self._step_counter),
        )
        return {"rc": int(rc)}

    def wm_link(
        self, ws_id: int, a: str, b: str,
        link_type: str, weight: float,
    ) -> dict:
        rc = self.lib.linkSlots(
            self.wm_pool,
            c_uint(int(ws_id)),
            a.encode()[:WM_LABEL_SIZE - 1],
            b.encode()[:WM_LABEL_SIZE - 1],
            (link_type or "").encode()[:WM_LABEL_SIZE - 1],
            c_float(float(weight)),
        )
        return {"rc": int(rc)}

    def wm_retrieve(self, ws_id: int, query_vec, k: int = 5) -> dict:
        q = self._wm_vec(query_vec)
        n = max(1, min(int(k), WM_MAX_SLOTS))
        out_idx = (c_int * n)()
        out_score = (c_float * n)()
        rc = self.lib.wmRetrieve(
            self.wm_pool,
            c_uint(int(ws_id)),
            cast(q, POINTER(c_float)),
            c_int(n),
            self.work_mem,
            c_uint(self._step_counter),
            out_idx,
            out_score,
            c_int(n),
        )
        if rc < 0:
            return {"rc": int(rc), "hits": []}
        ws = self._wm_find(ws_id)
        hits = []
        if ws is not None:
            for i in range(int(rc)):
                s = ws.slots[out_idx[i]]
                hits.append({
                    "label": self._wm_label(s.label),
                    "score": round(float(out_score[i]), 6),
                    "importance": round(float(s.importance), 6),
                    "step_tag": int(s.step_tag),
                })
        return {"rc": int(rc), "hits": hits}

    def wm_commit_slot(
        self, ws_id: int, label: str, step: int, blend: float,
    ) -> dict:
        rc = self.lib.wmCommitSlot(
            self.wm_pool,
            c_uint(int(ws_id)),
            label.encode()[:WM_LABEL_SIZE - 1],
            self.mem_sys,
            c_uint(int(step)),
            c_float(float(blend)),
        )
        return {"rc": int(rc)}

    def wm_validate(self, ws_id: int) -> dict:
        labels = workspace.slot_labels(self.wm_pool, ws_id)
        if not labels:
            return {"rc": 0, "results": []}
        n = len(labels)
        out_idx = (c_int * n)()
        out_flag = (c_int * n)()
        out_coh = (c_float * n)()
        out_cons = (c_float * n)()
        rc = self.lib.wmValidateWorkspace(
            self.wm_pool,
            c_uint(int(ws_id)),
            self.mem_sys,
            out_idx,
            out_flag,
            out_coh,
            out_cons,
            c_int(n),
        )
        if rc < 0:
            return {"rc": int(rc), "results": []}
        by_index = {idx: label for idx, label, _ in labels}
        flag_names = {
            0: "ok",
            1: "low_coherence",
            2: "low_consistency",
            3: "confabulated_dereferenced",
            4: "confabulated_kept",
        }
        results = []
        for i in range(int(rc)):
            flag = int(out_flag[i])
            results.append({
                "label": by_index.get(int(out_idx[i]), "?"),
                "flag": flag_names.get(flag, str(flag)),
                "coherence": round(float(out_coh[i]), 6),
                "consistency": round(float(out_cons[i]), 6),
            })
        return {"rc": int(rc), "results": results}

    def wm_discard_slot(self, ws_id: int, label: str) -> dict:
        imp = c_float(0.0)
        rc = self.lib.wmDiscardSlot(
            self.wm_pool,
            c_uint(int(ws_id)),
            label.encode()[:WM_LABEL_SIZE - 1],
            byref(imp),
            c_uint(self._step_counter),
        )
        return {"rc": int(rc), "importance": round(float(imp.value), 6)}

    def wm_replay(
        self, ws_id: int, from_step: int = 0, limit: int = 64,
    ) -> dict:
        n = max(1, min(int(limit), WM_HISTORY_SIZE))
        out = (workspace.WMSlotSnapshot * n)()
        rc = self.lib.replayWorkspace(
            self.wm_pool,
            c_uint(int(ws_id)),
            c_uint(int(from_step)),
            out,
            c_int(n),
        )
        if rc < 0:
            return {"rc": int(rc), "snapshots": []}
        snaps = [
            {
                "step": int(out[i].step),
                "label": self._wm_label(out[i].label),
                "importance": round(float(out[i].importance), 6),
            }
            for i in range(int(rc))
        ]
        return {"rc": int(rc), "snapshots": snaps}
