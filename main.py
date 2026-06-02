"""
To run the test framework instead of the default training uncomment these lines,
comment the preexisting import and main function.

from tests.test_framework import LarkosTestFramework

def main() -> None:
    fw = LarkosTestFramework()
    results = fw.run_all()
    fw.print_report(results)
    fw.save_report(results)


"""

from modules.backend_state import BackendState
from modules.training import training_loop
from modules.runner import run_model


def main() -> None:
    backend = BackendState()
    training_loop(backend, epochs=100, alpha=0.5)

    run_model(
        backend,
        ckpt_path="larkos_model.pt",
        mem_path="memory.bin",
        use_ema=True,
        steps=3,
    )


if __name__ == "__main__":
    main()
