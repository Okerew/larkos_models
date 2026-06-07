# <img src="larkos0.1.png" width=25px, height = 25px> Larkos 0.1

Self-learning model with a state-based fusion mechanism, transformers that guide the model,
based on the <a href="https://github.com/Okerew/larkos">larkos architecture</a>.
The main parts of the architecture are the fusion mechanism, the training loop.

To understand the architecture better I recommend firstly reading the [paper](https://github.com/Okerew/larkos_0.1/blob/main/Documents/thesis.pdf) about it.

<a href="https://mermaid.live/edit#pako:eNp1V-FO4zgQfpVRVjrdisCmBVoa3e2ptOkKLbAVcLu6W1bIJG7rI7EjxwUKy7vfeJw4KXfwB89kvs_j8cx4-hykKuNBHCw1K1dwNb2WgH-Xhmnz6_crzYQUcunkH-9hd_cjnEhhzrhh368DuxQsF08crMaaGQ6_wOnFdfDDMTXWBE1Kla5OlSoRm7B05RTe1n8m4wvOsmOW3nGZobmVoBbdPjHtGU6UNPzRhHDO11rJKkR1ofQmhJOCLYVkRigZQlIou6g-jBcLnhpxz6HaVIYXwE26513obEpOTLlG0znTrKjQCycCydxwXcUwzssVC3-71R8-Jo9lrjRtCBfoIbkC37hYrkzlt-hS0h6XrChzfoWHwB2cAFbC2JVrAwutCpgyDOFclDwXknuqFlnfDNo3RnQ9Ft8oYiAn4Vzpgu4MITO11oJrSKRNAyK54kWJh8htkNO7zj12uMlwpvQD09mcVTYyNwsnekDnM5mfTaZalWptD3k2gVqAuVa3mGHk2lemBZMph6QyoqA4ejoPd2TobT7X3GYGrcEKIqXQnzJ9pypSt_AGQPCJVlU1NkYinNaAApcWTY7suPijb__wdMsNj3Q8amkr4J7P1pUNuJcBFQ3ZZPdSYGyPtciW7dVtQV08CYIlJyuMZcE1EjoddJREecwwPb3LbchfM7hYjc9OT6QkPrsGEsDWGZGNM1YaDMyrgDUo59tDdsGrdW4vr75YcIoYCou7KTG2rgoWeKAshIIVtRaweSy5ad1syIjaVls3j25r2Zt3DeqoF5iL_FQRoJbAijFggqDLO3DLMKo7YPfHf2z9eJPj9070PQVRftIsExjNY5bbBETaRgNO1aQoOp4yrM-kuEXm16X9isZTT3JhW579520q4vuUq1usNVuT8Dv09qItLgJYki8lFsSl4ZaF1li-GqwCdojnMl3xbJ17JSRnY_izzLALeUZPQpT27F9ZTq3VrLUEG6Cbe9amQG3hehSmcEKdky6pQvGGO7ntRq1N3UkezZTbxoIQKignUZO1lfxF5huPbq19WmATdkdAfNOUnaKKmy5Pp29bPzb5-7rjn2S2PMxmK5E8pXuP6kfBPkduicdt--WVFstlXXKNaVh7AMdKZu3Zm8_1y7XIXd-g6DaCfa-0SNF3fLMWGCxMEef_VIuFfcHUPc8NvlwTteLafu68S57F7vAV35CFSFm9R1eEyYqnd0Q7Z9giNLWP9A4z2LNt2btkWOJTuUQqXDV-glEwEznvJARZuculmka3NK9Wz06Cmeb8idPe34TM1AN8uef6j5f6hrsQS_LzL179hFrhvrqAkcGE2ZyuUU3x2U1sf3Ff2-T5D_W5-gmfOS9nWj1xGyMrgJNqcxxU6FXrBrn1pB1YEpk9nynNnVQ1x2nZt0w7o0zihgh3TD_c_I-BdXauKkPzFvpq17vN8OXd8xZ1F5SVyoXNRSwFaoReURdHp911beuh455TppRKSDd44LPVaijgOb2kN9Th90rTmTu6YM93zs2D0ncNWS26aa3CaL_yqgNxjwwOa_lW7XzCa2jKuFMC7UuyjXD3ILPO4GobfY6T2o_3DlGZDXZvGmdhIfI8fncwGc8Oo7AyWt3x-F0_GU73-2GqcJKL3y0Wiy7O3pdDJaNeMtj3qKOjKDmYvYHqTkIO3e-NBrMW3TscHE6iN9Bbz5-Dz2ajo6h1ORkc9qK34N2m7NCjSX943KIH497xaPzWgf1E3uw86UVDj50dDme9YYONoigI8YeEyILY6DUPAxxBCmbF4NmyXgdmxQus2hiXGebWdXAtXxBTMvm3UkUD02q9XAXxguUVSmvqt1PB8CdK4bVsbdTlRqYNBimC-Dl4DOL-0dHeMIp6_cFo2DuIDvqHYbBB9fBgLxrsDwbR_qC3PzgcvoTBE226v4fRjKLhUR9Bo_3RAQKw-2ZcT9RaGst4EIUBzpdG6TP3c4l-Nb38C9lab5I">Diagram</a>

## Build with docker:
Build with `sh build.sh`
## Run with docker:
Run with `sh run.sh`

## Run normally:
Although I don't recommend it you can do this by firstly installing requirements `pip install --no-cache-dir -r requirements.txt`
also install pytorch: `pip3 install torch`
you also need to install json-c `sudo apt install libjson-c-dev`.
Then just run the main.py file normally `python main.py`

---

## Where is what

| Path | What |
|---|---|
| `main.py` | Entry point, runs training loop then inference |
| `modules/` | Python core: config, model, training, runner, checkpointing, data pipeline |
| `modules/backend/` | Reflection, identity, memory, motivation, imagination, decision paths |
| `modules/fusion_mechanism/` | C extension for neural fusion |
| `neural_web.c` / `immitrin_functions.c` | C backend, neural web + SIMD functions |
| `include/` | C headers (`definitions.h`, etc.) |
| `Documents/` | Architecture & design docs |
| `Documents/model_architecture.tex/pdf` | Full architecture paper (LaTeX) |
| `Documents/fusion_mechanism.tex/pdf` | Fusion mechanism paper (LaTeX) |
| `Documents/thesis.tex/pdf` | A full paper about the architecture and the CFM, experiments (LaTeX) |
| `Documents/scaling.md` | Guide for scaling model hyperparameters |
| `Documents/testing_framework_for_larkos.md` | Test definitions (9 tests: learning, transfer, continual, etc.) |
| `tests/` | Python test framework |
| `Dockerfile` / `build.sh` / `compile.sh` / `run.sh` | Build & run scripts |

---

## Notes:
I use docker to run everything and I recommend the same, the build.sh, run.sh scripts:
setup docker and run it. That being said you don't have to use docker.
This is obviously a certain model version so don't commit "architectural improvements",
you can commit bug fixes, though generally speaking you shouldn't commit anything to this.

The text in test_data was completely ai generated.

I generally recommended training the model by first using the testing feature 
then combining the test checkpoints and starting training from there.
You could also try after that running training normally than doing inference saving
outputs/inputs from inference and then again some like 3 training steps that would probably work good with the model.

Cuda is recommended.

The code is licensed under the Apache 2.0 License see [NOTICE](NOTICE) for relevant information.
