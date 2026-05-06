"""Adapter interface for Evo genomic foundation model embedding extraction.

Model file
----------
The extraction backend uses a modified version of the original ``StripedHyena``
model class located at::

    models/evo_1_131k_base/modified_model.py

The modification is backward-compatible with the original checkpoint and adds
three capabilities without altering original inference logic:

1. ``output_hidden_states: bool`` parameter on ``forward()``
   Collects ``x.clone()`` after every block, building a list of 32 tensors
   of shape ``(batch, seq_len, hidden_dim)``.

2. ``return_dict: bool`` parameter on ``forward()``
   Returns a dict with keys ``logits``, ``hidden_states``, ``last_hidden_state``
   (after the final RMSNorm), and ``inference_params``.

3. Convenience extraction methods::

       model.extract_layer_hidden_states(input_ids, target_layer)
       model.extract_multiple_layers(input_ids, layer_list)

   Both call ``forward(..., output_hidden_states=True, return_dict=True)``
   internally and index into ``outputs['hidden_states']``.

Hidden state indexing
---------------------
``hidden_states`` is a Python list indexed 0 to 31 (32 Hyena/attention blocks).
``hidden_states[k]`` contains activations *after* block ``k`` has processed
the sequence.  Layer 10 in thesis notation corresponds to index 10::

    outputs['hidden_states'][10]   # shape: (batch, n_tokens, hidden_dim)

``last_hidden_state`` is the output of the final RMSNorm layer, not one of the
32 block outputs.  It is the representation used for next-token prediction and
is *not* recommended for AMR downstream tasks (see layer selection notes below).

Layer 10 selection rationale
-----------------------------
A diagnostic suite was run across all 32 layers using
``extract_multiple_layers(input_ids, list(range(32)))``.  For each layer and
several random sequence seeds, the following metrics were computed:

- activation magnitude (mean L2 norm of hidden states per token)
- isotropic angular diversity (uniformity of direction distribution)
- effective rank (rank of the activation covariance matrix)
- singular spectrum shape
- token-norm concentration
- cross-seed stability (variance of metrics across seeds)

Results show a sharp stability boundary at Layer 11.  Layer 10 is the deepest
jointly stable extraction point before the stability boundary and is used for
all downstream AMR experiments.

Memory and precision notes
--------------------------
``modified_model.py`` overrides ``precompute_filters`` to:

- Compute filter poles/residues on CPU in float32 (numerical stability).
- Use a log-exp trick (``exp(log(poles) * t)``) instead of ``poles ** t``.
- Store the precomputed ``h`` buffer on GPU in bfloat16 to halve VRAM usage.

The method ``to_bfloat16_except_poles_residues()`` converts all parameters to
bfloat16 while keeping poles and residues in float32 for precision.
This is the recommended inference setup::

    model.to_bfloat16_except_poles_residues()
    model.precompute_filters(L=8192, device=device)

Usage
-----
Dry-run (no GPU, no model weights)::

    evo-amr embed --config configs/experiments/evo_minirocket_ampicillin.example.yaml

Single-layer extraction (GPU required)::

    outputs = model.extract_layer_hidden_states(
        input_ids=tokenised_genome,   # shape: (1, n_tokens)
        target_layer=10,
    )
    hidden = outputs['hidden_states']   # shape: (1, n_tokens, hidden_dim)
    pooled = hidden.mean(dim=1)         # shape: (1, hidden_dim) — global mean pool

Multi-layer diagnostic::

    layer_states = model.extract_multiple_layers(
        input_ids=tokenised_genome,
        layer_list=list(range(32)),
    )
    # layer_states[k] has shape (1, n_tokens, hidden_dim)

Full embedding pipeline::

    evo-embed --input manifest.csv \\
              --output embeddings/evo_l10/ \\
              --layer 10 \\
              --batch_size 8

The ``evo-embed`` entry point wraps ``embedding_pipeline/embedding_generator.py``,
which chunks long genomes into 8192-token windows with configurable stride,
extracts Layer 10 activations via ``extract_layer_hidden_states``, and writes
per-genome HDF5 artifacts.

HDF5 output schema (per genome)
--------------------------------
::

    /genome_id/layer10/embeddings   float32   shape (n_windows, hidden_dim)
    /genome_id/layer10/pooled       float32   shape (hidden_dim,)
    /genome_id/metadata             attrs: species, antibiotic, source_dataset

``n_windows`` depends on genome length and the sliding-window stride.
``pooled`` is the global mean across all token positions in all windows.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from evo_amr.diagnostics import validate_layer_choice


@dataclass(frozen=True)
class EvoEmbeddingConfig:
    """Configuration for an Evo Layer 10 embedding extraction run.

    All fields map directly to CLI flags for ``embedding_pipeline/embedding_generator.py``.

    Parameters
    ----------
    model:
        Model directory name inside ``models/``.  ``evo-1-8k-base`` is the
        variant used in all thesis experiments.  ``evo_1_131k_base`` is a
        larger variant with extended context and is the source of
        ``modified_model.py``.
    extraction_layer:
        0-based block index from which to read ``hidden_states[extraction_layer]``.
        Default is 10, as determined by the layer diagnostic suite.
    dtype:
        Inference precision.  ``bfloat16`` matches the model's training dtype
        and avoids float16 dynamic range loss on small activations.
    window_size:
        Token count per inference pass.  Must not exceed 8 192 for
        ``evo-1-8k-base``.  Genome is processed in overlapping windows.
    stride:
        Step between consecutive windows.  Windows overlap by
        ``window_size - stride`` tokens.
    output_dir:
        Directory where per-genome ``.h5`` files are written by the backend.
    """

    model: str = "evo-1-8k-base"
    extraction_layer: int = 10
    dtype: str = "bfloat16"
    window_size: int = 8192
    stride: int = 4096
    output_dir: Path = field(default_factory=lambda: Path("embeddings/evo_l10"))

    @classmethod
    def from_experiment_yaml(cls, config: dict) -> "EvoEmbeddingConfig":
        """Parse from the ``representation`` block of an experiment YAML."""
        rep = config.get("representation", {})
        return cls(
            model=rep.get("model", "evo-1-8k-base"),
            extraction_layer=int(rep.get("extraction_layer", 10)),
            dtype=rep.get("dtype", "bfloat16"),
            window_size=int(rep.get("window_size", 8192)),
            stride=int(rep.get("stride", 4096)),
            output_dir=Path(rep.get("output", "embeddings/evo_l10")),
        )


class EvoEmbeddingAdapter:
    """Dry-run-aware adapter wrapping ``embedding_pipeline/embedding_generator.py``.

    This adapter is kept deliberately thin.  It does not import torch or load
    any model weights, so it is safe to use in CI, dry-run validation, and
    testing contexts that lack GPU or model checkpoints.

    In dry-run mode (``--execute`` omitted from the CLI) it renders the exact
    backend command that *would* be launched.  In execute mode the CLI
    delegates to the backend script as a subprocess.

    Parameters
    ----------
    config:
        Embedding configuration.
    backend_script:
        Path to the extraction entry point.  Defaults to
        ``embedding_pipeline/embedding_generator.py``, which uses
        ``models/evo_1_131k_base/modified_model.py`` for hidden state
        extraction via ``extract_layer_hidden_states``.
    """

    def __init__(
        self,
        config: EvoEmbeddingConfig,
        backend_script: Path | None = None,
    ) -> None:
        self.config = config
        self.backend_script = backend_script or Path(
            "embedding_pipeline/embedding_generator.py"
        )

    def dry_run_command(self, manifest_path: str | Path) -> str:
        """Return the shell command that would launch embedding extraction."""
        return (
            f"python {self.backend_script}"
            f" --input {manifest_path}"
            f" --output {self.config.output_dir}"
            f" --model {self.config.model}"
            f" --layer {self.config.extraction_layer}"
            f" --dtype {self.config.dtype}"
            f" --window_size {self.config.window_size}"
            f" --stride {self.config.stride}"
        )

    def describe(self) -> str:
        """Human-readable plan description for dry-run output."""
        _, layer_note = validate_layer_choice(self.config.extraction_layer)
        return (
            f"Evo embedding adapter\n"
            f"  backend:  {self.backend_script}\n"
            f"  model:    {self.config.model}\n"
            f"  layer:    {self.config.extraction_layer}"
            f" (deepest stable layer by diagnostic suite)\n"
            f"  api:      model.extract_layer_hidden_states(input_ids, target_layer={self.config.extraction_layer})\n"
            f"  dtype:    {self.config.dtype}\n"
            f"  window:   {self.config.window_size} tokens\n"
            f"  stride:   {self.config.stride} tokens\n"
            f"  output:   {self.config.output_dir}\n"
            f"  note:     {layer_note}\n"
        )
