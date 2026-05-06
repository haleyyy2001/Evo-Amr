from pathlib import Path

from config.config_manager import ConfigManager


def test_config_manager_detects_repository_root():
    config = ConfigManager()

    assert config.base_dir == Path(__file__).resolve().parents[1]
    assert config.config_dir == config.base_dir / "config"


def test_config_manager_resolves_configured_paths():
    config = ConfigManager()

    assert config.get_model_path("evo_1_131k_base") == Path("models/evo_1_131k_base")
    assert str(config.get_path("data", "sequences")).endswith(
        "sequences/raw/nucleotide"
    )


def test_setup_environment_contains_cache_and_runtime_variables():
    env = ConfigManager().setup_environment()

    assert env["HF_HOME"] == "cache/hf"
    assert env["TORCH_HOME"] == "cache/torch"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
