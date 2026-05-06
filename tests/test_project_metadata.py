from pathlib import Path


def test_readme_describes_current_repository_structure():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "research/" in readme
    assert "src/evo_amr/" in readme
    assert "docs/migration_map.md" in readme
    assert "tests/" in readme
    assert "temp_trash" not in readme
    assert "/trash" not in readme


def test_gitignore_allows_tests_and_excludes_large_artifacts():
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "\ntests/\n" not in gitignore
    assert "*.h5" in gitignore
    assert "*.safetensors" in gitignore
    assert "!examples/**/*.csv" in gitignore
    assert "!docs/cross_species_amr_prediction_thesis.pdf" in gitignore
    assert "minirocket/**/trash/" in gitignore
