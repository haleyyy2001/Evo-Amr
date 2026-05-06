from evo_amr.pipelines import list_pipelines


def test_pipeline_catalog_exposes_rebuilt_project_systems():
    pipelines = {pipeline.name: pipeline for pipeline in list_pipelines()}

    assert "classical-benchmarking" in pipelines
    assert "trainable-amr-models" in pipelines
    assert "hpc-orchestration" in pipelines
    assert "evo_amr.baselines.registry" in pipelines["classical-benchmarking"].clean_modules


def test_pipeline_catalog_filters_by_status():
    pipelines = list_pipelines(status="SCAFFOLD_READY")

    assert pipelines
    assert all(pipeline.status == "SCAFFOLD_READY" for pipeline in pipelines)
