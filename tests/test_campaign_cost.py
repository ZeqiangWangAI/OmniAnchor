from scripts.collect_campaign_cost import allocated_gpus


def test_slurm_generic_and_typed_gpu_tres_are_not_double_counted():
    assert allocated_gpus("cpu=4,gres/gpu=1,gres/gpu:a5000=1,mem=48G") == 1
    assert allocated_gpus("cpu=4,gres/gpu:a5000=2,gres/gpu:ada=1") == 3
    assert allocated_gpus("cpu=4,mem=16G") == 0


def test_clean_acceptance_allocations_are_included_without_local_evidence_folders():
    import re
    from scripts.collect_campaign_cost import RUN_PREFIXES
    pattern = "(?:" + "|".join(RUN_PREFIXES) + r")-(\d+)"
    for name in ["clean-environment-47467", "clean-smoke-47472", "clean-e5-47491"]:
        assert re.fullmatch(pattern, name)
    for name in ["clean-smoke-evidence-47472", "clean-e5-evidence-47491", "unrelated-47491"]:
        assert re.fullmatch(pattern, name) is None
