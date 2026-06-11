from lm_polygraph.stat_calculators.sae import SAELatentActivationsCalculator


def load_stat_calculator(config, builder):
    return SAELatentActivationsCalculator(
        repo_id=config.repo_id,
        sae_path=config.sae_path,
        layer=config.layer,
        module_path=getattr(config, "module_path", None),
        hf_cache=getattr(config, "hf_cache", None),
        hf_token=getattr(config, "hf_token", None),
        device=getattr(config, "device", "auto"),
        dtype=getattr(config, "dtype", "auto"),
        use_threshold=getattr(config, "use_threshold", True),
        fallback_to_top_k=getattr(config, "fallback_to_top_k", True),
        k=getattr(config, "k", None),
        token_positions=getattr(config, "token_positions", "prediction"),
        aggregation=getattr(config, "aggregation", "mean"),
    )
