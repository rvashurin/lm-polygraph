import numpy as np
import torch

from lm_polygraph.estimators import SAELatentEntropy, SAEEffectiveNumFeatures
from lm_polygraph.stat_calculators.sae import _BatchTopKSAEEncoder


def test_sae_latent_entropy():
    estimator = SAELatentEntropy()
    stats = {"sae_latent_activations": np.array([[1.0, 1.0], [2.0, 0.0]])}

    result = estimator(stats)

    assert np.allclose(result, [np.log(2.0), 0.0])


def test_sae_effective_num_features():
    estimator = SAEEffectiveNumFeatures()
    stats = {"sae_latent_activations": np.array([[1.0, 1.0], [2.0, 0.0]])}

    result = estimator(stats)

    assert np.allclose(result, [2.0, 1.0])


def test_sae_estimators_handle_zero_activation_mass():
    stats = {"sae_latent_activations": np.array([[0.0, 0.0]])}

    assert np.allclose(SAELatentEntropy()(stats), [0.0])
    assert np.allclose(SAEEffectiveNumFeatures()(stats), [0.0])


def test_batch_top_k_sae_encoder_loads_torch_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "ae.pt"
    torch.save(
        {
            "encoder.weight": torch.eye(2),
            "encoder.bias": torch.zeros(2),
            "b_dec": torch.zeros(2),
            "threshold": torch.tensor(0.5),
            "k": torch.tensor(1),
        },
        checkpoint_path,
    )
    encoder = _BatchTopKSAEEncoder(
        repo_id="unused",
        sae_path=str(checkpoint_path),
        device="cpu",
        dtype="float32",
    )

    latents = encoder.encode(torch.tensor([[1.0, 0.25]]))

    assert torch.allclose(latents, torch.tensor([[1.0, 0.0]]))
