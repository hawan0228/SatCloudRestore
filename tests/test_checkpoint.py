import numpy as np
import pytest
import torch

from satcloudrestore.utils import (
    capture_random_state,
    load_checkpoint,
    resolve_device,
    restore_random_state,
    save_checkpoint,
    seed_everything,
    validate_experiment_name,
)


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "model.pt"; tensor = torch.randn(2)
    save_checkpoint(path, {"model": {"weight": tensor}, "epoch": 1})
    loaded = load_checkpoint(path)
    assert torch.equal(loaded["model"]["weight"], tensor)


def test_seed_reproducibility():
    seed_everything(42); first = torch.randn(4)
    seed_everything(42); second = torch.randn(4)
    assert torch.equal(first, second)


def test_random_state_resume_roundtrip():
    seed_everything(7)
    state = capture_random_state()
    expected_torch = torch.randn(3)
    expected_numpy = np.random.rand(3)
    restore_random_state(state)
    assert torch.equal(torch.randn(3), expected_torch)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)


def test_explicit_unavailable_cuda_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicitly requested"):
        resolve_device("cuda")


def test_experiment_name_validation():
    assert validate_experiment_name("pilot_gpu-20260906") == "pilot_gpu-20260906"
    with pytest.raises(ValueError):
        validate_experiment_name("../overwrite")
