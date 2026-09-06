import pytest

from satcloudrestore.dataset import stratified_split, validate_manifest


def test_split_is_disjoint_and_sized():
    samples = [(f"class_{i % 3}/image_{i}.jpg", i % 3) for i in range(30)]
    sizes = {"train": 18, "validation": 6, "test": 6}
    rows = stratified_split(samples, sizes, seed=42)
    validate_manifest(rows, sizes)
    assert len({r["path"] for r in rows}) == 30
    assert len({r["image_id"] for r in rows}) == 30


def test_insufficient_data_is_clear():
    with pytest.raises(ValueError, match="requested splits"):
        stratified_split([("a.jpg", 0)], {"train": 1, "validation": 1, "test": 1})
