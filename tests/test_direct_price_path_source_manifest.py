import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agri_data import direct_price_path_source_manifest as source_manifest


def test_canonical_json_and_self_hash_are_stable():
    document = {"é": [1, True], "a": {"z": None}}
    expected = '{"a":{"z":null},"é":[1,true]}\n'.encode("utf-8")
    assert source_manifest.canonical_json(document) == expected
    signed = dict(document)
    signed["sourceManifestSha256"] = source_manifest.sha256_bytes(expected)
    source_manifest._verify_self_hash(signed, "sourceManifestSha256", "fixture")


def test_self_hash_rejects_mutation():
    doc = {"one": 1}
    doc["cacheSha256"] = source_manifest.sha256_bytes(source_manifest.canonical_json(doc))
    source_manifest._verify_self_hash(doc, "cacheSha256", "fixture")
    doc["one"] = 2
    with pytest.raises(ValueError, match="internal cacheSha256 mismatch"):
        source_manifest._verify_self_hash(doc, "cacheSha256", "fixture")


@pytest.mark.parametrize(
    ("fit_id", "boundary"),
    [
        ("dev-cabbage-L84-U-base-F23-inner", "2022-10-01"),
        ("dev-cabbage-L84-U-base-F23-refit", "2023-01-01"),
        ("dev-radish-L168-M-base-F24-inner", "2023-10-01"),
        ("dev-radish-L168-M-base-F24-refit", "2024-01-01"),
        ("year2025-garlic-L84-M-base", "2025-01-01"),
    ],
)
def test_development_boundary_is_derived_from_frozen_fold(fit_id, boundary):
    config = {
        "folds": {
            "F23": {"innerFitBoundary": "2022-10-01", "refitBoundary": "2023-01-01"},
            "F24": {"innerFitBoundary": "2023-10-01", "refitBoundary": "2024-01-01"},
        }
    }
    assert source_manifest._development_boundary({"fitId": fit_id}, config) == boundary


def _write_baseline(project: Path, *, wrong_loader_reference=False):
    rhs_dir = project / source_manifest.RHS_DIR
    rhs_dir.mkdir(parents=True)
    csv_path = rhs_dir / "validation-predictions.csv"
    csv_path.write_text("already checked by the reviewed loader\n", encoding="utf-8")
    manifest = {"forecasts": [{"conditions": {"B": {}, "C": {}, "D": {}, "P": {}}}],
                "forecastFile": csv_path.name,
                "forecastFileSha256": source_manifest.sha256_file(csv_path)}
    manifest_path = rhs_dir / "validation-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected = source_manifest.sha256_file(manifest_path)
    def loader(_phase):
        returned_path = (rhs_dir / "other-manifest.json") if wrong_loader_reference else manifest_path
        return ({**manifest, "path": returned_path,
                 "forecastCsvPath": csv_path}, [{}] * 3020)
    return expected, loader


def test_baseline_wrapper_uses_reviewed_loader_and_pins_its_files(tmp_path):
    expected, loader = _write_baseline(tmp_path)
    result = source_manifest._baseline_source(tmp_path, "V", expected, loader)
    assert result["nestedForecasts"] == 1
    assert result["conditionRows"] == 4
    assert result["eligibleKeys"] == 3020
    assert result["forecastFile"] == "validation-predictions.csv"


def test_baseline_rejects_different_manifest_returned_by_reviewed_loader(tmp_path):
    expected, loader = _write_baseline(tmp_path, wrong_loader_reference=True)
    with pytest.raises(ValueError, match="returned another manifest"):
        source_manifest._baseline_source(tmp_path, "V", expected, loader)


def test_baseline_rejects_changed_pinned_manifest(tmp_path):
    rhs_dir = tmp_path / source_manifest.RHS_DIR
    rhs_dir.mkdir(parents=True)
    (rhs_dir / "validation-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or changed"):
        source_manifest._baseline_source(tmp_path, "V", "0" * 64)


def test_year2025_singular_completed_fit_is_included():
    cache = {"completedFit": {"fitId": "year2025-cabbage-L84-U-base"}}
    assert [row["fitId"] for row in source_manifest._completed_fit_records(cache)] == [
        "year2025-cabbage-L84-U-base"
    ]


def test_fixed_epoch_metadata_matches_final_update_not_checkpoint_best(tmp_path):
    selected = [{"channelId": ["110001"], "grain": "market"}]
    model = {"schemaVersion": 1, "fitId": "V-202604-cabbage-U",
             "experiment": source_manifest.EXPERIMENT, "contextDays": 84,
             "inputType": "U", "repair": "base", "bestEpoch": 20,
             "selectedInputChannels": selected, "stateDict": {"weights": b"x"}}
    checkpoint = {"fitId": model["fitId"], "product": "cabbage", "contextDays": 84,
                  "inputType": "U", "repair": "base", "epoch": 20,
                  "bestEpoch": 7, "fitLimits": {"fixedEpochs": 20},
                  "fitIdentity": "fit-identity"}
    model_path, checkpoint_path = tmp_path / "model.pt", tmp_path / "checkpoint.pt"
    model_path.write_bytes(b"model")
    checkpoint_path.write_bytes(b"checkpoint")
    fit = {"fitId": model["fitId"], "product": "cabbage", "contextDays": 84,
           "inputType": "U", "repairSetting": "base", "bestEpoch": 20,
           "fixedEpoch": 20, "modelPath": str(model_path.name),
           "modelSha256": source_manifest.sha256_file(model_path),
           "checkpointPath": str(checkpoint_path.name),
           "checkpointSha256": source_manifest.sha256_file(checkpoint_path)}

    class FakeTorch:
        @staticmethod
        def load(path, **_kwargs):
            return model if Path(path).name == model_path.name else checkpoint

    result = source_manifest._model_metadata(tmp_path, fit, FakeTorch)
    assert result["selectedChannels"] == selected
    assert result["checkpointFitIdentity"] == "fit-identity"


def test_model_metadata_checks_checkpoint_product_and_shared_schema(tmp_path):
    model = {"fitId": "dev-cabbage-L84-U-base-F23-inner",
             "experiment": source_manifest.EXPERIMENT, "contextDays": 84,
             "inputType": "U", "repair": "base", "bestEpoch": 3,
             "selectedInputChannels": [], "stateDict": {}}
    checkpoint = {"fitId": model["fitId"], "product": "radish", "contextDays": 84,
                  "inputType": "U", "repair": "base", "bestEpoch": 3,
                  "fitLimits": {"fixedEpochs": None}, "fitIdentity": "identity"}
    for name in ("model.pt", "checkpoint.pt"):
        (tmp_path / name).write_bytes(name.encode())
    fit = {"fitId": model["fitId"], "product": "cabbage", "contextDays": 84,
           "inputType": "U", "repair": "base", "bestEpoch": 3,
           "weightPath": "model.pt", "weightSha256": source_manifest.sha256_file(tmp_path / "model.pt"),
           "checkpointPath": "checkpoint.pt",
           "checkpointSha256": source_manifest.sha256_file(tmp_path / "checkpoint.pt")}

    class FakeTorch:
        @staticmethod
        def load(path, **_kwargs):
            return model if Path(path).name == "model.pt" else checkpoint

    with pytest.raises(ValueError, match="checkpoint metadata product mismatch"):
        source_manifest._model_metadata(tmp_path, fit, FakeTorch)
