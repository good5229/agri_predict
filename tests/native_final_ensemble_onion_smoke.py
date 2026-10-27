"""Native Torch smoke checks for paired final-ensemble optimization mechanics."""
from __future__ import annotations

import datetime as dt
import statistics
import tempfile
from pathlib import Path
import torch
from torch import nn

from agri_data import direct_price_path as frozen
from agri_data import final_ensemble_onion as onion


def main() -> None:
    frozen._seed_everything(torch, onion.SEED)
    a = frozen.build_model(onion.CONTEXT_DAYS, onion.REPAIR, torch=torch, nn=nn, F=torch.nn.functional)
    state_a = {k: v.clone() for k, v in a.state_dict().items()}
    perm_a = []
    for epoch in range(1, 4):
        values = list(range(129))
        import random
        random.Random(onion.SEED + epoch).shuffle(values)
        perm_a.append(values)
    frozen._seed_everything(torch, onion.SEED)
    b = frozen.build_model(onion.CONTEXT_DAYS, onion.REPAIR, torch=torch, nn=nn, F=torch.nn.functional)
    state_b = b.state_dict()
    assert all(torch.equal(value, state_b[key]) for key, value in state_a.items())
    perm_b = []
    for epoch in range(1, 4):
        values = list(range(129))
        import random
        random.Random(onion.SEED + epoch).shuffle(values)
        perm_b.append(values)
    assert perm_a == perm_b
    # A synthetic batch where every horizon's twofold blend gradient is exactly
    # half the direct gradient before the optimizer's F-only factor of two.
    rows = [{"targets": [100.0] * 10, "targetMask": [True] * 10}]
    p, base = [[90.0] * 10], [[100.0] * 10]
    ld = onion.weighted_price_loss(p, rows, 100.0)
    lf = onion.weighted_price_loss(p, rows, 100.0, blend=[[95.0] * 10])
    assert abs(2 * lf - ld) < 1e-12
    # Exercise one real optimizer epoch and a completed-checkpoint restart on
    # fully synthetic records. No source cache or target database is consulted.
    with tempfile.TemporaryDirectory(prefix="onion-train-smoke-") as temp:
        onion.OUT = Path(temp)
        def record(day: int, offset: float) -> dict:
            origin = dt.date(2025, 1, day)
            columns = []
            for column in range(11):
                columns.append({"normalized": ([0.01 * (day + column)] * 84 if column == 0 else [0.] * 84),
                    "observedMask": ([1] * 84 if column == 0 else [0] * 84),
                    "active": column == 0, "role": 0 if column == 0 else 1,
                    "metadata": [0.] * 4})
            return {"product": "onion", "origin": origin,
                "window": {"p0": 100. + offset, "representativeSigma": .1, "columns": columns},
                "calendar": [[0., 1., 0., 1.]] * 10,
                "targets": [101. + offset + h for h in range(10)],
                "targetMask": [True] * 10,
                "targetDays": [(origin + dt.timedelta(days=h+1)).isoformat() for h in range(10)]}
        train = [record(1, 0.), record(2, 1.)]
        validation = [record(3, 2.), record(4, 3.)]
        train[0]["targetMask"][9] = False
        validation[0]["targetMask"][9] = False
        b_train = {(r["origin"], h): 100. + h for r in train for h in range(1, 11)
                   if r["targetMask"][h-1]}
        b_validation = {(r["origin"], h): 100. + h for r in validation for h in range(1, 11)
                        if r["targetMask"][h-1]}
        paired_validation = onion._with_b(validation, b_validation)
        base_pred = [r["B"] for r in paired_validation]
        baseline_metrics = onion._metrics(paired_validation, base_pred, base_pred)["baseline"]
        checkpoint = Path(temp) / "checkpoints" / "inner.pt"
        kwargs = dict(b_train=b_train, b_validation=b_validation, q=110., penalty=1.,
            fit_id="synthetic-inner", checkpoint_path=checkpoint, optimizer_scale=2.,
            max_epochs=1, baseline_validation=baseline_metrics, provenance={"test": "synthetic"},
            phase="D25", objective="inner-F", penalty_value=1.)
        first = onion.train_one(train, validation, **kwargs)
        second = onion.train_one(train, validation, **kwargs)
        assert first["selectedEpoch"] == second["selectedEpoch"] == 1
        assert first["history"] == second["history"]
        assert abs(first["elapsedSeconds"] - second["elapsedSeconds"]) < 1e-9
        for objective in ("D", "F"):
            fixed_path = Path(temp) / "checkpoints" / f"full-{objective}.pt"
            first_fixed = onion.fit_fixed(train, b_train, q=110., penalty=1., epochs=1,
                fit_id=f"synthetic-full-{objective}", checkpoint_path=fixed_path,
                objective=objective, provenance={"test": "synthetic"}, phase="D25")
            second_fixed = onion.fit_fixed(train, b_train, q=110., penalty=1., epochs=1,
                fit_id=f"synthetic-full-{objective}", checkpoint_path=fixed_path,
                objective=objective, provenance={"test": "synthetic"}, phase="D25")
            assert first_fixed["fitIdentity"] == second_fixed["fitIdentity"]
            assert first_fixed["elapsedSeconds"] == second_fixed["elapsedSeconds"]
    print("paired initialization, row order, synthetic F scaling, one training epoch, and completed restart passed")


if __name__ == "__main__":
    main()
