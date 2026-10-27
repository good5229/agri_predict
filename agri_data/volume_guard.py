"""Retrospective volume-quality filter for the radish low-volume experiment."""
from __future__ import annotations

import datetime as dt
import math
import statistics
from typing import Any

from . import modeling

EVENT_DAY = dt.date(2023, 1, 21)
EVALUATION_START = dt.date(2017, 1, 1)
EVALUATION_END = dt.date(2026, 9, 27)
HORIZONS = (1, 5, 10)
LOOKBACK_OBSERVATIONS = 28
MIN_PRIOR_OBSERVATIONS = 14
VOLUME_FRACTION = 0.01


def _observations(product: dict[str, Any]) -> dict[dt.date, tuple[float, float]]:
    """Return validated date -> (price, net volume) observations."""
    result: dict[dt.date, tuple[float, float]] = {}
    for row in product.get("observations", []):
        day = modeling._parse_day(row.get("date"))
        price = modeling._finite_positive(row.get("actual"))
        volume = modeling._finite_positive(row.get("volumeKg"))
        if day is None or price is None or volume is None:
            raise ValueError(f"invalid radish observation: {row!r}")
        if day in result:
            raise ValueError(f"duplicate radish observation date: {day.isoformat()}")
        result[day] = (price, volume)
    return dict(sorted(result.items()))


def quarantine_dates(series: dict[dt.date, tuple[float, float]],
                     price_ratio_threshold: float | None = None) -> list[dict[str, Any]]:
    """Select low-volume dates, optionally requiring an extreme prior-only price ratio."""
    if price_ratio_threshold is not None and price_ratio_threshold < 1.0:
        raise ValueError("price ratio threshold must be at least 1")
    days = sorted(series)
    flagged: list[dict[str, Any]] = []
    for index, day in enumerate(days):
        prior_days = days[max(0, index - LOOKBACK_OBSERVATIONS):index]
        if len(prior_days) < MIN_PRIOR_OBSERVATIONS:
            continue
        prior_volumes = [series[prior][1] for prior in prior_days]
        median_volume = statistics.median(prior_volumes)
        volume = series[day][1]
        prior_prices = [series[prior][0] for prior in prior_days]
        median_price = statistics.median(prior_prices)
        price = series[day][0]
        ratio = price / median_price if median_price > 0 else None
        extreme_price = (price_ratio_threshold is not None and ratio is not None
                         and (ratio >= price_ratio_threshold or ratio <= 1.0 / price_ratio_threshold))
        low_volume = median_volume > 0 and volume < median_volume * VOLUME_FRACTION
        if low_volume and (price_ratio_threshold is None or extreme_price):
            flagged.append({"date": day.isoformat(), "volumeKg": volume,
                            "priorObservedDays": len(prior_days),
                            "prior28ObservedDayMedianKg": median_volume,
                            "fractionOfMedian": volume / median_volume,
                            "price": price,
                            "prior28ObservedDayMedianPrice": median_price,
                            "priceToPriorMedianRatio": ratio,
                            "priceRatioThreshold": price_ratio_threshold})
    return flagged


def quality_series(series: dict[dt.date, tuple[float, float]],
                   quarantined: list[dict[str, Any]]) -> dict[dt.date, tuple[float, float]]:
    excluded = {dt.date.fromisoformat(row["date"]) for row in quarantined}
    return {day: value for day, value in series.items() if day not in excluded}


def source_day(series: dict[dt.date, tuple[float, float]], origin: dt.date) -> dt.date | None:
    eligible = [day for day in series if day <= origin]
    return max(eligible) if eligible else None


def persistence_predictions(series: dict[dt.date, tuple[float, float]],
                           origins: list[dt.date], horizons: tuple[int, ...] = HORIZONS
                           ) -> list[dict[str, Any]]:
    result = []
    for origin in origins:
        src = source_day(series, origin)
        if src is None or (origin - src).days > modeling.MAX_SOURCE_AGE_DAYS:
            continue
        for horizon in horizons:
            result.append({"origin": origin.isoformat(),
                           "target": (origin + dt.timedelta(days=horizon)).isoformat(),
                           "horizon": horizon, "predicted": series[src][0],
                           "sourceDate": src.isoformat()})
    return result


def loss_metrics(actual: float, predicted: float) -> dict[str, float]:
    ae = abs(predicted - actual)
    return {"absoluteError": ae, "absolutePercentageError": ae / abs(actual) * 100.0}


def _normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def dm_calendar_hac(rows: list[tuple[dt.date, float]], bandwidth: int) -> dict[str, Any]:
    """Two-sided DM normal approximation with Bartlett calendar-day HAC."""
    n = len(rows)
    if n < 30:
        return {"n": n, "status": "inference_not_supported_under_30_pairs", "pValue": None,
                "meanLossDifference": (sum(v for _, v in rows) / n if n else None),
                "bandwidthCalendarDays": bandwidth}
    values = [value for _, value in rows]
    scale = max(1.0, *(abs(value) for value in values))
    if max(values) - min(values) <= 64.0 * math.ulp(scale):
        return {"n": n, "status": "inference_not_supported_constant_loss_difference", "pValue": None,
                "meanLossDifference": sum(values) / n, "bandwidthCalendarDays": bandwidth}
    by_day = dict(rows)
    mean = sum(value for _, value in rows) / n
    centered = {day: value - mean for day, value in rows}
    gamma0 = sum(value * value for value in centered.values()) / n
    long_run = gamma0
    for lag in range(1, bandwidth + 1):
        cross = sum(value * centered.get(day + dt.timedelta(days=lag), 0.0)
                    for day, value in centered.items()) / n
        long_run += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * cross
    if not math.isfinite(long_run) or long_run <= 0:
        return {"n": n, "status": "inference_not_supported_nonpositive_hac_variance", "pValue": None,
                "meanLossDifference": mean, "bandwidthCalendarDays": bandwidth}
    statistic = mean / math.sqrt(long_run / n)
    return {"n": n, "status": "tested", "meanLossDifference": mean,
            "hacLongRunVariance": long_run, "statistic": statistic,
            "pValue": _normal_two_sided_p(statistic), "bandwidthCalendarDays": bandwidth}


def holm_adjust(items: list[dict[str, Any]]) -> None:
    """Apply Holm correction in place; unsupported tests remain in the family at p=1."""
    ordered = sorted(range(len(items)), key=lambda i: (items[i].get("pValue")
                                                       if items[i].get("pValue") is not None else 1.0, i))
    running = 0.0
    m = len(items)
    for rank, idx in enumerate(ordered):
        p = items[idx].get("pValue")
        raw = 1.0 if p is None else p
        running = max(running, min(1.0, (m - rank) * raw))
        items[idx]["holmPValue"] = running
        items[idx]["holmFamilySize"] = m


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "mae": None, "mapePercent": None}
    return {"n": len(rows), "mae": sum(r["absoluteError"] for r in rows) / len(rows),
            "mapePercent": sum(r["absolutePercentageError"] for r in rows) / len(rows)}
