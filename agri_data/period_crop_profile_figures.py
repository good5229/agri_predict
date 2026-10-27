"""Plot the already calculated period/crop diagnostics tables.

This module is deliberately a renderer: it does not load source data, derive
metrics, score forecasts, fit models, or make database/API calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


CROPS = ("cabbage", "radish", "onion", "garlic")
CROP_KO = {"cabbage": "배추", "radish": "무", "onion": "양파", "garlic": "깐마늘"}
MODELS = ("P", "G-B", "B0", "L-L")


def _need(tables: Mapping[str, pd.DataFrame], key: str, columns: tuple[str, ...]) -> pd.DataFrame:
    if key not in tables:
        raise ValueError(f"required plot table missing: {key}")
    frame = tables[key]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"{key} missing plot columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError(f"required plot table has no supported rows: {key}")
    return frame


def _crop(value: str) -> str:
    return CROP_KO.get(str(value), str(value))


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")


def create_period_crop_profile_figures(
    tables: Mapping[str, pd.DataFrame], output_dir: str | Path
) -> dict[str, str]:
    """Render the nine requested PNGs from precomputed DataFrames.

    Missing plot support raises an error so callers can mark the figure as
    ``not_supported`` instead of publishing a substitute or placeholder.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")
    if not font_path.is_file():
        raise RuntimeError(f"required Korean font unavailable: {font_path}")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    made: dict[str, str] = {}

    # 1. Daily and dekad price-change distributions with their own support.
    daily = _need(tables, "daily_price_returns", ("crop", "simple_return"))
    dekad = _need(tables, "dekad_price_returns", ("crop", "simple_return"))
    dist = _need(tables, "crop_distribution", ("crop", "observed_days", "valid_dekads"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, frame, title, scale in (
        (axes[0, 0], daily, "인접 달력일 가격 변화율", 100),
        (axes[0, 1], dekad, "인접 달력 순 가격 변화율", 100),
    ):
        for c in CROPS:
            q = frame.loc[frame.crop == c, "simple_return"].dropna().to_numpy(float)
            if q.size:
                ax.hist(q * scale, bins=45, density=True, histtype="step", linewidth=1.5, label=_crop(c))
        ax.set_title(title)
        ax.set_xlabel("가격 변화율 (%)")
        ax.set_ylabel("밀도")
        ax.legend(ncol=2, fontsize=8)
    x = np.arange(len(CROPS))
    d = dist.set_index("crop").reindex(CROPS)
    axes[1, 0].bar(x, d["observed_days"].astype(float), color="#4472C4")
    axes[1, 0].set_xticks(x, [_crop(c) for c in CROPS])
    axes[1, 0].set_title("유효 일별 가격 관측 지원")
    axes[1, 0].set_ylabel("관측일 수 (일)")
    axes[1, 1].bar(x, d["valid_dekads"].astype(float), color="#70AD47")
    axes[1, 1].set_xticks(x, [_crop(c) for c in CROPS])
    axes[1, 1].set_title("유효 순별 가격 관측 지원")
    axes[1, 1].set_ylabel("관측 순 수 (순)")
    p = out / "crop_price_volatility.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 2. Within-year centered seasonal levels, with available year support.
    season = _need(tables, "seasonal_pattern", ("crop", "year", "dekad", "centered_log_price", "n_days"))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for ax, c in zip(axes.flat, CROPS):
        q = season.loc[season.crop == c]
        for year, g in q.groupby("year", sort=True):
            g = g.set_index("dekad").reindex(range(1, 37))
            values = g.centered_log_price.to_numpy(dtype=float)
            ax.plot(np.arange(1, 37), values, marker=".", linewidth=1, alpha=.65, label=str(year))
        years = q.loc[q.centered_log_price.notna(), "year"].nunique()
        ax.axhline(0, color="black", linewidth=.6)
        ax.set_title(f"{_crop(c)} · 관측 연도 {years}년")
        ax.set_xlabel("연중 순 (1–36)")
        ax.set_ylabel("연도 내 중심 로그가격 차이")
        ax.set_xlim(1, 36)
        ax.legend(title="연도", ncol=2, fontsize=7)
    p = out / "crop_seasonal_patterns.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 3. Annual price-day and positive-volume support/distribution.
    annual = _need(tables, "annual_volume_support", ("crop", "year", "n_valid_days", "n_valid_dekads", "volume_median", "common_price_days", "common_price_dekads"))
    years = sorted(annual.year.dropna().astype(int).unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, c in zip(axes.flat, CROPS):
        q = annual.loc[annual.crop == c].sort_values("year")
        ax.bar(q.year, q.n_valid_days, color="#4472C4", alpha=.8, label="가격 관측일")
        ax.plot(q.year, q.common_price_days, color="#222222", marker="s", linestyle="--",
                linewidth=1, label="네 품목 공통 관측일")
        ax.set_title(_crop(c)); ax.set_xlabel("연도"); ax.set_ylabel("유효 가격 관측일 (일)")
        twin = ax.twinx()
        twin.plot(q.year, q.volume_median, color="#C00000", marker="o", label="양수 중량 중앙값")
        twin.set_ylabel("일별 양수 거래중량 중앙값 (kg)")
        ax.set_xticks(years[::max(1, len(years)//8)])
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        # Put dekad counts inside the day-support bars to keep labels away from titles.
        for row in q.itertuples():
            if row.n_valid_days > 0:
                ax.text(row.year, row.n_valid_days * .55,
                        f"{int(row.n_valid_dekads)}/{int(row.common_price_dekads)}",
                        ha="center", va="center", fontsize=5, rotation=90, color="white", clip_on=True)
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, fontsize=6, loc="upper left")
    p = out / "crop_volume_support.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 4. Eight cell MAPE for all four policies and the 10% reference.
    metrics = _need(tables, "metric_reconciliation", ("crop", "horizon", "n", "P_MAPE_pct", "G-B_MAPE_pct", "B0_MAPE_pct", "L-L_MAPE_pct"))
    cells = metrics.loc[metrics.crop.isin(CROPS)].copy()
    cells["label"] = cells.apply(lambda r: f"{_crop(r.crop)} {int(r.horizon)}순\nn={int(r.n)}", axis=1)
    labels = cells.label.tolist(); xx = np.arange(len(cells)); width = .19
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for i, model in enumerate(MODELS):
        ax.bar(xx + (i - 1.5) * width, cells[f"{model}_MAPE_pct"], width, label=model)
    ax.axhline(10, color="#C00000", linewidth=1.3, linestyle="--", label="목표 10%")
    ax.set_xticks(xx, labels); ax.set_ylabel("MAPE (%)"); ax.set_title("품목·목표 거리별 MAPE (셀별 표본 수 포함)")
    ax.legend(ncol=5); ax.grid(axis="y", alpha=.2)
    p = out / "mape8_crop_horizon.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 5. Per-year cell differences and support. Diverging colors make 2026
    # worsening visible while retaining cabbage exceptions cell by cell.
    annual_cmp = _need(tables, "annual_crop_comparison", ("year", "crop", "horizon", "n_forecasts", "G_minus_P_MAPE_pp", "G_minus_P_MAE"))
    years = sorted(annual_cmp.year.astype(int).unique())
    fig, axes = plt.subplots(2, len(years), figsize=(5 * len(years), 8), constrained_layout=True, squeeze=False)
    labels = [f"{_crop(c)}\n{h}순" for c in CROPS for h in (2, 3)]
    for j, year in enumerate(years):
        q = annual_cmp.loc[annual_cmp.year.astype(int) == year].set_index(["crop", "horizon"])
        for i, (metric, title, unit) in enumerate((
            ("G_minus_P_MAPE_pp", "G-P MAPE 차이 (퍼센트포인트)", "pp"),
            ("G_minus_P_MAE", "G-P MAE 차이 (원/kg)", "원/kg"),
        )):
            vals = np.array([q.loc[(c, h), metric] if (c, h) in q.index else np.nan for c in CROPS for h in (2, 3)], dtype=float)
            lim = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 1.0
            ax = axes[i, j]
            image = ax.imshow(vals[None, :], cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            for k, (c, h) in enumerate((pair for c in CROPS for pair in ((c, 2), (c, 3)))):
                n = int(q.loc[(c, h), "n_forecasts"]) if (c, h) in q.index else 0
                val = vals[k]
                label = f"{val:+.2f}\nn={n}" if np.isfinite(val) else f"미지원\nn={n}"
                if np.isfinite(val):
                    r, g, b, _ = image.cmap(image.norm(val))
                    luminance = .2126 * r + .7152 * g + .0722 * b
                    color = "white" if luminance < .45 else "black"
                else:
                    color = "black"
                ax.text(k, 0, label, ha="center", va="center", fontsize=7, color=color)
            ax.set_xticks(range(8), labels, fontsize=7); ax.set_yticks([])
            ax.set_title(f"{year} · {title}")
            fig.colorbar(image, ax=ax, fraction=.03, pad=.02)
    p = out / "year_crop_delta.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 6. Actual and saved P/G price-ratio distributions by each crop/horizon.
    z = _need(tables, "prediction_z", ("crop", "horizon", "actual_z", "P_z", "G-B_z"))
    comp = _need(tables, "error_decomposition", ("diagnostic_type", "crop", "horizon", "actual_z_sd", "P_std_ratio_to_actual", "G-B_std_ratio_to_actual"))
    comp = comp.loc[comp.diagnostic_type == "prediction_compression"]
    if comp.empty:
        raise ValueError("prediction_compression rows unavailable")
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True, sharey=False)
    for ax, c, h in zip(axes.flat, (c for c in CROPS for _ in (2, 3)), (h for _ in CROPS for h in (2, 3))):
        q = z.loc[(z.crop == c) & (z.horizon == h)]
        if q.empty:
            raise ValueError(f"prediction_z support missing for {c} {h}순")
        vals = [q[col].dropna().to_numpy(float) for col in ("actual_z", "P_z", "G-B_z")]
        ax.boxplot(vals, labels=("실제", "P", "G-B"), showfliers=True, whis=(0, 100))
        row = comp.loc[(comp.crop == c) & (comp.horizon == h)]
        if row.empty:
            raise ValueError(f"compression ratios missing for {c} {h}순")
        rr = row.iloc[0]
        ax.set_title(f"{_crop(c)} {h}순 · σ비 P {rr['P_std_ratio_to_actual']:.2f}, G {rr['G-B_std_ratio_to_actual']:.2f}")
        ax.set_ylabel("z = log(가격 / 발행 기준가격)")
        ax.axhline(0, color="black", linewidth=.5)
    p = out / "prediction_compression.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 7. Keep realized jump regimes and origin-available indicators in separate
    # panels; the split is explicit in labels and uses precomputed group scores.
    regimes = _need(tables, "regime_plot_support", ("diagnostic_type", "crop", "horizon", "group", "n_forecasts", "unique_targets", "P_MAPE_pct", "G-B_MAPE_pct", "feature"))
    realized = regimes.loc[regimes.diagnostic_type == "realized_regime"]
    asof = regimes.loc[regimes.diagnostic_type == "asof_feature_tercile"]
    if realized.empty or asof.empty:
        raise ValueError("both realized and as-of regime support are required")
    feature_choices = ("price_sd_28", "volume_coverage_28")
    for feature in feature_choices:
        if not ((asof.feature == feature).any()):
            raise ValueError(f"development-frozen as-of support missing for {feature}")
    # Eight cells each get distinct realized-outcome, price-volatility and
    # observation-support panels.  The criteria remain visibly separate.
    fig, axes = plt.subplots(4, 6, figsize=(20, 15), constrained_layout=True, squeeze=False)
    width = .34
    for ci, c in enumerate(CROPS):
        for hi, h in enumerate((2, 3)):
            row_ix = ci
            col_ix = hi * 3
            cell_name = f"{_crop(c)} {h}순"
            q = realized.loc[(realized.crop == c) & (realized.horizon == h)].set_index("group")
            order = [x for x in ("realized_abs_change_lt20", "realized_abs_change_ge20") if x in q.index]
            if len(order) < 2:
                raise ValueError(f"posthoc realized groups missing for {c} {h}순")
            q = q.reindex(order); xx = np.arange(len(q))
            for j, model in enumerate(("P", "G-B")):
                axes[row_ix, col_ix].bar(xx + (j - .5) * width, q[f"{model}_MAPE_pct"], width, label=model)
            axes[row_ix, col_ix].set_xticks(xx, ["|Δ|<20%", "|Δ|≥20%"])
            axes[row_ix, col_ix].set_title(f"{cell_name} · 사후 정답 구분")
            axes[row_ix, col_ix].set_ylabel("MAPE (%)")
            axes[row_ix, col_ix].legend(fontsize=7)
            for i, (_, item) in enumerate(q.iterrows()):
                top = max(float(item["P_MAPE_pct"]), float(item["G-B_MAPE_pct"]))
                axes[row_ix, col_ix].text(i, top, f"n={int(item.n_forecasts)} · 목표순 {int(item.unique_targets)}",
                                          ha="center", va="bottom", fontsize=6)
            for offset, feature in enumerate(feature_choices, start=1):
                ax = axes[row_ix, col_ix + offset]
                q = asof.loc[(asof.crop == c) & (asof.feature == feature) & (asof.horizon == h)].set_index("group")
                group_order = [x for x in ("low", "middle", "high") if x in q.index]
                if len(group_order) < 2:
                    raise ValueError(f"as-of {feature} groups missing for {c} {h}순")
                q = q.reindex(group_order); xx = np.arange(len(q))
                for j, model in enumerate(("P", "G-B")):
                    ax.bar(xx + (j - .5) * width, q[f"{model}_MAPE_pct"], width,
                           label=model if ci == 0 and hi == 0 else None)
                ax.set_xticks(xx, [f"{g}\nn={int(n)}" for g, n in zip(q.index, q.n_forecasts)])
                short = "발행 전 28일 가격 변동성" if feature == "price_sd_28" else "발행 전 거래량 지원"
                ax.set_title(f"{cell_name} · {short}\n개발구간 경계")
                ax.set_ylabel("MAPE (%)")
                if ci == 0 and hi == 0:
                    ax.legend(fontsize=7)
    p = out / "error_regimes.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 8. Pearson return-correlation matrix for same-calendar dekads, with n.
    corr = _need(tables, "cross_crop_relationships", ("crop_a", "crop_b", "era", "lag_periods", "n", "pearson"))
    q = corr.loc[(corr.era == "all") & (corr.lag_periods == 0)]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    matrix = np.full((4, 4), np.nan); annotations = [["" for _ in CROPS] for _ in CROPS]
    for i, a in enumerate(CROPS):
        for j, b in enumerate(CROPS):
            if i == j:
                continue
            row = q.loc[((q.crop_a == a) & (q.crop_b == b)) | ((q.crop_a == b) & (q.crop_b == a))]
            if not row.empty:
                r = row.iloc[0]
                matrix[i, j] = float(r.pearson) if pd.notna(r.pearson) else np.nan
                annotations[i][j] = f"r={r.pearson:.2f}\nn={int(r.n)}" if pd.notna(r.pearson) else f"미지원\nn={int(r.n)}"
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(4), [_crop(c) for c in CROPS]); ax.set_yticks(range(4), [_crop(c) for c in CROPS])
    for i in range(4):
        for j in range(4):
            if i == j: annotations[i][j] = "같은 품목"
            ax.text(j, i, annotations[i][j], ha="center", va="center", fontsize=9)
    ax.set_title("같은 달력 순 로그수익률 Pearson 상관 · 공통 유효 순")
    fig.colorbar(image, ax=ax, label="Pearson 상관계수")
    p = out / "cross_crop_return_correlation.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    # 9. Crop-by-year fit and evaluation support. The training values are
    # upstream sums over fit origins, so repeated exposure is labeled clearly.
    support = _need(tables, "annual_data_support", ("crop", "year", "train_rows", "unique_crop_target", "duplicate_rows", "eval_unique_targets", "eval_forecasts"))
    support = support.loc[support.crop.isin(CROPS)].copy()
    fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True, squeeze=False)
    for i, c in enumerate(CROPS):
        q = support.loc[support.crop == c].sort_values("year")
        if q.empty:
            raise ValueError(f"annual training/evaluation support missing for {c}")
        x = np.arange(len(q)); width = .26
        left, right = axes[i]
        left.bar(x - width, q.train_rows, width, label="학습행 합계")
        left.bar(x, q.unique_crop_target, width, label="적합 원점별 고유 목표순 합")
        left.bar(x + width, q.duplicate_rows, width, label="중복 학습행")
        left.set_xticks(x, q.year.astype(int).astype(str), rotation=40)
        left.set_ylabel("적합 원점 누적 행/목표순")
        left.set_title(f"{_crop(c)} · 학습 지원 (반복 적합 포함)")
        left.legend(fontsize=7)
        right.bar(x - .18, q.eval_forecasts, .36, label="평가 예측 발행")
        right.bar(x + .18, q.eval_unique_targets, .36, label="고유 품목×목표순")
        right.set_xticks(x, q.year.astype(int).astype(str), rotation=40)
        right.set_ylabel("평가 예측/고유 목표순")
        right.set_title(f"{_crop(c)} · 평가 지원 (반복 발행 포함)")
        right.legend(fontsize=7)
    p = out / "cross_crop_data_support.png"; _save(fig, p); plt.close(fig); made[p.name] = "supported"

    return made
