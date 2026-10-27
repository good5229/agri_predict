import datetime as dt

from agri_data import seoan_pepper_target_mapping as target


def test_same_date_ratio_keeps_questionable_prices_but_blocks_daily_ratio():
    seoan_rows = [
        {"date": "2024-08-01", "product": "화건", "source_row": 2, "price_kg": "1000", "issues": []},
        {"date": "2024-08-01", "product": "화건", "source_row": 3, "price_kg": "-5", "issues": ["negative_value"]},
    ]
    national_rows = [{"date": "2024-08-01", "productCode": "1207", "productName": "건고추",
                      "quantityRaw": "2", "amountRaw": "1000"}]

    report = target.analyze_rows(seoan_rows, national_rows)
    day = report["sameDate"]["byDateAndSeoanName"]["2024-08-01"]["화건"]

    assert day["priceKgDistribution"]["n"] == 1
    assert day["invalidPriceCount"] == 1
    assert day["pairedRawPrices"] == ["1000", "-5"]
    assert day["ratioMin"] is None
    assert day["questionableRowsBlockDailyRatio"] is True


def test_archive_coverage_check_is_independent_of_seoan_overlap():
    report = target.analyze_rows([], [], national_coverage=(dt.date(2024, 7, 8), dt.date(2026, 6, 30)))
    assert report["national"]["periodOverlapsSeoan"] is True
    assert report["national"]["periodCovers2026August"] is False


def test_wrong_national_product_name_blocks_daily_price():
    rows = [{"date": "2024-08-01", "productCode": "1207", "productName": "unexpected",
             "quantityRaw": "2", "amountRaw": "1000"}]
    report = target.analyze_rows([], rows)
    daily = report["national"]["daily"]["2024-08-01"]
    assert daily["productNameMismatchCount"] == 1
    assert daily["validPrice"] is False


def test_report_identifies_manifest_not_raw_capture():
    report = target.build_report()
    manifest = report["sourceManifests"]["seoanAttemptManifest"]
    assert manifest.endswith(".manifest.json")
    assert manifest == str(target.SEOAN_MANIFEST.relative_to(target.ROOT))
