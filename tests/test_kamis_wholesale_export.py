import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest

from agri_data.kamis_wholesale_export import parse_export, validate_manifest


def _write(tmp_path, body):
    p = tmp_path / "export.xls"
    p.write_text(body, encoding="utf-8")
    return p


def _html(rows, headers="<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th><th>대전</th>", caption="중도매인 판매가격 채소류, 배추, 전체, 상품, 10kg(그물망 3포기) 기간별/일간 가격정보"):
    trs = "".join(f"<tr><th><span>20</span>{d}</th><td>{price}</td><td>-</td>" + "".join("<td>100</td>" for _ in range(headers.count("<th>") - 3)) + "</tr>" for d, price in rows)
    return f"<table><tr><td><table id='data'><caption>{caption}</caption><thead><tr>{headers}</tr></thead><tbody>{trs}</tbody></table></td></tr></table>"


def _parse(path, **overrides):
    args = dict(start=dt.date(2025, 9, 1), end=dt.date(2025, 9, 3),
                item="배추", variety="전체", grade="상품", unit="10kg(그물망 3포기)")
    args.update(overrides)
    return parse_export(path, **args)


def test_preserves_raw_strings_negative_decimal_missing_and_date_support(tmp_path):
    p = _write(tmp_path, _html([("25-09-03", "-1,234.50"), ("25-09-01", "-")]))
    data = _parse(p)
    assert data["rawRows"][0]["rawCells"] == ["2025-09-03", "-1,234.50", "-", "100", "100", "100", "100", "100"]
    assert data["rawRows"][0]["priceDecimal"] == "-1234.50"
    assert data["rawRows"][0]["flags"] == ["negative_price"]
    assert data["rawRows"][0]["prices"]["평균"]["flags"] == ["negative_price"]
    assert data["rawRows"][1]["priceStatus"] == "missing"
    assert data["support"]["missingDates"] == ["2025-09-02"]
    assert data["publicationTime"] is None and data["unitConversion"] is None


@pytest.mark.parametrize("html, message", [
    (_html([("25-09-01", "1"), ("25-09-01", "2")]), "duplicate date"),
    (_html([("25-09-04", "1")]), "outside requested range"),
    (_html([("25-09-01", "1")], headers="<th>구분</th><th>평균</th><th>평균</th>"), "duplicate header"),
    (_html([("25-09-01", "1")], caption="중도매인 판매가격 채소류, 배추, 전체, 상품, 다른단위 기간별/일간 가격정보"), "caption unit mismatch"),
    (_html([("25-09-01", "1")]).replace("<td>-</td>", ""), "cells; expected"),
    (_html([("25-09-01", "1")], caption="중도매인 판매가격 채소류, 열무, 전체, 상품, 10kg(그물망 3포기) 기간별/일간 가격정보"), "item mismatch"),
    (_html([("25-09-01", "1")], caption="중도매인 판매가격 채소류, 배추, 전체, 최상품, 10kg(그물망 3포기) 기간별/일간 가격정보"), "grade mismatch"),
    (_html([("25-09-01", "1")], caption="중도매인 판매가격 채소류, 배추, 전체, 상품, 11kg(그물망 3포기) 기간별/일간 가격정보"), "unit mismatch"),
    (_html([("25-09-01", "1")], headers="<th>구분</th><th>평균</th><th>서울</th><th>서울</th>"), "duplicate header"),
    (_html([("25-09-01", "1")], headers="<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th>"), "schema mismatch"),
])
def test_rejects_invalid_exports(tmp_path, html, message):
    with pytest.raises(ValueError, match=message):
        _parse(_write(tmp_path, html))


def test_rejects_duplicate_or_truncated_data_table(tmp_path):
    html = _html([("25-09-01", "1")])
    with pytest.raises(ValueError, match="exactly one complete"):
        _parse(_write(tmp_path, html + html))
    with pytest.raises(ValueError, match="exactly one complete"):
        _parse(_write(tmp_path, html.replace("</table>", "")))


def test_rejects_overlapping_rows_cells_and_nested_tables(tmp_path):
    bad_row = ("<table><table id='data'><caption>" +
               "中도매인 판매가격 채소류, 배추, 전체, 상품, 10kg(그물망 3포기) 기간별/일간 가격정보" +
               "</caption><thead><tr><th>구분</th><th>평균</th><th>등락율</th></tr></thead><tbody>"
               "<tr><th>2025-09-01</th><td>10</td><tr><th>2025-09-02</th><td>20</td></tr></table></table>")
    with pytest.raises(ValueError, match="complete table"):
        _parse(_write(tmp_path, bad_row))
    nested = _html([("25-09-01", "1")]).replace("<td>1</td>", "<td><table><tr><td>1</td></tr></table></td>")
    with pytest.raises(ValueError, match="complete table"):
        _parse(_write(tmp_path, nested))


def test_validates_each_regional_price(tmp_path):
    html = _html([("25-09-01", "10")])
    html = html.replace("<td>100</td></tr>", "<td>-Infinity</td></tr>")
    row = _parse(_write(tmp_path, html))["rawRows"][0]
    assert row["prices"]["대전"]["raw"] == "-Infinity"
    assert row["prices"]["대전"]["flags"] == ["nonfinite_price"]


def test_actual_public_fixture_is_readable():
    root = Path(__file__).resolve().parents[1] / "data/raw/kamis_wholesale_exports_20261010"
    path = root / "cabbage_20250901_20250907_high.xls"
    q1_path = root / "cabbage_20240101_20240331_high.xls"
    if not path.exists() or not q1_path.exists(): pytest.skip("optional downloaded fixtures are absent")
    data = parse_export(path, start=dt.date(2025, 9, 1), end=dt.date(2025, 9, 7),
                        item="배추", variety="전체", grade="상품", unit="10kg(그물망 3포기)")
    assert data["support"]["observedDates"] == 5
    assert len(data["rawRows"]) == 5
    assert all(len(row["rawCells"]) == 8 for row in data["rawRows"])
    q1 = parse_export(q1_path,
                      start=dt.date(2024, 1, 1), end=dt.date(2024, 3, 31),
                      item="배추", variety="전체", grade="상품", unit="10kg(그물망 3포기)")
    assert q1["support"]["observedDates"] == 61
    assert q1["rawRows"][0]["date"].startswith("2024-")


def test_selects_item_table_without_changing_default_data_table(tmp_path):
    html = _html([("25-09-01", "10")]).replace("id='data'", "id='itemTable_1'")
    raw = html.encode("utf-8")
    parsed = parse_export(raw, start=dt.date(2025, 9, 1), end=dt.date(2025, 9, 1),
                          item="배추", variety="전체", grade="상품", unit="10kg(그물망 3포기)",
                          table_id="itemTable_1")
    assert parsed["support"]["observedDates"] == 1
    with pytest.raises(ValueError, match="table id=data"):
        parse_export(raw, start=dt.date(2025, 9, 1), end=dt.date(2025, 9, 1),
                     item="배추", variety="전체", grade="상품", unit="10kg(그물망 3포기)")


def test_manifest_validation_is_create_only_and_compares_week_to_quarter(tmp_path):
    headers = "<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th><th>대전</th>"
    caption = "중도매인 판매가격 채소류, 배추, 전체, 상품, 10kg(그물망 3포기) 기간별/일간 가격정보"
    rows = "<tr><th>2024-01-01</th><td>1,000</td><td>0.0</td><td>900</td><td>910</td><td>920</td><td>930</td><td>940</td></tr>"
    template = f"<table><table id='data'><caption>{caption}</caption><thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></table>"
    input_dir = tmp_path / "inputs"; input_dir.mkdir()
    entries = []
    for name, start, end in (("week.xls", "2024-01-01", "2024-01-07"), ("quarter.xls", "2024-01-01", "2024-03-31")):
        p = input_dir / name; p.write_text(template, encoding="utf-8"); raw = p.read_bytes()
        entries.append({"path": name, "item": "배추", "variety": "전체", "grade": "상품", "unit": "10kg(그물망 3포기)",
                        "start": start, "end": end, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "uiDataRows": 1})
    manifest = tmp_path / "provenance.json"; manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    output = tmp_path / "validation.json"
    report = validate_manifest(manifest, input_dir, output)
    assert report["fileCount"] == 2 and all(x["uiDataRows"] == 1 for x in report["files"])
    comparison = report["cabbageWeekQuarterComparisons"][0]
    assert comparison["priceColumns"] == ["평균", "서울", "부산", "대구", "광주", "대전"]
    assert comparison["cellComparisons"] == 6 and comparison["exactMatch"]
    assert report["qualityStatus"] == "clear"
    assert report["missingCellsAreSeparateFromPriceErrors"]
    with pytest.raises(FileExistsError): validate_manifest(manifest, input_dir, output)


def test_manifest_marks_nan_overlap_and_price_flags_for_review(tmp_path):
    headers = "<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th><th>대전</th>"
    caption = "중도매인 판매가격 채소류, 배추, 전체, 상품, 10kg(그물망 3포기) 기간별/일간 가격정보"
    input_dir = tmp_path / "inputs"; input_dir.mkdir()
    entries = []
    for name, start, end, mean, seoul in (("week.xls", "2024-01-01", "2024-01-07", "NaN", "-2"),
                                          ("quarter.xls", "2024-01-01", "2024-03-31", "sNaN", "-2")):
        vals = [mean, "0", seoul, "10", "10", "10", "10"]
        row = "<tr><th>2024-01-01</th>" + "".join(f"<td>{v}</td>" for v in vals) + "</tr>"
        html = f"<table><table id='data'><caption>{caption}</caption><thead><tr>{headers}</tr></thead><tbody>{row}</tbody></table></table>"
        p = input_dir / name; p.write_text(html, encoding="utf-8"); raw = p.read_bytes()
        entries.append({"path": name, "item": "배추", "variety": "전체", "grade": "상품", "unit": "10kg(그물망 3포기)",
                        "start": start, "end": end, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "uiDataRows": 1})
    manifest = tmp_path / "provenance.json"; manifest.write_text(json.dumps({"files": entries}), encoding="utf-8")
    report = validate_manifest(manifest, input_dir, tmp_path / "report.json")
    assert report["qualityStatus"] == "review_needed"
    assert all(f["qualityStatus"] == "review_needed" for f in report["files"])
    assert all(f["nonfinitePriceCells"] >= 1 and f["negativePriceCells"] == 1 for f in report["files"])
    comparison = report["cabbageWeekQuarterComparisons"][0]
    assert not comparison["exactMatch"] and comparison["qualityStatus"] == "review_needed"
    assert all(d["numericComparison"] == "unavailable" for d in comparison["differences"] if d["column"] == "평균")
