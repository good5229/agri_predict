import json
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from agri_data import regional_gap, regional_gap_audit as audit


def payload(codes,names=("깐마늘","특","대","망")):
    keys=("gds_sclsf_nm","grd_nm","sz_nm","pkg_nm")
    p=dict(zip(keys,names))
    p.update(dict(zip(("gds_sclsf_cd","grd_cd","sz_cd","pkg_cd"),codes)))
    return p


def add(cells,market,day,codes,amount=1000,weight=10,names=("깐마늘","특","대","망")):
    p=payload(codes,names);key=("garlic",day,market,codes[0],tuple(codes))
    item=cells.setdefault(key,{"amount":Decimal(0),"weight":Decimal(0),"rows":0,"names":[]})
    item["amount"]+=Decimal(amount);item["weight"]+=Decimal(weight);item["rows"]+=1
    item["names"]=regional_gap._merge_names(item["names"],regional_gap._names_for(p))


class RegionalGapAuditTests(unittest.TestCase):
    def test_progressive_code_matching_and_name_conflict_attrition(self):
        d="2018-04-02";cells={}
        common=("06","1","10","2");grade=("06","2","10","2");size=("06","1","20","2");pkg=("06","1","20","3")
        for market in regional_gap.MARKETS:
            add(cells,market,d,common)
        add(cells,regional_gap.GWANGJU,d,grade)
        add(cells,regional_gap.GARAK,d,grade)
        add(cells,regional_gap.GWANGJU,d,size)
        add(cells,regional_gap.GARAK,d,pkg)
        report=audit.audit_cells(cells)
        product=next(x for x in report["products"] if x["productId"]=="garlic")
        self.assertEqual(product["commonDates"],1)
        self.assertEqual([x["matchedKeyCount"] for x in product["stages"]],[1,2,3,2])
        self.assertEqual(product["matchedCells"],2)
        self.assertEqual(product["matchedSharedNetWeightKg"],20.0)
        self.assertIn("not a count of cells lost",report["auditDefinition"]["stageCountMeaning"])
        self.assertIn("complete four-code cells",report["auditDefinition"]["auditedPopulation"])
        self.assertIn("at least one Seoul market",report["auditDefinition"]["commonDateDefinition"])
        self.assertNotIn("gwangjuPrice",json.dumps(report))
        self.assertEqual(len(product["codeCombinationsByMarket"][regional_gap.GWANGJU]),3)

    def test_same_codes_with_conflicting_names_are_counted_but_not_matched(self):
        d="2019-07-12";codes=("06","1","10","2");cells={}
        for m in regional_gap.MARKETS:
            names=("깐마늘","특","대" if m==regional_gap.GWANGJU else "중","망")
            add(cells,m,d,codes,names=names)
        report=audit.audit_cells(cells)
        p=next(x for x in report["products"] if x["productId"]=="garlic")
        self.assertEqual(p["stages"][-1]["matchedKeyCount"],1)
        self.assertEqual(p["nameCompatibility"]["dimension_name_conflict"]["cells"],1)
        self.assertEqual(p["matchedCells"],0)

    def test_common_dates_are_blocked_by_incomplete_market_query(self):
        d="2018-04-02";codes=("06","1","10","2");cells={}
        for m in regional_gap.MARKETS:add(cells,m,d,codes)
        scope={f"garlic|{d}|{m}":["complete"] for m in regional_gap.MARKETS}
        scope[f"garlic|{d}|{regional_gap.GARAK}"]=["source_incomplete:failed"]
        p=next(x for x in audit.audit_cells(cells,{"scopeStates":scope})["products"] if x["productId"]=="garlic")
        self.assertEqual(p["commonDates"],0)
        self.assertEqual(p["matchedCells"],0)

    def test_request_scope_fixture_keeps_only_ids_hashes_and_states(self):
        db=sqlite3.connect(":memory:");db.row_factory=sqlite3.Row
        db.executescript("""
          CREATE TABLE scope(latest_job_id,selected_job_id,selected_epoch_id,source_url,task_key,query_json,selected_state,latest_state,selected_previous,dataset);
          CREATE TABLE source_requests(request_id,job_id,dataset,source_url,task_key,epoch_id,state,retrieved_at,published_at,version_published_at,response_sha256);
        """)
        query={"cond[trd_clcln_ymd::EQ]":"2018-04-02","cond[whsl_mrkt_cd::EQ]":regional_gap.GWANGJU,"cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09","cond[gds_sclsf_cd::EQ]":"06"}
        db.execute("INSERT INTO scope VALUES(1,1,'e1','url','task',?,'complete','complete',0,'prices_origin')",(json.dumps(query),))
        db.execute("INSERT INTO source_requests VALUES(1,1,'prices_origin','url','task','e1','complete','2026-01-01',NULL,NULL,'sha')")
        db.commit()
        cte="""WITH picked AS (SELECT latest_job_id,selected_job_id,selected_epoch_id,source_url,task_key,query_json,selected_state,latest_state,selected_previous,dataset FROM scope),
        request_pages AS (SELECT request_id,job_id,dataset,source_url,task_key,epoch_id,state,retrieved_at,published_at,version_published_at FROM source_requests)"""
        with patch.object(audit.market_extension,"_scoped_selection_cte",return_value=cte):
            scopes=audit._request_scopes(db)
        items=scopes["garlic|2018-04-02|240004"]
        self.assertEqual(items[0]["requestId"],1)
        self.assertEqual(items[0]["responseSha256"],"sha")
        self.assertNotIn("query",items[0])
        report=audit.audit_cells({},provenance={"requests":{"1":{"requestId":1}},"selectedScopes":[{"task":"opaque"}]})
        self.assertNotIn("selectedScopes",report["provenance"])

    def test_cli_emits_json_and_markdown_without_overwriting_db(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);db=root/"db.sqlite";db.touch();out=root/"audit.json"
            report=audit.audit_cells({})
            with patch.object(audit,"run_audit",return_value=report):
                self.assertEqual(audit.main(["--db",str(db),"--output",str(out)]),0)
            self.assertEqual(json.loads(out.read_text())["status"],"complete")
            self.assertTrue((root/"report.md").exists())
            with self.assertRaisesRegex(ValueError,"must not overwrite"):
                audit.main(["--db",str(db),"--output",str(db)])


if __name__=="__main__":unittest.main()
