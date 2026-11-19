from __future__ import annotations
import csv
import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC=importlib.util.spec_from_file_location('w2',Path(__file__).resolve().parents[1]/'scripts/build_title_name_security_crosswalk.py')
w2=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(w2)
LKOH='RU0009024277';GAZP='RU0007661625'


def meta(url='https://iss.moex.com/iss/securities.json?q=EXACT'):
 return {'url':url,'observed_at_utc':'2026-10-26T10:00:00+00:00','raw_path':'fixture.json','sha256':'fixture'}


def obj(rows):
 fields=['regnumber','isin','secid','emitent_id','emitent_title','type']
 return {'securities':{'columns':fields,'data':rows}}


def code_fixture():
 return {'1-01-00077-A':{'raw_codes':{'1-01-00077-A'},'rows':3,'equity_rows':3,'dates':{'2010-01-31','2026-05-29'}}}


class CrosswalkTests(unittest.TestCase):
 def test_normalization_is_typographic_not_fuzzy(self):
  self.assertEqual(w2.normalized(' 1–01–00077-a '),'1-01-00077-A')
  self.assertNotEqual(w2.normalized('1-01-00077-А'),w2.normalized('1-01-00077-A'))
  self.assertNotEqual(w2.normalized('4B02-01-00001-A-001P'),w2.normalized('4B02-01-00001-A'))

 def test_isin_checksum(self):
  for isin in (LKOH,GAZP,'RU0009029540'):self.assertTrue(w2.valid_isin(isin))
  self.assertFalse(w2.valid_isin('RU0009024278'))
  self.assertFalse(w2.valid_isin('RU000902427'))

 def test_only_exact_registration_is_accepted(self):
  evidence={}
  rows=[['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share'],['1-01-00077-A-EXTRA',GAZP,'WRONG',1,'Lukoil','common_share']]
  w2.add_candidates(evidence,obj(rows),meta(),code_fixture())
  self.assertEqual(len(evidence),1)
  self.assertEqual(next(iter(evidence.values()))['isin'],LKOH)

 def test_multiple_isins_and_issuers_are_not_resolved_by_first_row(self):
  evidence={}
  w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share'],['1-01-00077-A',GAZP,'OTHER',123,'Other','common_share']]),meta(),code_fixture())
  rows,conflicts=w2.consolidate(evidence,{}, {},code_fixture(),[])
  self.assertEqual(rows[0]['mapping_status'],'CONFLICT')
  self.assertIsNone(rows[0]['isin'])
  self.assertIn('multiple_legal_issuer_source_ids',rows[0]['conflict_reasons'])
  self.assertEqual(len(conflicts),1)

 def test_source_dates_and_ticker_changes_never_create_validity_interval(self):
  evidence={};w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share']]),meta(),code_fixture())
  description={'description':{'columns':['name','title','value'],'data':[['REGNUMBER','reg','1-01-00077-A'],['ISIN','isin',LKOH],['ISSUEDATE','Дата начала торгов','2003-08-20'],['REGISTRY_DATE','Дата государственной регистрации','2003-06-25']]}}
  rows,conflicts=w2.consolidate(evidence,{'LKOH':(description,meta('https://iss.moex.com/iss/securities/LKOH.json'))},{LKOH:{'company1'}},code_fixture(),[{'old_secid':'OLD','new_secid':'LKOH','action_date':'2011-01-01'}])
  row=rows[0]
  self.assertEqual(row['mapping_status'],'EXACT_CURRENT_CONFIRMED')
  self.assertFalse(row['historical_baseline_eligible'])
  self.assertIsNone(row['identity_valid_from']);self.assertIsNone(row['identity_valid_to'])
  self.assertEqual(row['dated_source_facts'][0]['source_title'],'Дата начала торгов')
  self.assertEqual(row['temporal_evidence_level'],'OFFICIAL_DATED_FACTS_NO_IDENTITY_INTERVAL')
  self.assertEqual(conflicts,[])

 def test_missing_card_identity_is_absence_not_contradiction(self):
  evidence={};w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share']]),meta(),code_fixture())
  rows,_=w2.consolidate(evidence,{'LKOH':({'description':{'columns':['name','value'],'data':[['NAME','x']]}},meta())},{LKOH:{'c1'}},code_fixture(),[])
  self.assertEqual(rows[0]['mapping_status'],'EXACT_CURRENT_CONFIRMED')
  self.assertEqual(rows[0]['temporal_evidence_level'],'CURRENT_METADATA_ONLY')

 def test_conflicting_card_is_quarantined(self):
  evidence={};w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share']]),meta(),code_fixture())
  details={'LKOH':({'description':{'columns':['name','value'],'data':[['REGNUMBER','1-01-00077-A'],['ISIN',GAZP]]}},meta())}
  rows,_=w2.consolidate(evidence,details,{},code_fixture(),[])
  self.assertIn('catalog_vs_security_card_identity_mismatch',rows[0]['conflict_reasons'])
  observation=rows[0]['security_card_observations'][0]
  self.assertEqual(observation['comparison'],'CONFLICTING_IDENTITY')
  self.assertEqual(observation['isin_observed'],GAZP)
  self.assertEqual(observation['source']['sha256'],'fixture')

 def test_matching_security_with_conflicting_card_issuer_is_quarantined(self):
  evidence={};w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share']]),meta(),code_fixture())
  card={'description':{'columns':['name','value'],'data':[['REGNUMBER','1-01-00077-A'],['ISIN',LKOH],['EMITTER_ID','999']]}}
  rows,_=w2.consolidate(evidence,{'LKOH':(card,meta())},{LKOH:{'company1'}},code_fixture(),[])
  self.assertEqual(rows[0]['mapping_status'],'CONFLICT')
  self.assertIn('catalog_vs_security_card_issuer_mismatch',rows[0]['conflict_reasons'])
  self.assertEqual(rows[0]['issuer_source_ids'],['770','999'])
  self.assertEqual(rows[0]['security_card_observations'][0]['comparison'],'CONFLICTING_ISSUER')

 def test_card_can_supply_missing_catalog_issuer_after_exact_identity_match(self):
  evidence={};w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',None,None,'common_share']]),meta(),code_fixture())
  card={'description':{'columns':['name','value'],'data':[['REGNUMBER','1-01-00077-A'],['ISIN',LKOH],['EMITTER_ID','770']]}}
  rows,_=w2.consolidate(evidence,{'LKOH':(card,meta())},{},code_fixture(),[])
  self.assertEqual(rows[0]['mapping_status'],'EXACT_CURRENT_CONFIRMED')
  self.assertEqual(rows[0]['issuer_source_ids'],['770'])

 def test_later_registration_date_creates_review_issue_without_backdating(self):
  evidence={};w2.add_candidates(evidence,obj([['1-01-00077-A',LKOH,'LKOH',770,'Lukoil','common_share']]),meta(),code_fixture())
  card={'description':{'columns':['name','title','value'],'data':[['REGNUMBER','reg','1-01-00077-A'],['ISIN','isin',LKOH],['REGISTRY_DATE','Дата регистрации','2012-02-14']]}}
  rows,_=w2.consolidate(evidence,{'LKOH':(card,meta())},{},code_fixture(),[])
  self.assertEqual(rows[0]['mapping_status'],'EXACT_CURRENT_CONFIRMED')
  self.assertEqual(rows[0]['temporal_review_issues'][0]['first_observed_holding'],'2010-01-31')
  self.assertIsNone(rows[0]['identity_valid_from'])
  self.assertFalse(rows[0]['historical_baseline_eligible'])

 def test_cache_detects_corruption(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);url=meta()['url'];key=hashlib.sha256(url.encode()).hexdigest();raw=b'{"ok":true}'
   (root/(key+'.json')).write_bytes(raw)
   (root/(key+'.meta.json')).write_text(json.dumps({'url':url,'sha256':hashlib.sha256(raw).hexdigest()}))
   cache=w2.ISSCache(root,offline=True)
   self.assertTrue(cache.get(url)[0]['ok'])
   (root/(key+'.json')).write_bytes(b'{"ok":false}')
   with self.assertRaises(RuntimeError):cache.get(url)

 def test_coverage_counts_leaf_once_and_excludes_ambiguous_company(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'funds/mac-20261008';folder.mkdir(parents=True)
   base={'table_type':'asset','row_type':'asset_item','registration_number_security':'1-01-00077-A','parent_category':'Акции','report_date':'2020-01-31','fund_id':'f1'}
   rows=[base,dict(base,row_type='asset_category'),dict(base,registration_number_security=GAZP)]
   with gzip.open(folder/'fund_holdings_long.csv.gz','wt',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=list(base));writer.writeheader();writer.writerows(rows)
   x=[{'registration_code_normalized':'1-01-00077-A','mapping_status':'EXACT_CURRENT_CONFIRMED','isin':LKOH}]
   result=w2.coverage(root,{},x,{LKOH:{'c1'},GAZP:{'c2','c3'}},{'c1'})
   self.assertEqual(result['after']['counts']['asset_item_rows'],2)
   self.assertEqual(result['after']['counts']['unique_static_company_rows'],1)
   self.assertEqual(result['after']['counts']['ambiguous_company_rows'],1)
   self.assertEqual(result['after']['unique_companies'],1)

 def test_annual_entity_counts_deduplicate_and_do_not_bleed_between_years(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);folder=root/'funds/mac-20261008';folder.mkdir(parents=True)
   base={'table_type':'asset','row_type':'asset_item','registration_number_security':LKOH,'parent_category':'Акции','report_date':'2020-01-31','fund_id':'f1'}
   rows=[base,dict(base,report_date='2020-02-29'),dict(base,registration_number_security=GAZP,report_date='2021-01-31',fund_id='f2')]
   with gzip.open(folder/'fund_holdings_long.csv.gz','wt',encoding='utf-8',newline='') as h:
    writer=csv.DictWriter(h,fieldnames=list(base));writer.writeheader();writer.writerows(rows)
   result=w2.coverage(root,{},[],{LKOH:{'c1'},GAZP:{'c2'}},{'c1'})['after']
   self.assertEqual(result['unique_companies'],2)
   self.assertEqual(result['unique_relevant_companies'],1)
   self.assertEqual(result['by_year']['2020']['single_isin_rows'],2)
   self.assertEqual(result['by_year']['2020']['unique_securities'],1)
   self.assertEqual(result['by_year']['2020']['funds_with_unique_company'],1)
   self.assertEqual(result['by_year']['2021']['unique_companies'],1)
   self.assertEqual(result['by_year']['2021']['unique_relevant_companies'],0)


if __name__=='__main__':unittest.main()
