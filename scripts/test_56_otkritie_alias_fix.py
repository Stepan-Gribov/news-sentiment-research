"""Bare ordinary words must never identify Otkritie; qualified names still work."""
import importlib.util
import unittest
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer

spec=importlib.util.spec_from_file_location("mapper",Path(__file__).with_name("45_map_articles_to_companies.py"))
mapper=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)

class OtkritieAliasTests(unittest.TestCase):
    def setUp(self):
        mapper._WORKER_STEMMER=SnowballStemmer("russian")
        mapper._stem_token.cache_clear()
        companies=[dict(company_id="holding",name="Открытие Холдинг",industry="финансовый холдинг",product=""),dict(company_id="legacy",name="Открытие Холдинг (бывш. ФК Открытие)",industry="",product=""),dict(company_id="bank",name="Банк ОТКРЫТИЕ",industry="банк",product="кредит; вклад")]
        self.patterns=mapper.build_pattern_records(companies,15)
        mapper._WORKER_TRIE,mapper._WORKER_FAIL,mapper._WORKER_OUTPUTS,mapper._WORKER_REFS=mapper.build_automaton(self.patterns)

    def hits(self,text,title=True):
        tokens=mapper.tokenize(text)
        result={}
        mapper.add_pattern_hits(mapper.scan_tokens(mapper.stem_tokens(tokens)),result,title,tokens)
        return result

    def test_generic_opening_and_adjectives_not_names(self):
        for phrase in ["Открытие салона красоты","Открытие школы","Концерт под открытым небом","Открытый чемпионат","Открытые двери музея","Он открыто заявил о решении"]:
            for title in (True,False):
                with self.subTest(phrase=phrase,title=title):
                    self.assertFalse(any("name" in hit["signals"] for hit in self.hits(phrase,title).values()))
        self.assertFalse(any(ref[2]=="name" and tuple(ref[6])==("открытие",) for p in self.patterns for ref in p["refs"]))

    def test_qualified_issuer_names_survive(self):
        for text,cid in [("Открытие Холдинг выпустил облигации","holding"),("ФК Открытие привлекла капитал","legacy"),("Банк Открытие опубликовал отчет","bank")]:
            for title in (True,False):
                self.assertIn("name",self.hits(text,title)[cid]["signals"])

    def test_banking_topics_not_blacklisted(self):
        hit=self.hits("Банки снизили ставки по кредитам и вкладам")["bank"]
        self.assertEqual(hit["terms"]["industry"],{"банк"})
        self.assertEqual(hit["terms"]["product"],{"кредит","вклад"})

if __name__=="__main__":
    unittest.main()
