import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

mapper = module("mapper", "45_map_articles_to_companies.py")
selector = module("selector", "54_prepare_single_industry.py")

class SingleIndustryTests(unittest.TestCase):
    def setUp(self):
        mapper._WORKER_STEMMER = SnowballStemmer("russian")
        mapper._stem_token.cache_clear()

    def test_semantic_not_alphabetical_selection(self):
        for name, old, expected in [
            ("Сбербанк России", "банк; кредит; финансы", "банк"),
            ("Полюс", "горнодобыча; золото; золотодобыча; металлургия", "золотодобыча"),
            ("Открытие Холдинг", "банк; инвестиции; финансы", "финансовый холдинг"),
            ("Магнит", "продукт; розница; торговля", "розница"),
            ("Кировский завод", "машиностроение; сельское хозяйство; энергетика", "машиностроение"),
            ("Неизвестный", "", ""),
            ("СПВ", "облигация; финансирование; финансы", ""),
        ]:
            self.assertEqual(selector.select(dict(company=name,industry=old))[0],expected)

    def test_one_industry_name_and_product_patterns_unchanged(self):
        old = [dict(company_id=f"c{i}",name=f"Компания{i}",industry="металлургия; горнодобыча",product="алюминий; прокат") for i in range(25)]
        old.append(dict(company_id="bank",name="Открытие Холдинг",industry="банк; инвестиции; финансы",product="кредит; вклад"))
        new = [dict(c,industry="металлургия" if c["company_id"]!="bank" else "финансовый холдинг",original_industry=c["industry"],primary_industry_explicit=True,bank_product_eligible=mapper.is_bank_company(c)) for c in old]
        before,after = mapper.build_pattern_records(old,15),mapper.build_pattern_records(new,15)
        def refs(records,signal):
            return {(tuple(p["tokens"]),tuple(ref)) for p in records for ref in p["refs"] if ref[2]==signal}
        for signal in ("name","product"):
            self.assertEqual(refs(before,signal),refs(after,signal))
        industries = [(r[0],r[3]) for p in after for r in p["refs"] if r[2]=="industry"]
        self.assertEqual(len(industries),26)
        self.assertFalse(any(term=="банк" for cid,term in industries if cid=="bank"))

    def test_industry_title_only_and_inflections(self):
        row=dict(company_id="c",name="Компания",industry="розница",original_industry="магазин; розница; торговля",primary_industry_explicit=True,product="")
        patterns=mapper.build_pattern_records([row],15)
        mapper._WORKER_TRIE,mapper._WORKER_FAIL,mapper._WORKER_OUTPUTS,mapper._WORKER_REFS=mapper.build_automaton(patterns)
        def hits(title):
            surfaces=mapper.tokenize("Рост розницы в России")
            result={}
            mapper.add_pattern_hits(mapper.scan_tokens(mapper.stem_tokens(surfaces)),result,title,surfaces)
            return result
        self.assertEqual(hits(True)["c"]["terms"]["industry"],{"розница"})
        self.assertFalse(hits(False))

if __name__ == "__main__":
    unittest.main()
