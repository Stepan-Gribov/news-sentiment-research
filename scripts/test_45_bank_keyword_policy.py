"""Bank topic exceptions: issuer scope, signal roles, owner cap, inflections."""
import importlib.util
import unittest
from pathlib import Path
from nltk.stem.snowball import SnowballStemmer

spec = importlib.util.spec_from_file_location("mapper45", Path(__file__).with_name("45_map_articles_to_companies.py"))
mapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mapper)


class BankPolicyTests(unittest.TestCase):
    def setUp(self):
        mapper._WORKER_STEMMER = SnowballStemmer("russian")
        mapper._stem_token.cache_clear()

    def bank(self, i=0):
        return dict(company_id=f"b{i}", name=f"Банк Альфа{i}", industry="банк; кредит; финансы", product="вклад; карта; кредит; платёж")

    def test_scope_and_roles(self):
        b = self.bank()
        self.assertTrue(mapper.bank_keyword_exception(b, "industry", "банк"))
        self.assertTrue(mapper.bank_keyword_exception(b, "product", "кредит"))
        self.assertTrue(mapper.bank_keyword_exception(b, "product", "вклад"))
        self.assertFalse(mapper.bank_keyword_exception(b, "industry", "кредит"))
        self.assertFalse(mapper.bank_keyword_exception(b, "product", "карта"))
        self.assertFalse(mapper.is_bank_company(dict(name="Совкомбанк Лизинг", industry="лизинг; финансы")))
        self.assertFalse(mapper.is_bank_company(dict(name="ГАЗКОН", industry="финансы")))

    def test_many_banks_not_dropped_by_owner_cap(self):
        records = mapper.build_pattern_records([self.bank(i) for i in range(40)], 15)
        refs = [r for p in records for r in p["refs"]]
        for signal, keyword in [("industry", "банк"), ("product", "кредит"), ("product", "вклад")]:
            self.assertEqual(len({r[0] for r in refs if r[2] == signal and r[3] == keyword}), 40)
        self.assertFalse(any(r[2] == "industry" and r[3] in {"кредит", "финансы"} for r in refs))
        self.assertFalse(any(r[2] == "product" and r[3] in {"карта", "платёж"} for r in refs))

    def test_inflections_title_only_no_substrings(self):
        records = mapper.build_pattern_records([self.bank()], 15)
        mapper._WORKER_TRIE, mapper._WORKER_FAIL, mapper._WORKER_OUTPUTS, mapper._WORKER_REFS = mapper.build_automaton(records)
        def hits(text, title=True):
            surface = mapper.tokenize(text)
            result = {}
            mapper.add_pattern_hits(mapper.scan_tokens(mapper.stem_tokens(surface)), result, title, surface)
            return result
        r = hits("Банки снизили ставки по кредитам и вкладам")["b0"]
        self.assertEqual(r["terms"]["industry"], {"банк"})
        self.assertEqual(r["terms"]["product"], {"кредит", "вклад"})
        self.assertFalse(hits("банкротство и вкладка браузера"))
        self.assertFalse(hits("Банки снизили ставки по кредитам", title=False))


if __name__ == "__main__":
    unittest.main()
