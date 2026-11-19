"""Format completed PC statistics; no observations or regressions are processed."""
import hashlib
import html
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1] / 'outputs/h1_entries_trim_20261102'
qa = json.loads((root / 'qa_independent.json').read_text(encoding='utf-8'))
assert qa['fail_count'] == 0
assert qa['status'] in ['PASS', 'PASS_WITH_WARNINGS']
text = (root / 'report.html').read_text(encoding='utf-8')
banner = (
    '<p><strong>Независимая проверка: '
    f'{qa["pass_count"]} PASS, {qa["fail_count"]} FAIL.</strong> '
    'Основная модель и сравнимая модель после P99 отдельно воспроизведены '
    'через fund-within OLS и двухкластерную ковариацию без PanelOLS. '
    'В основной очищенной модели все стандартные ошибки определены. '
    'Предупреждения относятся к отдельным контролям в неочищенных моделях. '
    '<a href="qa_independent.json">Численная проверка</a> · '
    '<a href="result_review.json">Интерпретация</a>.</p>'
    '<p>R² относится ко всему уравнению с контролями. Входы +100% — условный код, '
    'поэтому результат описывает связь с кодированным изменением количества, '
    'а не буквальную процентную доходность или изменение доли портфеля.</p>'
    '<p><a href="conditional_entries_100.csv.gz">Все 10 329 условных входов</a> · '
    '<a href="excluded_upper_p99.csv.gz">Все 2 042 исключённых прироста</a>.</p>'
)
text = text.replace('</h1>', '</h1>' + banner, 1)
text = text.replace(
    '<a href="row_decisions.csv.gz">Все строки и причины</a>',
    'Все строки и причины: <code>row_decisions.csv.gz</code> на research-host '
    'в <code>./analysis_outputs/h1_entries_trim_20261102</code>'
)
(root / 'review.html').write_text(text, encoding='utf-8')
links = re.findall(r'href="([^"]+)"', text)
missing = [link for link in links if not link.startswith(('#', 'http')) and not (root / link).exists()]
assert not missing, missing
print(json.dumps({'review_bytes': (root / 'review.html').stat().st_size,
                  'broken_local_links': missing,
                  'numeric_summary_sha256': hashlib.sha256((root / 'summary.json').read_bytes()).hexdigest()}))
