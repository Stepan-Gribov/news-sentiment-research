"""Render saved PC regression statistics only; no observations or fits on Mac."""
from pathlib import Path
import hashlib
import json
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'outputs/h1_panel_report_20261030'
source = ROOT / 'h1_panel_results.json'
data = json.loads(source.read_text())
assert data['status'] == 'COMPLETED_H1_ONLY_THREE_COLUMN_REPORT'
assert data['estimation_host'] == 'research-host'
fonts = Path('/System/Library/Fonts/Supplemental')


def font(size, style=''):
    name = 'Times New Roman' + (' ' + style if style else '') + '.ttf'
    return ImageFont.truetype(str(fonts / name), size)


image = Image.new('RGB', (1800, 1900), 'white')
draw = ImageDraw.Draw(image)
black, grey = '#141414', '#555555'


def text(x, y, value, size=32, style='', center=False, fill=black):
    f = font(size, style)
    if center:
        bounds = draw.textbbox((0, 0), value, font=f)
        x -= (bounds[2] - bounds[0]) / 2
    draw.text((x, y), value, font=f, fill=fill)


text(900, 65, 'H1. Panel regression', 50, 'Bold', True)
text(900, 139, 'Price-adjusted position growth', 37, center=True)
text(900, 206, 'Dependent variable: 100 x [ln(V1/V0) - ln(P1/P0)]', 30, center=True)
text(900, 261, 'June 2018 - May 2026 | Same observations in every model', 28, center=True, fill=grey)
draw.line((100, 320, 1700, 320), fill=black, width=3)
xs = [930, 1250, 1570]
for x, number, label in zip(xs, ['(1)', '(2)', '(3)'], ['Positive only', 'Negative only', 'Joint: primary']):
    text(x, 336, number, 34, 'Bold' if number == '(3)' else '', True)
    text(x, 387, label, 29, 'Bold' if number == '(3)' else '', True)
draw.line((100, 440, 1700, 440), fill=grey, width=2)
y = 465
for row in data['comparison_rows'][:4]:
    text(100, y + 5, row['label'], 34)
    for index, (x, value) in enumerate(zip(xs, row['values'])):
        if value is None:
            text(x, y + 5, '--', 34, center=True)
        else:
            text(x, y, value[0], 35, 'Bold' if index == 2 else '', True)
            text(x, y + 50, value[1], 30, center=True, fill=grey)
    y += 135
draw.line((100, y + 1, 1700, y + 1), fill=grey, width=2)
y += 22
for row in data['comparison_rows'][4:]:
    text(100, y, row['label'], 31)
    for x, value in zip(xs, row['values']):
        text(x, y, value[0], 30, center=True)
    y += 57
draw.line((100, y + 4, 1700, y + 4), fill=black, width=3)
y += 27
notes = [
    'Standard errors in parentheses. * p < 0.10; ** p < 0.05; *** p < 0.01.',
    'CGM errors clustered by fund and issuer; full absorbed-rank correction.',
    'NAV, HHI and duration are absorbed by fund x interval fixed effects.',
    'Columns (1)-(2) are auxiliary; (3) is primary. Stars use raw two-sided p.',
    'The outcome is a continuing-position growth proxy, not whole-portfolio AT.',
]
for note in notes:
    text(100, y, note, 27)
    y += 36
assert y < image.height - 20, y
image.save(ROOT / 'h1_panel_table.png')
manifest = dict(status='FORMATTED_SAVED_PC_ESTIMATES', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                numeric_values_reestimated=False, image='h1_panel_table.png', image_size=list(image.size))
(ROOT / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(ROOT / 'h1_panel_table.png')
