"""実行後のノートブックの出力を確認する(開発用)。

    py _build/inspect_nb.py            # エラーと各コードセルの出力の要約
    py _build/inspect_nb.py 57         # セル57の出力をすべて表示
"""
import glob, json, os, sys

NB = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '*.ipynb'))[0]
nb = json.load(open(NB, encoding='utf-8'))
cells = nb['cells']

def text_of(o):
    if o['output_type'] == 'stream':
        return ''.join(o['text'])
    if o['output_type'] == 'error':
        return f"ERROR {o['ename']}: {o['evalue']}\n" + '\n'.join(o['traceback'][-3:])
    data = o.get('data', {})
    kinds = ','.join(data)
    if 'text/html' in data and 'video' in ''.join(data['text/html'])[:300]:
        return f'[video {len("".join(data["text/html"])) // 1024} KB]'
    return f'[{kinds}]'

if len(sys.argv) > 1:
    c = cells[int(sys.argv[1])]
    print(''.join(c['source'])[:3000])
    print('-' * 40)
    for o in c.get('outputs', []):
        print(text_of(o))
    sys.exit()

size = os.path.getsize(NB) // 1024
print(f'{NB}: {len(cells)} cells, {size} KB')
for i, c in enumerate(cells):
    if c['cell_type'] != 'code':
        continue
    first = ''.join(c['source']).split('\n')[0][:60]
    outs = [text_of(o) for o in c.get('outputs', [])]
    summary = ' | '.join(s.replace('\n', ' / ')[:160] for s in outs)
    flag = 'ERR ' if any(o['output_type'] == 'error' for o in c.get('outputs', [])) else '    '
    print(f'{flag}{i:3d} {first:<60} {summary}')
