"""ノートブック生成スクリプト。

    py _build/nbbuild.py          # 章ファイルからノートブックを組み立てる(出力なし)
    py _build/nbbuild.py --run    # 組み立てたうえで全セルを実行し，出力を保存する

第0章の 0.1 / 0.2 のセル(既存)は元のノートブックから引き継ぎ，
それ以降のセルは _build/ch*.py の md() / code() から生成する。
"""
import glob, importlib, json, os, sys
from textwrap import dedent

HERE = os.path.dirname(os.path.abspath(__file__))
NB = glob.glob(os.path.join(HERE, '..', '*.ipynb'))[0]
CHAPTERS = ['ch00', 'ch01', 'ch02', 'ch03', 'ch04', 'ch05', 'ch06',
            'ch07', 'ch08', 'ch09', 'ch10', 'ch11', 'appendix']
KEEP = 3  # 元のノートブックから引き継ぐ先頭セル数(0, 0.1, 0.2)

cells = []


def _lines(s):
    parts = s.split('\n')
    return [p + '\n' for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def md(s):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': _lines(dedent(s).strip('\n'))})


def code(s):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': _lines(dedent(s).strip('\n'))})


def build():
    nb = json.load(open(NB, encoding='utf-8'))
    head = nb['cells'][:KEEP]
    sys.path.insert(0, HERE)
    import nbbuild  # 章ファイルは nbbuild.md / nbbuild.code に追記する(__main__ とは別モジュール)
    for name in CHAPTERS:
        if os.path.exists(os.path.join(HERE, name + '.py')):
            importlib.import_module(name)
    nb['cells'] = head + nbbuild.cells
    return nb


def save(nb):
    with open(NB, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(nb, f, ensure_ascii=False, indent=2)
        f.write('\n')


def _collapse_streams(outputs):
    """連続するstream出力をまとめ，'\\r' で上書きされる途中経過を消す(画面の見た目と同じにする)"""
    merged = []
    for o in outputs:
        if o['output_type'] == 'stream' and merged and merged[-1]['output_type'] == 'stream' and merged[-1]['name'] == o['name']:
            merged[-1]['text'] += ''.join(o['text'])
        else:
            o = dict(o)
            if o['output_type'] == 'stream':
                o['text'] = ''.join(o['text'])
            merged.append(o)
    for o in merged:
        if o['output_type'] == 'stream':
            lines = o['text'].split('\n')
            o['text'] = _lines('\n'.join(l.rstrip('\r').split('\r')[-1] for l in lines))
    return merged


def run(nb):
    import nbformat
    from nbclient import NotebookClient
    node = nbformat.reads(json.dumps(nb), as_version=4)
    client = NotebookClient(node, timeout=3600, kernel_name=os.environ.get('NB_KERNEL', 'python3'),
                            resources={'metadata': {'path': os.path.dirname(NB)}}, allow_errors=True)
    client.execute()
    out = json.loads(nbformat.writes(node))
    for c in out['cells']:
        c.get('metadata', {}).pop('execution', None)
        if c['cell_type'] == 'code':
            c['outputs'] = _collapse_streams(c['outputs'])
    out['nbformat'], out['nbformat_minor'] = nb['nbformat'], nb['nbformat_minor']
    return strip_widgets(out)


def strip_widgets(nb):
    """ウィジェット(スライダー)の保存状態を消す。Colabでは使われず，ファイルが大きくなるだけなので"""
    nb.get('metadata', {}).pop('widgets', None)
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = [o for o in c['outputs'] if 'application/vnd.jupyter.widget-view+json' not in o.get('data', {})]
    return nb


if __name__ == '__main__':
    nb = build()
    if '--run' in sys.argv:
        nb = run(nb)
    save(nb)
    n_err = sum(1 for c in nb['cells'] for o in c.get('outputs', []) if o.get('output_type') == 'error')
    print(f'cells: {len(nb["cells"])}, errors: {n_err}')
