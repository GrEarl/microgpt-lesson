"""図とアニメーションの見た目を点検するための開発用スクリプト。

ノートブックのコードセルをこのプロセスの中で順番に実行し，
- すべての【スライダー】を「既定値」「各スライダーの最小値・最大値」「ドロップダウンの別の値」で描画して PNG に保存
- 静的なグラフも PNG に保存
- 【アニメーション】を描画して，コマ送りの一覧画像(コンタクトシート)を保存
する。学習は短く(30ステップ)して速く回す(見た目の確認用なので)。

    <venv>/python _build/preview.py [OUTDIR=_preview] [--anim ch03,ch06 | --anim all] [--sheets]
"""
import glob, importlib, math, os, re, subprocess, sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nbbuild

rest = [a for a in sys.argv[1:] if not a.startswith('--')]
OUT = os.path.abspath(rest[0]) if rest else os.path.join(HERE, '..', '_preview')
os.makedirs(OUT, exist_ok=True)
ANIM = []
if '--anim' in sys.argv:
    ANIM = sys.argv[sys.argv.index('--anim') + 1].split(',')

# 章ごとのセル範囲を記録しながら組み立てる
spans = []
for name in nbbuild.CHAPTERS:
    start = len(nbbuild.cells)
    importlib.import_module(name)
    spans.append((name, start, len(nbbuild.cells)))
chapter_of = {i: name for name, s, e in spans for i in range(s, e)}

state = {'tag': 'x', 'n': 0}

def save_show(*a, **k):
    figs = [plt.figure(n) for n in plt.get_fignums()]
    for j, fig in enumerate(figs):
        state['n'] += 1
        suffix = f'_{j}' if len(figs) > 1 else ''
        fig.savefig(os.path.join(OUT, f"{state['tag']}{suffix}.png"), dpi=90)
    plt.close('all')

plt.show = save_show

def fake_interact(func, **kw):
    import ipywidgets as w
    defaults = {k: (v.value if hasattr(v, 'value') else v) for k, v in kw.items()}
    base = state['tag']
    variants = [('default', dict(defaults))]
    for k, v in kw.items():
        if hasattr(v, 'min') and hasattr(v, 'max'):
            variants += [(f'{k}=min', {**defaults, k: v.min}), (f'{k}=max', {**defaults, k: v.max})]
        elif isinstance(v, w.Dropdown):
            variants.append((f'{k}=last', {**defaults, k: v.options[-1]}))
    for label, args in variants:
        state['tag'] = f'{base}_{func.__name__}_{label}'
        try:
            func(**args)
            save_show()
        except Exception as e:
            print(f'  !! {state["tag"]}: {type(e).__name__}: {e}')
            plt.close('all')
    state['tag'] = base

def make_fake_anim(ns, cell_idx):
    def show_anim(make_scene):
        ch = chapter_of[cell_idx]
        if not ('all' in ANIM or ch in ANIM):
            return
        import manim as mn, manimpango
        manimpango.register_font(ns['font_file']())
        T = lambda s, **k: mn.Text(s, font='IBM Plex Sans JP', **k)
        with mn.tempconfig({'quality': 'low_quality', 'verbosity': 'WARNING', 'disable_caching': True,
                            'progress_bar': 'none', 'media_dir': os.path.join(HERE, '..', 'media')}):
            scene = make_scene(mn, T)()
            scene.render()
            path = str(scene.renderer.file_writer.movie_file_path)
        name = f'{cell_idx:03d}_{type(scene).__name__}'
        dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path],
                                   capture_output=True, text=True).stdout)
        n = 20
        fps = n / dur
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', path, '-vf',
                        f'fps={fps:.4f},scale=427:-1,tile=5x4:padding=4:color=gray', '-frames:v', '1',
                        os.path.join(OUT, f'{name}_sheet.png')], check=True)
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-sseof', '-0.3', '-i', path, '-frames:v', '1',
                        os.path.join(OUT, f'{name}_last.png')], check=True)
        print(f'  anim {name}: {dur:.1f}s')
    return show_anim

ns = {'__name__': '__main__'}
os.chdir(os.path.dirname(nbbuild.NB))
for i, c in enumerate(nbbuild.cells):
    if c['cell_type'] != 'code':
        continue
    src = ''.join(c['source'])
    src = src.replace('num_steps=1000', 'num_steps=30').replace('TEST_NAMES = docs[-300:]', 'TEST_NAMES = docs[-40:]')
    src = src.replace('train_adam(gpt_adam, all_params(gpt_adam_state))', 'train_adam(gpt_adam, all_params(gpt_adam_state), num_steps=30)')
    first = src.split('\n')[0]
    state['tag'] = f'{i:03d}_' + re.sub(r'[^\w]+', '', first.replace('#@title', ''))[:24]
    if 'show_anim' in ns:
        ns['show_anim'] = make_fake_anim(ns, i)
    try:
        exec(compile(src, f'cell{i}', 'exec'), ns)
    except Exception as e:
        print(f'!! cell {i} ({chapter_of[i]}): {type(e).__name__}: {e}')
    save_show()
    if 'interact' in ns and ns['interact'] is not fake_interact:
        ns['interact'] = fake_interact
    if 'show_anim' in ns:
        ns['show_anim'] = make_fake_anim(ns, i)
print('done')

if '--sheets' in sys.argv:
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(ns.get('font_file', lambda: 'IBMPlexSansJP-Regular.ttf')(), 18)
    pngs = sorted(p for p in glob.glob(os.path.join(OUT, '*.png')) if not os.path.basename(p).startswith('sheet_') and '_sheet' not in p and '_last' not in p)
    per = 6
    for k in range(0, len(pngs), per):
        group = pngs[k:k + per]
        ims = []
        for p in group:
            im = Image.open(p).convert('RGB')
            im.thumbnail((700, 520))
            canvas = Image.new('RGB', (700, im.height + 26), 'white')
            canvas.paste(im, (0, 26))
            ImageDraw.Draw(canvas).text((4, 2), os.path.basename(p)[:-4], fill='red', font=font)
            ims.append(canvas)
        rows = [ims[r:r + 2] for r in range(0, len(ims), 2)]
        H = sum(max(i.height for i in row) for row in rows)
        sheet = Image.new('RGB', (1400, H), 'white')
        y = 0
        for row in rows:
            for j, im in enumerate(row):
                sheet.paste(im, (700 * j, y))
            y += max(i.height for i in row)
        sheet.save(os.path.join(OUT, f'sheet_{k // per:02d}.png'))
    print('sheets', math.ceil(len(pngs) / per))
