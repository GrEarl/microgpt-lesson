from nbbuild import md, code

md(r"""
## 0.3 このNotebookの進め方 - How to use this Notebook

ここからは，0.1のプログラムを部品ごとに分解して，1章ずつ作っていきます。
各章は「なぜそれが必要か」→「必要な数学」→「実装」の順に進みます。
> From here, we will take the program from 0.1 apart and build it piece by piece, one chapter at a time.
Each chapter goes in this order: "why we need it" → "the math we need" → "implementation".

### 目次 - Contents

| 章 | 作るもの | 使う数学 |
|---|---|---|
| 第1章 | 文字を数字に: Tokenizer | 集合，対応 |
| 第2章 | 数えて名前を作る: bigramモデル | 確率，条件付き確率 |
| 第3章 | どれくらい外れた？: 損失とsoftmax | 指数・対数 |
| 第4章 | lossを小さくする: 勾配降下法 | 二次関数，微分 |
| 第5章 | 微分を自動で: 計算グラフと連鎖律 ★ | 合成関数 |
| 第6章 | 文字をベクトルに: 埋め込みと線形変換 | ベクトル，内積，三角比 |
| 第7章 | ニューラルネットワーク: MLPとReLU | 関数とグラフ |
| 第8章 | 前の文字に注目する: Attention | 内積，加重平均 |
| 第9章 | 学習を安定させる: RMSNormと残差接続 | 平均，分散，標準偏差 |
| 第10章 | 賢い下り方: Adam | 移動平均 |
| 第11章 | 全部組み立てる: MicroGPTの完成 | ここまでの全部 |
| 付録 | 用語集・公式まとめ・参考資料 | |

数学IIや数学Cで習う内容(指数・対数，微分，ベクトル)は，数学Iと数学Aの知識から出発して，必要な分だけ説明します。事前に知らなくても大丈夫です。
> Topics from Math II and Math C (exponents, logarithms, derivatives, vectors) are explained from what you know in Math I and Math A, only as much as we need, so you do not need to know them in advance.

### 記号の意味 - What the marks mean

- **[スライダー]**: 自分で値を動かして，結果がどう変わるかを確かめられるセルです。数式を見るだけでなく，ぜひ動かしてみてください。
- **[アニメーション]**: 動きで概念を説明する動画です。動画はあらかじめ保存されているので，実行しなくても見られます。
- **★**: 少し難しい節です。最初は飛ばして先に進んでも大丈夫です。
> - **[Slider]**: A cell where you can move values yourself and see how the result changes. Please try moving them, not just reading the formulas.
> - **[Animation]**: A video that explains an idea with movement. The videos are already saved, so you can watch them without running anything.
> - **★**: A harder section. It is fine to skip it at first.

### セルの実行について - About running cells

1. まず下の「共通準備」セルを実行してください。途中の章から始めるときも，このセルだけは必ず先に実行します。
2. あとは上から順番に実行していきましょう。学習(訓練)をするセルには，かかる時間の目安を書いてあります。
3. [アニメーション] のセルを実行すると，動画を作り直します。そのためには「アニメーションの準備」セルを先に実行しておく必要があります(準備なしで実行すると，保存されていた動画が消えてしまいます。そのときはNotebookを開き直してください)。
> 1. First, run the "common setup" cell below. Even when you start from a later chapter, always run this cell first.
> 2. Then run the cells in order from top to bottom. Cells that train a model show a rough estimate of how long they take.
> 3. Running an [Animation] cell re-creates the video. To do that, run the "animation setup" cell first. (If you run it without setup, the saved video disappears. In that case, reopen the Notebook.)

> グラフや動画を描くためのコードは，Colabでは折りたたんで隠してあります。中身が気になる人は，セルのタイトルをダブルクリックすると見られます。
""")

code(r'''
#@title 共通準備(最初に必ず実行してね){ display-mode: "form" }
# このNotebookのどこから始めても必要になる道具を，まとめて用意するセルだよ
import os, math, random, csv, urllib.request
import matplotlib.pyplot as plt
from matplotlib import font_manager
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display, Video

# --- 学習データ: 0.1と同じ手順で読み込んで，同じ順番にシャッフルするよ ---
if not os.path.exists('first_name_woman_org.csv'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/shuheilocale/japanese-personal-name-dataset/main/japanese_personal_name_dataset/dataset/first_name_woman_org.csv', 'first_name_woman_org.csv')
with open('first_name_woman_org.csv', encoding='utf-8') as f:
    docs = [row[0] for row in csv.reader(f) if row]
random.seed(42)
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
def tok_str(i): return 'BOS' if i == BOS else uchars[i]  # Tokenの番号を表示用の文字に戻すよ

# --- グラフで日本語を表示するためのフォントと洗練されたスタイル ---
FONT_URL = 'https://github.com/google/fonts/raw/main/ofl/ibmplexsansjp/IBMPlexSansJP-Regular.ttf'
FONT_PATH = 'IBMPlexSansJP-Regular.ttf'
def font_file():
    if not os.path.exists(FONT_PATH):
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
    return os.path.abspath(FONT_PATH)
def _setup_font():
    try:
        font_manager.fontManager.addfont(font_file())   # 同梱フォントを使うよ(環境差で文字化けしないように)
        return font_manager.FontProperties(fname=font_file()).get_name()
    except Exception:
        pass
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ['IBM Plex Sans JP', 'BIZ UDPGothic', 'Noto Sans CJK JP', 'Noto Sans JP', 'Yu Gothic', 'Meiryo', 'Hiragino Sans']:
        if name in have:
            return name
    return 'sans-serif'

# 読みやすく洗練されたプロット用テーマ設定
plt.rcParams.update({
    'font.family': _setup_font(),
    'axes.unicode_minus': False,
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#cbd5e1',
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.color': '#f1f5f9',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.titlepad': 10,
})

# --- 0.1で使った道具(各章で1つずつ作り直していくよ) ---
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1
    def backward(self):
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

# --- [アニメーション] を作り直すための道具 ---
def show_anim(make_scene):
    try:
        import manim as mn, manimpango
    except ImportError:
        print('このアニメーションを作り直すには，0.3の「アニメーションの準備」セルを先に実行してね。')
        return
    manimpango.register_font(font_file())
    fonts = set(manimpango.list_fonts())
    font = next((f for f in ['IBM Plex Sans JP', 'BIZ UDPGothic', 'Noto Sans CJK JP', 'Noto Sans JP', 'Yu Gothic'] if f in fonts), '')
    T = lambda s, **k: mn.Text(s, font=font, **k)
    with mn.tempconfig({'quality': 'low_quality', 'verbosity': 'WARNING', 'disable_caching': True,
                        'progress_bar': 'none', 'media_dir': 'media'}):
        scene = make_scene(mn, T)()
        scene.render()
        path = str(scene.renderer.file_writer.movie_file_path)
    display(Video(path, embed=True, width=640, html_attributes='controls loop autoplay muted playsinline'))

print(f'準備完了！ 学習データ {len(docs)}個 / 語彙の大きさ {vocab_size}')
''')

code(r'''
#@title アニメーションの準備(アニメーションを自分で作り直したい人だけ・2〜3分かかるよ){ display-mode: "form" }
import sys, subprocess, importlib.util
if importlib.util.find_spec('manim') is None:
    if 'google.colab' in sys.modules:
        subprocess.run('apt-get -qq update && apt-get -qq install -y libpango1.0-dev libcairo2-dev ffmpeg > /dev/null', shell=True, check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'manim'], check=True)
import manim
print(f'manim {manim.__version__} の準備ができたよ')
''')
