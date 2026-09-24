from nbbuild import md, code

md(r"""
---
# 第6章 文字をベクトルに: 埋め込みと線形変換 - Chapter 6: Characters as vectors: embeddings and linear transformations

第1章では文字に番号を付けました。でも番号には「あ(0)と い(1)は近い」のような意味はありません。
この章では，1つの文字を **いくつかの数の組(ベクトル)** で表す方法(**埋め込み**)と，ベクトルを変換する **線形変換** を学びます。
そして，いよいよ `Value` と勾配降下法を使って，初めてのニューラルネットワークを学習させます。
> In Chapter 1, we gave numbers to characters. But the numbers have no meaning like "あ (0) and い (1) are close".
In this chapter, we learn how to represent one character as **a group of several numbers (a vector)** (**embedding**), and **linear transformations**, which transform vectors.
Then, using `Value` and gradient descent, we train our first neural network.

**この章で使う数学**: ベクトル(数学Cの内容をここで学びます)，三角比(数学I「図形と計量」)，三平方の定理(中学)
> **Math in this chapter**: vectors (a Math C topic, learned here), trigonometric ratios (Math I), the Pythagorean theorem (junior high)

## 6.1 番号だけでは「似ている」を表せない - Numbers alone cannot express "similar"

「こ」と「み」は，どちらも名前の最後によく来る文字です。だから，モデルにとっては「似た性質」を持っています。
でも番号で見ると「こ」は14番，「み」は53番で遠く，逆に「さ」は16番で「こ」のすぐ近くです。番号の近さと性質の近さは関係ありません。
> "こ" and "み" both often come at the end of names, so for the model they have "similar properties".
But as numbers, "こ" is 14 and "み" is 53, far apart, while "さ" is 16, right next to "こ". How close the numbers are has nothing to do with how similar the properties are.

そこで，1つの文字をいくつかの数の組で表すことにします。例えば
> So we represent each character by a group of numbers. For example:

- 「こ」→ $(0.9,\ -0.2)$
- 「み」→ $(0.8,\ -0.1)$
- 「さ」→ $(-0.7,\ 0.5)$

のように表せば，「こ」と「み」は近く，「さ」は遠い，ということを表せます。
しかも，**この数の組そのものを学習で決める** のです。
> This way we can express that "こ" and "み" are close and "さ" is far.
Moreover, **these groups of numbers themselves are decided by learning**.

## 6.2 ベクトル: 数の組 - Vectors: groups of numbers

$(3,\ 1)$ のような数の組を **ベクトル** といいます。平面上の **矢印** だと考えることもできます(原点から点 $(3, 1)$ へ向かう矢印)。
ベクトルの足し算と実数倍は，成分ごとに計算します。
> A group of numbers like $(3,\ 1)$ is called a **vector**. You can also think of it as an **arrow** on a plane (from the origin to the point $(3, 1)$).
Adding vectors and multiplying them by a number are done component by component:

$$(a_1,\ a_2) + (b_1,\ b_2) = (a_1 + b_1,\ a_2 + b_2), \qquad k(a_1,\ a_2) = (k a_1,\ k a_2)$$

ベクトルの長さ $|\vec{a}|$ は，三平方の定理から $|\vec{a}| = \sqrt{a_1^2 + a_2^2}$ です。
> The length of a vector $|\vec{a}|$ is $\sqrt{a_1^2 + a_2^2}$, by the Pythagorean theorem.

Pythonでは，ベクトルはリストで表します。成分が2つでも16個でも，考え方は同じです。
> In Python, we represent vectors as lists. Whether there are 2 components or 16, the idea is the same.

### [スライダー] ベクトルの足し算 - [Slider] Adding vectors
""")

code(r'''
#@title [スライダー] ベクトルの和と実数倍 { display-mode: "form" }
def plot_vectors(a1=3.0, a2=1.0, b1=1.0, b2=2.0, k=1.0):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    def arrow(x0, y0, x, y, c, l, lw=2.2, alpha=1.0):
        ax.annotate('', xy=(x0 + x, y0 + y), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=c, lw=lw, alpha=alpha))
        if l: ax.plot([], [], color=c, lw=lw, label=l)
    arrow(0, 0, k * a1, k * a2, '#10b981', f'{k:.2f}a = ({k * a1:.1f}, {k * a2:.1f})', lw=5, alpha=0.3)
    arrow(0, 0, a1, a2, '#3b82f6', f'a = ({a1:.1f}, {a2:.1f})')
    arrow(0, 0, b1, b2, '#f97316', f'b = ({b1:.1f}, {b2:.1f})')
    arrow(a1, a2, b1, b2, '#f97316', None, alpha=0.5)
    arrow(0, 0, a1 + b1, a2 + b2, '#ef4444', f'a + b = ({a1 + b1:.1f}, {a2 + b2:.1f})', lw=2.8)
    ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect('equal')
    ax.axhline(0, color='#64748b', lw=0.8); ax.axvline(0, color='#64748b', lw=0.8)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.set_title('ベクトルの足し算 (平行四辺形の対角線)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

sl = lambda v, d: widgets.FloatSlider(value=v, min=-4, max=4, step=0.5, description=d)
interact(plot_vectors, a1=sl(3, 'a の x:'), a2=sl(1, 'a の y:'), b1=sl(1, 'b の x:'), b2=sl(2, 'b の y:'),
         k=widgets.FloatSlider(value=1.0, min=-2, max=2, step=0.25, description='倍率 k:'));
''')

md(r"""
## 6.3 内積: 2つのベクトルの「似ている度合い」 - The dot product: how similar two vectors are

2つのベクトルについて，**成分どうしを掛けて足したもの** を **内積** といいます。
> For two vectors, **multiplying matching components and adding them up** gives the **dot product**:

$$\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2$$

実は，内積は2つのベクトルのなす角 $\theta$ を使って，次のようにも表せます(数学Iで習う $\cos$ です)。
> In fact, the dot product can also be written with the angle $\theta$ between the two vectors (the $\cos$ from Math I):

$$\vec{a} \cdot \vec{b} = |\vec{a}|\,|\vec{b}| \cos\theta$$

$\cos\theta$ は，$\theta = 0°$ で $1$，$90°$ で $0$，$180°$ で $-1$ でしたね。だから内積は
> $\cos\theta$ is $1$ at $0°$, $0$ at $90°$, and $-1$ at $180°$. So the dot product is:

- 同じ向き → 大きいプラス
- 直角 → 0
- 反対向き → マイナス
> - Same direction → large positive
> - Perpendicular → 0
> - Opposite directions → negative

となり，**2つのベクトルがどれくらい同じ向きを向いているか(似ているか)** を表す数になります。これは第8章のAttentionで大活躍します。
> So it is a number that shows **how much two vectors point in the same direction (how similar they are)**. This plays a big role in Attention in Chapter 8.

### [アニメーション] 向きが変わると内積が変わる - [Animation] The dot product changes with direction
""")

code(r'''
#@title アニメーション: 内積と角度 { display-mode: "form" }
def make_scene(mn, T):
    class DotProduct(mn.Scene):
        def construct(self):
            plane = mn.NumberPlane(x_range=[-4, 4], y_range=[-3, 3], x_length=7, y_length=5.25, background_line_style={'stroke_opacity': 0.3}).shift(mn.LEFT * 2.5)
            title = T('内積: 同じ向きほど値が大きくなる！', font_size=32).to_edge(mn.UP)
            self.play(mn.Create(plane), mn.Write(title))
            a = [2.5, 0.0]
            va = mn.Arrow(plane.c2p(0, 0), plane.c2p(*a), buff=0, color=mn.BLUE)
            la = T('a', font_size=30, color=mn.BLUE).next_to(va.get_end(), mn.DOWN)
            ang = mn.ValueTracker(20)
            bvec = lambda: [2 * math.cos(math.radians(ang.get_value())), 2 * math.sin(math.radians(ang.get_value()))]
            vb = mn.always_redraw(lambda: mn.Arrow(plane.c2p(0, 0), plane.c2p(*bvec()), buff=0, color=mn.ORANGE))
            def info():
                b = bvec(); d = a[0] * b[0] + a[1] * b[1]
                col = mn.GREEN if d > 0.05 else (mn.RED if d < -0.05 else mn.WHITE)
                return mn.VGroup(T(f'角度 θ = {ang.get_value():.0f}°', font_size=28),
                                 T(f'cos θ = {math.cos(math.radians(ang.get_value())):+.2f}', font_size=28),
                                 T(f'内積 = {d:+.2f}', font_size=34, color=col)).arrange(mn.DOWN, aligned_edge=mn.LEFT).to_edge(mn.RIGHT).shift(mn.LEFT * 0.3)
            txt = info()
            self.play(mn.GrowArrow(va), mn.FadeIn(la), mn.FadeIn(vb), mn.FadeIn(txt))
            for target in [60, 90, 120, 180, 135, 45, 20]:
                self.play(ang.animate.set_value(target), run_time=0.8)
                self.play(mn.Transform(txt, info()), run_time=0.3)
                self.wait(0.4 if target in (90, 180) else 0.1)
            self.wait(1)
    return DotProduct

show_anim(make_scene)
''')

md(r"""
### [スライダー] 角度と長さを変えて内積を見よう - [Slider] Change the angle and length and see the dot product
""")

code(r'''
#@title [スライダー] 内積 { display-mode: "form" }
def plot_dot(angle=45, length_b=2.0):
    a = (2.5, 0.0)
    b = (length_b * math.cos(math.radians(angle)), length_b * math.sin(math.radians(angle)))
    dot = a[0] * b[0] + a[1] * b[1]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for v, c, l in [(a, '#3b82f6', 'a'), (b, '#f97316', 'b')]:
        ax.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=c, lw=3))
        ax.text(v[0] * 1.12, v[1] * 1.12, l, color=c, fontsize=14, fontweight='bold')
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect('equal')
    ax.set_title(f'成分計算: a·b = {a[0]}×{b[0]:.2f} + {a[1]}×{b[1]:.2f} = {dot:.2f}\n幾何定義: |a||b|cosθ = {2.5 * length_b * math.cos(math.radians(angle)):.2f}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_dot, angle=widgets.IntSlider(value=45, min=0, max=360, step=5, description='角度 θ:'),
         length_b=widgets.FloatSlider(value=2.0, min=0, max=3, step=0.1, description='b の長さ:'));
''')

md(r"""
2つの式で計算した値が一致することを確かめてください。
> Check that the values calculated by the two formulas match.

## 6.4 埋め込み: 文字をベクトルにする表 - Embeddings: a table that turns characters into vectors

**埋め込み(embedding)** は，とてもシンプルです。語彙の大きさ × ベクトルの成分の数(`n_embd`)の **表** を用意して，Token番号の行をそのまま取り出すだけです。
0.1のコードでは `state_dict['wte']` がこの表で，大きさは $72 \times 16$ でした(wteは word token embedding の略です)。
> An **embedding** is very simple. We prepare a **table** of size vocabulary × number of vector components (`n_embd`) and just take out the row for the token number.
In 0.1, `state_dict['wte']` is this table, of size $72 \times 16$ (wte stands for word token embedding).

```python
tok_emb = state_dict['wte'][token_id]   # token_id 行目を取り出すだけで，その文字のベクトルになるよ
```

最初は表の中身はでたらめな数ですが，学習によって「似た使われ方をする文字は似たベクトル」になるように調整されていきます。
同じように，「何文字目か」を表すベクトルの表 `wpe`(word position embedding)もあります。これは第8章で使います。
> At first the table is filled with random numbers, but learning adjusts it so that "characters used in similar ways get similar vectors".
Similarly, there is a table `wpe` (word position embedding) with vectors for "which position". We will use it in Chapter 8.

## 6.5 行列と線形変換 - Matrices and linear transformations

数を長方形に並べたものを **行列** といいます。行列は「ベクトルを何本か重ねたもの」と考えることができます。
0.1のコードの `linear(x, w)` は，**行列 `w` の各行と，ベクトル `x` の内積を並べたもの** を計算しています。これを **線形変換** といいます。
> Numbers arranged in a rectangle are called a **matrix**. You can think of a matrix as "several vectors stacked together".
`linear(x, w)` in 0.1 calculates **the dot product of each row of the matrix `w` with the vector `x`, and lines them up**. This is called a **linear transformation**.

$$\begin{pmatrix} 1 & 2 \\ 0 & -1 \\ 3 & 1 \end{pmatrix} \text{ で } \begin{pmatrix} 2 \\ 1 \end{pmatrix} \text{ を変換} \;\to\; \begin{pmatrix} 1 \cdot 2 + 2 \cdot 1 \\ 0 \cdot 2 + (-1) \cdot 1 \\ 3 \cdot 2 + 1 \cdot 1 \end{pmatrix} = \begin{pmatrix} 4 \\ -1 \\ 7 \end{pmatrix}$$

各行は「ある特徴を調べるための物差し」のようなもので，内積が大きいほど「その特徴に近い」ことを表します。
行の数を変えれば，**2成分のベクトルを3成分に** のように，ベクトルの成分の数も変えられます。
> Each row is like "a ruler that checks for a certain feature", and a larger dot product means "closer to that feature".
By changing the number of rows, we can also change the number of components, like **from 2 components to 3**.
""")

code(r'''
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]  # 各行 wo と入力 x の内積を計算してリストにするよ

W = [[1, 2], [0, -1], [3, 1]]
print('線形変換の計算結果:', linear([2, 1], W))
''')

md(r"""
### [アニメーション] 線形変換の計算 - [Animation] How a linear transformation is calculated
""")

code(r'''
#@title アニメーション: 線形変換 = 各行との内積 { display-mode: "form" }
def make_scene(mn, T):
    class LinearScene(mn.Scene):
        def construct(self):
            W = [[1, 2], [0, -1], [3, 1]]; x = [2, 1]
            title = T('linear(x, W): 行列の各行と x の内積を並べる', font_size=32).to_edge(mn.UP)
            self.play(mn.Write(title))
            def mat(rows_, color=mn.WHITE):
                rows_g = mn.VGroup(*[mn.VGroup(*[T(str(e), font_size=36, color=color) for e in r]).arrange(mn.RIGHT, buff=0.6) for r in rows_]).arrange(mn.DOWN, buff=0.4)
                for r in rows_g:
                    for j, e in enumerate(r): e.set_x(rows_g[0][j].get_x())
                h = rows_g.height + 0.3
                lb = mn.VGroup(mn.Line(mn.UP * h / 2, mn.DOWN * h / 2), mn.Line(mn.UP * h / 2, mn.UP * h / 2 + mn.RIGHT * 0.15), mn.Line(mn.DOWN * h / 2, mn.DOWN * h / 2 + mn.RIGHT * 0.15)).next_to(rows_g, mn.LEFT, buff=0.15)
                rb = lb.copy().flip(mn.UP).next_to(rows_g, mn.RIGHT, buff=0.15)
                return mn.VGroup(rows_g, lb, rb)
            m = mat(W).shift(mn.LEFT * 3.5)
            v = mat([[e] for e in x]).next_to(m, mn.RIGHT)
            eq = T('=', font_size=40).next_to(v, mn.RIGHT)
            out = [sum(a * b for a, b in zip(row, x)) for row in W]
            res = mat([[o] for o in out], color=mn.YELLOW).next_to(eq, mn.RIGHT)
            for e in res[0]: e.set_opacity(0)
            self.play(mn.FadeIn(m), mn.FadeIn(v), mn.FadeIn(eq), mn.FadeIn(res[1]), mn.FadeIn(res[2]))
            rows = m[0]
            for i, row in enumerate(W):
                box = mn.SurroundingRectangle(rows[i], color=mn.YELLOW)
                vbox = mn.SurroundingRectangle(v[0], color=mn.YELLOW)
                calc = T(f'{row[0]}×{x[0]} + {row[1]}×{x[1]} = {out[i]}', font_size=30, color=mn.YELLOW).to_edge(mn.DOWN).shift(mn.UP * 0.5)
                self.play(mn.Create(box), mn.Create(vbox), mn.Write(calc), run_time=0.7)
                self.play(res[0][i].animate.set_opacity(1), run_time=0.5)
                self.play(mn.FadeOut(box), mn.FadeOut(vbox), mn.FadeOut(calc), run_time=0.4)
            note = T('3行の行列 → 2成分のベクトルが3成分に変わるよ', font_size=26).to_edge(mn.DOWN)
            self.play(mn.Write(note))
            self.wait(1.5)
    return LinearScene

show_anim(make_scene)
''')

md(r"""
## 6.6 実装: 埋め込みを使った初めてのニューラルネットワーク - Implementation: our first neural network with embeddings

いよいよ，埋め込みと線形変換を使ったモデルを学習させます。仕組みは次のとおりです。
> Now let's train a model using embeddings and a linear transformation. It works like this:

1. 前の文字のTokenから，埋め込みの表 `wte` でベクトルを取り出す
2. そのベクトルを線形変換 `lm_head`(72行)で，72個のスコアに変える
3. softmaxで確率にして，lossを計算する(第3章)
4. `backward()` で傾きを求めて(第5章)，勾配降下法でパラメーターを少し動かす(第4章)
> 1. From the token of the previous character, take out a vector from the embedding table `wte`.
> 2. Turn that vector into 72 scores with the linear transformation `lm_head` (72 rows).
> 3. Turn them into probabilities with softmax and calculate the loss (Chapter 3).
> 4. Find the slopes with `backward()` (Chapter 5) and move the parameters a little with gradient descent (Chapter 4).

図にできるように，まずはベクトルの成分を **2つ** にしてみます(`n_embd = 2`)。
まず，この先の章でも使う道具をまとめて用意しておきます。
> So that we can draw it, we first use **2** components (`n_embd = 2`).
First, let's prepare tools that we will also use in later chapters.
""")

code(r'''
# パラメーターの表を作る: 平均0，標準偏差stdの正規分布(第9章)に従う乱数で埋めるよ
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
def all_params(state): return [p for mat in state.values() for row in mat for p in row]
block_size = max(len(doc) for doc in docs) + 1

def name_loss(model, name):
    """1つの名前についての平均loss。model(tokens) は各位置の「次のTokenのスコア」のリストを返す関数だよ"""
    tokens = tokenize(name)
    logits_list = model(tokens[:-1])
    losses = [-softmax(logits)[target].log() for logits, target in zip(logits_list, tokens[1:])]
    return sum(losses) * (1 / len(losses))

def train_sgd(model, params, num_steps=1000, learning_rate=0.5):
    """1ステップに1つの名前を使って，勾配降下法で学習するよ"""
    history = []
    for step in range(num_steps):
        loss = name_loss(model, docs[step % len(docs)])
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)   # 学習率をだんだん小さくするよ(第4章)
        for p in params:
            p.data -= lr_t * p.grad
            p.grad = 0
        history.append(loss.data)
        if (step + 1) % 100 == 0:
            print(f'step {step + 1:4d} / {num_steps} | 直近100回の平均loss: {sum(history[-100:]) / 100:.4f}', end='\r')
    print()
    return history

TEST_NAMES = docs[-300:]  # 学習には使わない，テスト用の300個の名前だよ
def test_loss(model):
    return sum(name_loss(model, name).data for name in TEST_NAMES) / len(TEST_NAMES)

def generate(model, n=10, temperature=0.5, seed=0):
    rng, names = random.Random(seed), []
    for _ in range(n):
        tokens = [BOS]
        while len(tokens) < block_size:
            logits = model(tokens)[-1]
            probs = softmax([l / temperature for l in logits])
            token = rng.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token == BOS:
                break
            tokens.append(token)
        names.append(''.join(itos[t] for t in tokens[1:]))
    return names

def plot_history(histories, window=50):
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for label, h in histories.items():
        avg = [sum(h[max(0, i - window + 1):i + 1]) / len(h[max(0, i - window + 1):i + 1]) for i in range(len(h))]
        ax.plot(avg, label=label, lw=1.8)
    ax.axhline(BIGRAM_TEST_LOSS, color='#64748b', ls='--', lw=1.5, label='数えるbigram(テスト)')
    ax.set_xlabel('ステップ', fontsize=11)
    ax.set_ylabel(f'loss(直近{window}回の平均)', fontsize=11)
    ax.set_ylim(2.2, 4.5)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()
''')

md(r"""
**テスト用の名前について**: 学習に使った名前でlossを測ると，「答えを丸暗記しただけ」でもlossが下がってしまいます。
そこで，学習には使わない300個の名前を取っておいて，それでlossを測ります(`test_loss`)。
比べる基準として，第2章の「数えるbigramモデル」のテストlossも計算しておきます。
(テスト用の名前を除いた残りで数え直し，一度も出てこなかった組の確率が0にならないように，全部の組に0.1回ずつ足しておきます)
> **About the test names**: If we measure the loss on names used for training, the loss can go down even by "just memorizing the answers".
So we keep 300 names that are not used for training and measure the loss with them (`test_loss`).
As a baseline, we also calculate the test loss of the "counting bigram model" from Chapter 2.
(We count again without the test names, and add 0.1 to every pair so that pairs that never appeared do not get probability 0.)
""")

code(r'''
train_counts = [[0.1] * vocab_size for _ in range(vocab_size)]
for name in docs[:-300]:
    ids = tokenize(name)
    for a, b in zip(ids, ids[1:]):
        train_counts[a][b] += 1
total, n = 0.0, 0
for name in TEST_NAMES:
    ids = tokenize(name)
    for a, b in zip(ids, ids[1:]):
        total += -math.log(train_counts[a][b] / sum(train_counts[a]))
        n += 1
BIGRAM_TEST_LOSS = total / n
print(f'数えるbigramモデルのテストloss: {BIGRAM_TEST_LOSS:.4f}')
''')

md(r"""
それでは，埋め込みモデルを作って学習させましょう(数秒で終わります)。
> Now let's build the embedding model and train it (it takes a few seconds).
""")

code(r'''
random.seed(0)
n_embd = 2
emb_state = {'wte': matrix(vocab_size, n_embd, std=1.0),   # 2成分と小さいので，最初の値を少し大きめにしているよ
             'lm_head': matrix(vocab_size, n_embd, std=1.0)}
emb_params = all_params(emb_state)
print('パラメーター数:', len(emb_params))

def emb_model(tokens):
    logits_list = []
    for token in tokens:
        x = emb_state['wte'][token]                           # 1. 埋め込み
        logits_list.append(linear(x, emb_state['lm_head']))   # 2. 線形変換でスコアに
    return logits_list

emb_history = train_sgd(emb_model, emb_params, num_steps=1000, learning_rate=2.0)
print(f'テストloss: {test_loss(emb_model):.4f}')
print('作った名前:', ' '.join(generate(emb_model, 12)))
plot_history({'埋め込みモデル (n_embd=2)': emb_history})
''')

md(r"""
lossは下がりましたが，作った名前は1文字のものが多く，まだあまり名前らしくありません。成分が2つだけでは，文字の違いを表すのに足りないのです。
それでも，学習した埋め込みベクトルを平面上に描いてみると面白いことがわかります。各文字が，学習で決まった2成分のベクトルの位置に置かれます。
> The loss went down, but many of the names it makes are only one character long, so they do not look much like names yet. Just 2 components are not enough to express the differences between characters.
Still, drawing the learned embedding vectors on a plane shows something interesting. Each character is placed at the position of its learned 2-component vector.
""")

code(r'''
#@title 学習した埋め込みベクトル { display-mode: "form" }
fig, ax = plt.subplots(figsize=(8, 7.5))
for i in range(vocab_size):
    x, y = emb_state['wte'][i][0].data, emb_state['wte'][i][1].data
    big = sum(counts[i]) > 150
    ax.text(x, y, tok_str(i), fontsize=13 if big else 9, ha='center', va='center',
            color='#1e293b' if big else '#94a3b8', fontweight='bold' if big else 'normal')
xs = [r[0].data for r in emb_state['wte']]; ys = [r[1].data for r in emb_state['wte']]
ax.set_xlim(min(xs) - 0.4, max(xs) + 0.4); ax.set_ylim(min(ys) - 0.4, max(ys) + 0.4)
ax.set_title('学習した文字のベクトル (よく出る文字は大きく表示)', fontsize=12, fontweight='bold')
ax.set_xlabel('埋め込み次元 1', fontsize=10)
ax.set_ylabel('埋め込み次元 2', fontsize=10)
plt.tight_layout()
plt.show()
''')

md(r"""
よく出てくる文字どうしで，近くに集まっているものがあるか探してみましょう。
このモデルでは，近くにある文字は「次に来る文字の傾向」が似ています。どの文字が近くに来るかは，学習をやり直すと変わります。
> Look for frequent characters that are gathered close together.
In this model, characters that are close have similar "tendencies for the next character". Which characters end up close changes if you train again.

**でも，このモデルもまだ直前の1文字しか見ていません。** テストlossは数えるbigramモデルと同じくらいか，それより悪いはずです。
しかも成分が2つだけでは，72種類の文字の違いを十分に表せません。0.1のGPTのように `n_embd = 16` にして，さらに次の章で「ニューラルネットワーク」らしい仕組みを加えていきます。
> **But this model still only looks at the one previous character.** Its test loss should be about the same as the counting bigram model, or worse.
Also, just 2 components are not enough to express the differences among 72 characters. In the next chapter, we use `n_embd = 16` like the GPT in 0.1, and add a mechanism that is more like a real "neural network".

## 6.7 0.1のコードとの対応 - Matching with the code in 0.1

```python
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
def linear(x, w): ...                      # 6.5 線形変換だよ
tok_emb = state_dict['wte'][token_id]      # 6.4 埋め込みだよ
logits = linear(x, state_dict['lm_head'])  # 6.6 スコアに変換するよ
```

## 練習問題 - Exercises

1. $\vec{a} = (1, 2)$，$\vec{b} = (4, -2)$ の内積を求めましょう。2つのベクトルのなす角は何度でしょうか？
2. $\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$ でベクトル $(x, y)$ を変換すると，どうなるでしょうか？
3. `emb_model` のパラメーター数が288個になる理由を説明してみましょう。

<details><summary>答え - Answers</summary>

1. $1 \times 4 + 2 \times (-2) = 0$ です。内積が0なので，なす角は90°(直角)です。
2. $(1 \cdot x + 0 \cdot y,\ 0 \cdot x + 1 \cdot y) = (x, y)$ で，何も変わりません(このような行列を単位行列といいます)。
3. `wte` が $72 \times 2 = 144$ 個，`lm_head` が $72 \times 2 = 144$ 個で，合計288個です。
</details>
""")
