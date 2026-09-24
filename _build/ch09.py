from nbbuild import md, code

md(r"""
---
# 第9章 学習を安定させる: RMSNormと残差接続 - Chapter 9: Stabilizing training: RMSNorm and residual connections

AttentionとMLPを組み合わせ，さらに何層も重ねれば，もっと賢いモデルになりそうです。
でも実際には，計算を何層も重ねると **値が大きくなりすぎたり小さくなりすぎたりして，学習がうまくいかなくなる** という問題が起きます。
この章では，それを防ぐ2つの工夫，**RMSNorm** と **残差接続** を学んで，0.1のGPTを完成させます。
> Combining Attention and MLP and stacking many layers seems like it would make a smarter model.
But in practice, stacking many layers causes a problem: **values become too large or too small, and training stops working**.
In this chapter, we learn two tricks to prevent this, **RMSNorm** and **residual connections**, and complete the GPT from 0.1.

**この章で使う数学**: 平均・分散・標準偏差(数学I「データの分析」)，正規分布(数学Bの内容を少しだけ)
> **Math in this chapter**: mean, variance, standard deviation (Math I), the normal distribution (a small part of Math B)

## 9.1 復習: 平均・分散・標準偏差 - Review: mean, variance, standard deviation

数学Iの「データの分析」で，データの **散らばり** を表す数を学びました。
> In Math I, we learned numbers that describe how **spread out** data is:

- **平均** $\bar{x} = \dfrac{1}{n}\sum x_i$
- **分散** $s^2 = \dfrac{1}{n}\sum (x_i - \bar{x})^2$ (平均からのずれの2乗の平均)
- **標準偏差** $s = \sqrt{s^2}$ (分散の平方根。元のデータと同じ単位で散らばりを表す)
> - **Mean** $\bar{x} = \dfrac{1}{n}\sum x_i$
> - **Variance** $s^2 = \dfrac{1}{n}\sum (x_i - \bar{x})^2$ (the average of the squared differences from the mean)
> - **Standard deviation** $s = \sqrt{s^2}$ (the square root of the variance, in the same units as the data)
""")

code(r'''
data = [2, 4, 4, 4, 5, 5, 7, 9]
mean = sum(data) / len(data)
var = sum((x - mean) ** 2 for x in data) / len(data)
print('平均:', mean, ' 分散:', var, ' 標準偏差:', var ** 0.5)
''')

md(r"""
## 9.2 RMS: 二乗平均平方根 - RMS: root mean square

**RMS(Root Mean Square，二乗平均平方根)** は，「2乗して，平均して，平方根をとる」という，名前のとおりの計算です。
> **RMS (Root Mean Square)** is exactly what its name says: "square, take the mean, take the square root":

$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum x_i^2}$$

分散の式から平均を引く部分をなくしたもので，**ベクトルの成分がだいたいどれくらいの大きさか** を表します。平均が0なら，標準偏差と同じ値になります。
> It is the variance formula without subtracting the mean, and it shows **roughly how large the components of a vector are**. If the mean is 0, it equals the standard deviation.

## 9.3 RMSNorm: 大きさをそろえる - RMSNorm: making the size uniform

**RMSNorm** は，ベクトルの各成分を **RMSで割る** ことで，RMSが1になるようにそろえる処理です。
向き(成分どうしの比)は変わらず，大きさだけがそろいます。
> **RMSNorm** divides each component of a vector **by its RMS**, so that the RMS becomes 1.
The direction (the ratios between components) stays the same; only the size is made uniform.
""")

code(r'''
def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)   # 2乗の平均 (mean square) だよ
    scale = (ms + 1e-5) ** -0.5              # 1/√ms。0で割らないように，ほんの少し(0.00001)足しておくよ
    return [xi * scale for xi in x]

v = [3.0, -1.0, 2.0, 0.5]
nv = rmsnorm(v)
rms = lambda x: (sum(xi * xi for xi in x) / len(x)) ** 0.5
print('元のベクトル:', v, ' RMS =', round(rms(v), 3))
print('RMSNorm後  :', [round(x, 3) for x in nv], ' RMS =', round(rms(nv), 3))
''')

md(r"""
### [スライダー] ベクトルを何倍にしても，RMSNorm後は同じ - [Slider] No matter how much you scale the vector, RMSNorm gives the same result
""")

code(r'''
#@title [スライダー] RMSNorm によるスケーリングの安定化 { display-mode: "form" }
def plot_rmsnorm(scale=1.0):
    base = [3.0, -1.0, 2.0, 0.5, -2.5, 1.0]
    x = [scale * b for b in base]
    nx = rmsnorm(x)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax1.bar(range(6), x, color='#3b82f6', edgecolor='#1d4ed8')
    ax1.axhline(0, color='#64748b', lw=1.0)
    m = max(30, max(abs(v) for v in x) * 1.15)
    ax1.set_ylim(-m, m)
    ax1.set_title(f'元のベクトル (RMS = {rms(x):.2f})', fontsize=11, fontweight='bold')
    ax1.set_ylabel('値')

    ax2.bar(range(6), nx, color='#10b981', edgecolor='#047857')
    ax2.set_ylim(-3.0, 3.0)
    ax2.axhline(0, color='#64748b', lw=1.0)
    ax2.set_title(f'RMSNorm後 (RMS = {rms(nx):.2f}) - 常に一定！', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_rmsnorm, scale=widgets.FloatSlider(value=1.0, min=0.01, max=10, step=0.1, description='拡大倍率:'));
''')

md(r"""
## 9.4 なぜ大きさをそろえるの？ - Why make the size uniform?

線形変換を何回も重ねると，1回ごとに値がだいたい同じ倍率で大きく(または小さく)なっていきます。
1回で2倍になるなら，10層で $2^{10} = 1024$ 倍，1回で0.5倍なら $0.5^{10} \approx 0.001$ 倍です(第3章の指数ですね！)。
値が大きすぎると計算があふれ，小さすぎると傾きがほとんど0になって学習が進みません。
> When we stack many linear transformations, the values grow (or shrink) by roughly the same factor each time.
If they double each time, 10 layers make them $2^{10} = 1024$ times larger; if they halve, $0.5^{10} \approx 0.001$ times (exponents from Chapter 3!).
Values that are too large overflow; values that are too small make the slopes almost 0, so learning stops.

### [スライダー] 層を重ねたときの値の大きさ - [Slider] The size of values when stacking layers

ランダムな行列で線形変換を何回も重ねたとき，ベクトルのRMSがどう変わるかを見てみましょう。縦軸は対数の目盛りです(1目盛りで10倍)。
行列の数の散らばり(標準偏差)を変えると，RMSNormなし(青)ではすぐに爆発したり消えたりしますが，RMSNormあり(オレンジ)ではずっと1のままです。
> Let's see how the RMS of a vector changes when we stack linear transformations with random matrices. The vertical axis is logarithmic (each step is 10 times).
When you change the spread (standard deviation) of the numbers in the matrices, without RMSNorm (blue) the values quickly explode or vanish, but with RMSNorm (orange) they stay at 1.
""")

code(r'''
#@title [スライダー] 層を重ねたときのRMSの比較 (対数目盛り) { display-mode: "form" }
import numpy as np
def plot_depth(std=0.3, layers=20):
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, 16)
    x_plain, x_norm = x0.copy(), x0.copy()
    r_plain, r_norm = [np.sqrt(np.mean(x0**2))], [np.sqrt(np.mean(x0**2))]
    for _ in range(layers):
        W = rng.normal(0, std, (16, 16))
        x_plain = W @ x_plain
        x_norm = W @ x_norm
        x_norm = x_norm / np.sqrt(np.mean(x_norm**2) + 1e-5)
        r_plain.append(np.sqrt(np.mean(x_plain**2)))
        r_norm.append(np.sqrt(np.mean(x_norm**2)))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.semilogy(r_plain, 'o-', color='#ef4444', lw=1.8, ms=4, label='RMSNormなし (制御不能！)')
    ax.semilogy(r_norm, 'o-', color='#10b981', lw=2.0, ms=4, label='RMSNormあり (常に安定！)')
    ax.set_xlabel('層の深さ (レイヤー数)', fontsize=11)
    ax.set_ylabel('ベクトルのRMS (対数目盛り)', fontsize=11)
    ax.set_ylim(1e-10, 1e10)
    ax.set_title(f'層の深さとベクトルの大きさの推移 (重みのstd: {std:.2f})', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()

interact(plot_depth, std=widgets.FloatSlider(value=0.3, min=0.05, max=0.6, step=0.01, description='重みのバラつき:'),
         layers=widgets.IntSlider(value=20, min=1, max=50, step=1, description='層の深さ:'));
''')

md(r"""
16成分のベクトルでは，標準偏差が $\frac{1}{\sqrt{16}} = 0.25$ のときにだけ，ほぼ大きさが保たれます。それより少しでもずれると，指数関数的に爆発したり消えたりします。
RMSNormをはさめば，標準偏差がいくつでも大きさがそろいます。
> With 16-component vectors, the size is roughly kept only when the standard deviation is $\frac{1}{\sqrt{16}} = 0.25$. Even a small difference makes it explode or vanish exponentially.
With RMSNorm in between, the size stays uniform whatever the standard deviation.

## 9.5 残差接続: 近道を作る - Residual connections: making a shortcut

もう1つの工夫が **残差接続(Residual Connection)** です。AttentionやMLPの **出力に，入力をそのまま足します**。
> The other trick is the **residual connection**. We **add the input directly to the output** of Attention or MLP:

$$x \leftarrow x + f(x)$$

こうすると，AttentionやMLP($f$)は「入力をまるごと作り直す」のではなく，「入力に **足すべき修正分** だけを作ればいい」ことになります。
さらに大事なのが逆伝播です。第5章で見たように，**足し算の局所的な傾きは1** でした。だから傾きは，近道を通って **小さくならずに** 入力側まで届きます。
何層重ねても，傾きが途中で消えてしまわないのです。
> This way, Attention or MLP ($f$) does not need to "rebuild the whole input"; it only needs to produce "**the correction to add** to the input".
Even more important is the backward pass. As we saw in Chapter 5, **the local slope of addition is 1**. So the slope travels through the shortcut to the input side **without shrinking**.
No matter how many layers we stack, the slopes do not vanish along the way.

### [アニメーション] 残差接続と傾きの流れ - [Animation] Residual connections and the flow of slopes
""")

code(r'''
#@title アニメーション: 残差接続 { display-mode: "form" }
def make_scene(mn, T):
    class ResidualScene(mn.Scene):
        def construct(self):
            title = T('残差接続: x ← x + f(x)', font_size=34).to_edge(mn.UP)
            self.play(mn.Write(title))
            xin = T('x', font_size=36).move_to(mn.LEFT * 5.5)
            norm = mn.VGroup(mn.RoundedRectangle(width=2, height=0.9, corner_radius=0.15, color=mn.TEAL), T('RMSNorm', font_size=24)).move_to(mn.LEFT * 2.8 + mn.DOWN * 0.8)
            f = mn.VGroup(mn.RoundedRectangle(width=2.4, height=0.9, corner_radius=0.15, color=mn.BLUE), T('Attention / MLP', font_size=22)).move_to(mn.RIGHT * 0.2 + mn.DOWN * 0.8)
            plus = mn.VGroup(mn.Circle(0.3, color=mn.YELLOW), T('+', font_size=34, color=mn.YELLOW)).move_to(mn.RIGHT * 3)
            xout = T('次の層へ', font_size=28).move_to(mn.RIGHT * 5.3)
            a1 = mn.Arrow(xin.get_right(), norm.get_left(), buff=0.15)
            a2 = mn.Arrow(norm.get_right(), f.get_left(), buff=0.1)
            a3 = mn.Arrow(f.get_right(), plus.get_bottom() + mn.LEFT * 0.1, buff=0.1)
            skip = mn.CurvedArrow(xin.get_top() + mn.UP * 0.1, plus.get_top(), angle=-0.8, color=mn.YELLOW)
            skip_lab = T('近道ハイウェイ (そのまま足す)', font_size=24, color=mn.YELLOW).move_to(mn.UP * 1.9)
            a4 = mn.Arrow(plus.get_right(), xout.get_left(), buff=0.1)
            self.play(mn.FadeIn(xin), mn.FadeIn(norm), mn.FadeIn(f), mn.FadeIn(plus), mn.FadeIn(xout), *[mn.GrowArrow(a) for a in (a1, a2, a3, a4)])
            self.play(mn.Create(skip), mn.FadeIn(skip_lab))
            fw = T('順伝播: f(x) は「差分の修正」だけを学習すればOKだよ', font_size=26).to_edge(mn.DOWN)
            self.play(mn.FadeIn(fw))
            dot = mn.Dot(color=mn.WHITE).move_to(xin)
            self.play(mn.MoveAlongPath(dot, mn.VMobject().set_points_as_corners([xin.get_center(), norm.get_center(), f.get_center(), plus.get_center()])), run_time=1.5)
            self.play(mn.FadeOut(dot))
            bw = T('逆伝播: 足し算の傾きは 1.0 → 近道を通って勾配がそのまま届くよ！', font_size=25, color=mn.RED).to_edge(mn.DOWN)
            self.play(mn.Transform(fw, bw))
            back = skip.copy().set_color(mn.RED).set_stroke(width=8)
            for _ in range(2):
                self.play(mn.ShowPassingFlash(back.copy().reverse_direction(), time_width=0.5), run_time=1.2)
            self.wait(1.5)
    return ResidualScene

show_anim(make_scene)
''')

md(r"""
## 9.6 最初のパラメーター: 正規分布 - Initial parameters: the normal distribution

0.1のコードでは，パラメーターの最初の値を `random.gauss(0, 0.08)` で決めていました。
これは，**平均0，標準偏差0.08の正規分布** にしたがう乱数です。正規分布は，平均のまわりに多く集まり，離れるほど少なくなる「つりがね型」の分布で，**平均 ± 標準偏差の範囲に約68%** が入ります。
9.4で見たように，最初の値の散らばりが大きすぎても小さすぎても学習がうまくいかないので，小さめの値からスタートするのです。
> In 0.1, the initial values of the parameters were set with `random.gauss(0, 0.08)`.
These are random numbers from **a normal distribution with mean 0 and standard deviation 0.08**. The normal distribution is "bell-shaped": many values gather around the mean and fewer appear farther away. **About 68% fall within mean ± standard deviation.**
As we saw in 9.4, training does not work well if the initial spread is too large or too small, so we start with smallish values.
""")

code(r'''
#@title [スライダー] 正規分布に従う乱数のヒストグラム { display-mode: "form" }
def plot_gauss(std=0.08):
    rng = random.Random(0)
    xs = [rng.gauss(0, std) for _ in range(5000)]
    inside = sum(abs(x) <= std for x in xs) / len(xs)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hist(xs, bins=60, range=(-0.5, 0.5), color='#3b82f6', edgecolor='#1d4ed8', alpha=0.8)
    ax.axvspan(-std, std, color='#f59e0b', alpha=0.25, label=f'平均±1σの範囲 (全体の {inside:.1%})')
    ax.set_title(f'random.gauss(0, {std:.2f}) による 5000個の乱数分布', fontsize=12, fontweight='bold')
    ax.set_xlabel('生成された値')
    ax.set_ylabel('度数')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()

interact(plot_gauss, std=widgets.FloatSlider(value=0.08, min=0.01, max=0.2, step=0.01, description='標準偏差:'));
''')

md(r"""
## 9.7 実装: GPTの完成 - Implementation: completing the GPT

ついに，0.1のGPTと同じモデルを組み立てます。これまでの部品を，次の順番につなぐだけです。
> Finally, we assemble the same model as the GPT in 0.1. We just connect the parts so far in this order:

1. 文字の埋め込み + 位置の埋め込み(第6章，第8章) → RMSNorm
2. **Attentionブロック**: RMSNorm → Attention(第8章) → 残差接続
3. **MLPブロック**: RMSNorm → MLP(第7章) → 残差接続
4. `lm_head` でスコアに(第6章)
> 1. Character embedding + position embedding (Chapters 6, 8) → RMSNorm
> 2. **Attention block**: RMSNorm → Attention (Chapter 8) → residual connection
> 3. **MLP block**: RMSNorm → MLP (Chapter 7) → residual connection
> 4. Scores with `lm_head` (Chapter 6)

2と3をまとめて1つの **層(レイヤー)** といいます。0.1では `n_layer = 1` なので1層ですが，この数を増やすと層を何重にも重ねられます。
> Steps 2 and 3 together are called one **layer**. In 0.1, `n_layer = 1`, so there is one layer, but increasing this number stacks more layers.
""")

code(r'''
n_layer = 1

def new_gpt_state(seed=0):
    random.seed(seed)
    state = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        state[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
        state[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
        state[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
        state[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
        state[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
        state[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
    return state

def make_gpt(state):
    def gpt_model(tokens, record=None):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        logits_list = []
        for pos_id, token in enumerate(tokens):
            x = [t + p for t, p in zip(state['wte'][token], state['wpe'][pos_id])]   # 1. 文字 + 位置埋め込み
            x = rmsnorm(x)
            for li in range(n_layer):
                x_residual = x                                                     # 2. Attentionブロック
                x = rmsnorm(x)
                x = attention(x, keys[li], values[li], state[f'layer{li}.attn_wq'], state[f'layer{li}.attn_wk'],
                              state[f'layer{li}.attn_wv'], state[f'layer{li}.attn_wo'], record)
                x = [a + b for a, b in zip(x, x_residual)]                         #    残差接続
                x_residual = x                                                     # 3. MLPブロック
                x = rmsnorm(x)
                x = linear(x, state[f'layer{li}.mlp_fc1'])
                x = [xi.relu() for xi in x]
                x = linear(x, state[f'layer{li}.mlp_fc2'])
                x = [a + b for a, b in zip(x, x_residual)]                         #    残差接続
            logits_list.append(linear(x, state['lm_head']))                        # 4. スコア
        return logits_list
    return gpt_model

gpt_sgd_state = new_gpt_state()
gpt_sgd = make_gpt(gpt_sgd_state)
print('パラメーター数:', len(all_params(gpt_sgd_state)))
''')

md(r"""
パラメーター数が0.1のGPTと同じになりました(0.1では `block_size = 6`)。
勾配降下法で学習させてみましょう。**2〜4分かかります。**
> The number of parameters is the same as the GPT in 0.1 (where `block_size = 6`).
Let's train it with gradient descent. **It takes 2–4 minutes.**
""")

code(r'''
gpt_sgd_history = train_sgd(gpt_sgd, all_params(gpt_sgd_state), num_steps=1000, learning_rate=0.3)
print(f'テストloss: {test_loss(gpt_sgd):.4f}  (数えるbigram: {BIGRAM_TEST_LOSS:.4f})')
print('作った名前:', ' '.join(generate(gpt_sgd, 12)))
plot_history({'Attentionモデル (第8章)': attn_history, 'GPT (勾配降下法)': gpt_sgd_history})
''')

md(r"""
これで，0.1のGPTと同じ仕組みのモデルができあがりました。
でも，まだ1つだけ違うところがあります。0.1では，勾配降下法ではなく **Adam** という方法でパラメーターを更新していました。次の章で，その理由を学びます。
> Now we have a model with the same mechanism as the GPT in 0.1.
But one thing is still different. In 0.1, the parameters were updated not with plain gradient descent but with a method called **Adam**. We learn why in the next chapter.

## 9.8 0.1のコードとの対応 - Matching with the code in 0.1

```python
def rmsnorm(x): ...                           # 9.3 RMSNorm関数だよ
x = rmsnorm(x)                                # 9.7 のステップ1
x_residual = x                                # 9.5 入力ベクトルを残差接続用にキープするよ
x = rmsnorm(x)                                # 9.3
... Attention または MLP ...
x = [a + b for a, b in zip(x, x_residual)]    # 9.5 残差接続で足し算合流するよ
matrix = lambda nout, nin, std=0.08: ...      # 9.6 正規分布によるパラメータ初期化だよ
```

## 練習問題 - Exercises

1. ベクトル $(3, 4)$ のRMSを求めましょう。RMSNormをかけるとどうなるでしょうか？
2. 残差接続 $y = x + f(x)$ で，$f(x)$ の傾きが0でも，$y$ の $x$ に対する傾きは0になりません。なぜでしょうか？

<details><summary>答え - Answers</summary>

1. $\sqrt{\frac{9 + 16}{2}} = \sqrt{12.5} \approx 3.54$ です。RMSNorm後は $(3, 4) \div 3.54 \approx (0.85, 1.13)$ で，RMSは1になります。
2. $y$ の傾きは「$x$ の傾き(1)」+「$f(x)$ の傾き」なので，$f(x)$ の傾きが0でも1が残るからです。
</details>
""")
