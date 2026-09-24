from nbbuild import md, code

md(r"""
---
# 第10章 賢い下り方: Adam - Chapter 10: A smarter way down: Adam

第4章の勾配降下法は「傾きの逆向きに，学習率の分だけ進む」というシンプルな方法でした。
0.1のコードでは，これを改良した **Adam(アダム)** という方法が使われています。今では，ほとんどのLLMの学習にAdamやその仲間が使われています。
この章では，数学Iの平均の考え方を使って，Adamの仕組みを学びます。
> Gradient descent in Chapter 4 was a simple method: "move against the slope by the learning rate".
The code in 0.1 uses an improved method called **Adam**. Today, Adam and its relatives are used to train almost all LLMs.
In this chapter, we learn how Adam works using the idea of averages from Math I.

**この章で使う数学**: 平均，移動平均(数学I「データの分析」の発展)，平方根
> **Math in this chapter**: averages, moving averages (an extension of Math I), square roots

## 10.1 勾配降下法の2つの弱点 - Two weaknesses of gradient descent

1. **傾きがばらつく**: 0.1のコードでは，1ステップごとに1つの名前だけを使ってlossを計算します。名前によって傾きが大きく変わるので，進む方向が毎回ふらふらします。
2. **方向によって坂の急さが違う**: 第4章の等高線のスライダーで見たように，急な方向に合わせて学習率を小さくすると，ゆるやかな方向ではなかなか進みません。
> 1. **The slopes are noisy**: In 0.1, each step computes the loss with just one name. The slope changes a lot from name to name, so the direction wobbles every step.
> 2. **Steepness differs by direction**: As we saw with the contour slider in Chapter 4, if we lower the learning rate to suit the steep direction, we barely move in the gentle direction.

Adamは，この2つを **移動平均** という考え方で解決します。
> Adam solves both with the idea of **moving averages**.

## 10.2 移動平均: 最近の値の平均 - Moving averages: the average of recent values

株価のグラフでよく見る「5日移動平均線」は，毎日「直近5日分の平均」を計算してつないだ線です。細かいギザギザがならされて，全体の流れが見やすくなります。
> The "5-day moving average line" often seen on stock charts connects "the average of the last 5 days" calculated every day. It smooths out small zigzags and makes the overall trend easier to see.

Adamでは，もっと簡単に計算できる **指数移動平均** を使います。新しい値 $g$ が来るたびに，次のように更新します。
> Adam uses an **exponential moving average**, which is even easier to calculate. Every time a new value $g$ comes in, we update:

$$m \leftarrow \beta\, m + (1 - \beta)\, g$$

「今までの平均を $\beta$ の割合だけ残して，新しい値を $1 - \beta$ の割合だけ混ぜる」という意味です。
例えば $\beta = 0.9$ なら，前の平均を90%，新しい値を10%混ぜます。古い値ほど $0.9,\ 0.9^2,\ 0.9^3, \ldots$ と重みが指数関数的に小さくなるので，「指数」移動平均といいます。
> It means "keep a fraction $\beta$ of the average so far and mix in a fraction $1 - \beta$ of the new value".
For example, with $\beta = 0.9$, we keep 90% of the previous average and mix in 10% of the new value. Older values get weights $0.9,\ 0.9^2,\ 0.9^3, \ldots$ that shrink exponentially, which is why it is called an "exponential" moving average.

### [スライダー] βを変えて移動平均を見よう - [Slider] Change β and look at the moving average
""")

code(r'''
#@title [スライダー] 指数移動平均 (EMA) による平滑化 { display-mode: "form" }
def plot_ema(beta=0.9):
    rng = random.Random(3)
    g = [math.sin(t / 15) + rng.gauss(0, 0.6) for t in range(150)]
    m, ema = 0.0, []
    for x in g:
        m = beta * m + (1 - beta) * x
        ema.append(m)
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.plot(g, '.', color='#94a3b8', alpha=0.7, ms=6, label='毎回の値(ばらつきが大きい)')
    ax.plot(ema, color='#ef4444', lw=2.5, label=f'指数移動平均 (β = {beta:.2f})')
    ax.set_ylim(-3, 3)
    ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    ax.set_title(f'β = {beta:.2f} による移動平均', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_ema, beta=widgets.FloatSlider(value=0.9, min=0.0, max=0.99, step=0.01, description='減衰率 β:'));
''')

md(r"""
$\beta$ を大きくするほど線はなめらかになりますが，変化に追いつくのが遅くなります。$\beta = 0$ なら，毎回の値そのものです。
> The larger $\beta$, the smoother the line, but the slower it follows changes. With $\beta = 0$, it is just each value itself.

### [アニメーション] ばらつく値を移動平均でならす - [Animation] Smoothing noisy values with a moving average
""")

code(r'''
#@title アニメーション: 指数移動平均 { display-mode: "form" }
def make_scene(mn, T):
    class EMAScene(mn.Scene):
        def construct(self):
            rng = random.Random(3)
            g = [math.sin(t / 6) + rng.gauss(0, 0.6) for t in range(60)]
            ax = mn.Axes(x_range=[0, 60, 10], y_range=[-2.5, 2.5, 1], x_length=10, y_length=5, tips=False).shift(mn.DOWN * 0.4)
            title = T('m ← 0.8 m + 0.2 g', font_size=34).to_edge(mn.UP)
            self.play(mn.Create(ax), mn.Write(title))
            m, prev = 0.0, None
            dots, lines = mn.VGroup(), mn.VGroup()
            for t, x in enumerate(g):
                d = mn.Dot(ax.c2p(t, x), radius=0.05, color=mn.GRAY)
                new_m = 0.8 * m + 0.2 * x
                seg = mn.Line(ax.c2p(t - 1, m), ax.c2p(t, new_m), color=mn.RED, stroke_width=5) if t > 0 else None
                anims = [mn.FadeIn(d, scale=2)]
                if seg: anims.append(mn.Create(seg))
                self.play(*anims, run_time=0.08 if t > 8 else 0.35)
                m = new_m
            lab = mn.VGroup(T('灰色: 毎回の傾き(ばらばら)', font_size=24, color=mn.GRAY),
                             T('赤: 移動平均(なめらか)', font_size=24, color=mn.RED)).arrange(mn.DOWN, aligned_edge=mn.LEFT).to_corner(mn.DR)
            self.play(mn.FadeIn(lab))
            self.wait(1.5)
    return EMAScene

show_anim(make_scene)
''')

md(r"""
## 10.3 1つ目の工夫 m: 傾きの移動平均(勢い) - Trick 1, m: the moving average of the slope (momentum)

Adamは，毎回の傾き $g$ をそのまま使わずに，**傾きの移動平均 $m$** を使って進みます。
> Adam does not use each slope $g$ directly; it moves using **the moving average of the slopes, $m$**:

$$m \leftarrow \beta_1\, m + (1 - \beta_1)\, g$$

ばらつきがならされるので，進む方向が安定します。
また，ジグザグする方向ではプラスとマイナスの傾きが打ち消し合い，いつも同じ向きの方向では傾きが積み重なって，**坂を転がるボールのように勢いがつきます**。
> Because the noise is smoothed out, the direction becomes stable.
Also, in a direction that zigzags, positive and negative slopes cancel out, while in a direction that always has the same sign, the slopes add up, and **it gains momentum like a ball rolling down a hill**.

## 10.4 2つ目の工夫 v: 傾きの2乗の移動平均(歩幅をそろえる) - Trick 2, v: the moving average of the squared slope (equalizing step sizes)

もう1つ，**傾きの2乗の移動平均 $v$** も記録します。
> Adam also records **the moving average of the squared slopes, $v$**:

$$v \leftarrow \beta_2\, v + (1 - \beta_2)\, g^2$$

$\sqrt{v}$ は，第9章のRMSと同じ「2乗して平均して平方根」なので，**そのパラメーターの傾きがふだんどれくらいの大きさか** を表します。
そして，$m$ を $\sqrt{v}$ で割って進みます。
> $\sqrt{v}$ is "square, average, square root", just like RMS in Chapter 9, so it shows **how large the slope of that parameter usually is**.
Then Adam moves by $m$ divided by $\sqrt{v}$:

$$\theta \leftarrow \theta - \eta \, \frac{m}{\sqrt{v} + \varepsilon}$$

($\theta$ はパラメーター，$\varepsilon$ は0で割らないための小さな数です)
> ($\theta$ is a parameter, and $\varepsilon$ is a tiny number to avoid dividing by 0.)

急な方向(傾きがいつも大きい)では大きな数で割るので歩幅が小さくなり，ゆるやかな方向では小さな数で割るので歩幅が大きくなります。
その結果，**どの方向にも，だいたい学習率 $\eta$ くらいの歩幅で進む** ようになります。第4章の「方向によって坂の急さが違う」問題が解決できるのです。
> In steep directions (where slopes are always large) we divide by a large number, so the step gets smaller; in gentle directions we divide by a small number, so the step gets larger.
As a result, **it moves with a step of roughly the learning rate $\eta$ in every direction**. This solves the "steepness differs by direction" problem from Chapter 4.

## 10.5 最初のずれを直す ★ - Correcting the start ★

$m$ と $v$ は0からスタートするので，最初の何ステップかは，本当の平均よりずっと小さな値になってしまいます。
例えば $\beta_1 = 0.9$ で，傾きがずっと1だったとしても，1ステップ目の $m$ は $0.9 \times 0 + 0.1 \times 1 = 0.1$ です。
そこで，$t$ ステップ目では $1 - \beta^t$ で割って補正します(**バイアス補正**)。1ステップ目なら $0.1 \div (1 - 0.9) = 1$ となり，正しい平均になります。
> Since $m$ and $v$ start from 0, for the first few steps they are much smaller than the true average.
For example, with $\beta_1 = 0.9$, even if the slope is always 1, $m$ after step 1 is $0.9 \times 0 + 0.1 \times 1 = 0.1$.
So at step $t$, we divide by $1 - \beta^t$ to correct it (**bias correction**). At step 1, $0.1 \div (1 - 0.9) = 1$, the correct average.

$$\hat{m} = \frac{m}{1 - \beta_1^{\,t}}, \qquad \hat{v} = \frac{v}{1 - \beta_2^{\,t}}$$

$t$ が大きくなると $\beta^t$ は0に近づくので(第3章の指数関数のグラフ！)，補正はだんだん効かなくなります。
> As $t$ grows, $\beta^t$ approaches 0 (the exponential graph from Chapter 3!), so the correction fades away.
""")

code(r'''
beta1 = 0.9
m = 0.0
for t in range(1, 6):
    m = beta1 * m + (1 - beta1) * 1.0   # 傾きがずっと1だったとするよ
    print(f'step {t:2d}: m = {m:.4f},  補正後 m_hat = {m / (1 - beta1**t):.4f}')
''')

md(r"""
## 10.6 勾配降下法とAdamを比べよう - Compare gradient descent and Adam

### [スライダー] 等高線の上で比べる - [Slider] Compare on a contour map

第4章と同じ関数 $f(x, y) = (x - 1)^2 + 2(y + 2)^2$ で，勾配降下法(赤)とAdam(青)の進み方を比べます。
さらに「方向による急さの違い」を大きくするスライダーも付けました($y$ の方向の係数)。急さの差を大きくしても，Adamはまっすぐ谷底に向かえるでしょうか？
> Using the same function as Chapter 4, $f(x, y) = (x - 1)^2 + 2(y + 2)^2$, we compare how gradient descent (red) and Adam (blue) move.
There is also a slider to increase "the difference in steepness by direction" (the coefficient in the $y$ direction). Even with a large difference, can Adam head straight for the bottom?
""")

code(r'''
#@title [スライダー] 勾配降下法 vs Adam { display-mode: "form" }
import numpy as np
def plot_optimizers(learning_rate=0.1, steepness=2.0, steps=40):
    fxy = lambda x, y: (x - 1)**2 + steepness * (y + 2)**2
    grad = lambda x, y: (2 * (x - 1), 2 * steepness * (y + 2))
    x, y = -3.0, 1.5; sgd = [(x, y)]
    for _ in range(steps):
        gx, gy = grad(x, y); x, y = x - learning_rate * gx, y - learning_rate * gy
        sgd.append((x, y))
    p, m, v = [-3.0, 1.5], [0.0, 0.0], [0.0, 0.0]; adam = [tuple(p)]
    for t in range(1, steps + 1):
        g = grad(*p)
        for i in range(2):
            m[i] = 0.9 * m[i] + 0.1 * g[i]
            v[i] = 0.999 * v[i] + 0.001 * g[i]**2
            m_hat = m[i] / (1 - 0.9**t)
            v_hat = v[i] / (1 - 0.999**t)
            p[i] -= learning_rate * 2.5 * m_hat / (v_hat**0.5 + 1e-8)
        adam.append(tuple(p))
    X, Y = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-5, 3, 200))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.contour(X, Y, fxy(X, Y), levels=25, cmap='viridis', alpha=0.5)
    ok = lambda path: [q for q in path if abs(q[0]) < 8 and abs(q[1]) < 8]
    ax.plot(*zip(*ok(sgd)), 'o-', color='#ef4444', ms=3.5, lw=1.5, label=f'勾配降下法 (f = {fxy(*sgd[-1]):.3g})')
    ax.plot(*zip(*ok(adam)), 'o-', color='#3b82f6', ms=3.5, lw=1.8, label=f'Adam (f = {fxy(*adam[-1]):.3g})')
    ax.plot([1], [-2], 'x', color='#0f172a', ms=12, mew=3, label='谷底')
    ax.set_xlim(-4, 4); ax.set_ylim(-5, 3); ax.set_aspect('equal'); ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f'最適化対決 (急さの差: {steepness:.1f}倍)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_optimizers,
         learning_rate=widgets.FloatSlider(value=0.1, min=0.01, max=0.5, step=0.01, description='学習率 η:'),
         steepness=widgets.FloatSlider(value=2.0, min=1.0, max=20.0, step=0.5, description='yの急さ:'),
         steps=widgets.IntSlider(value=40, min=1, max=100, step=5, description='ステップ数:'));
''')

md(r"""
「yの急さ」を大きくすると，勾配降下法は学習率を小さくしないと発散してしまい，小さくすると $x$ の方向になかなか進みません。
Adamは急さに関係なく，どちらの方向にも同じくらいの歩幅で進めます。
(このスライダーではAdamの歩幅を見やすくするため，学習率を2.5倍にしています)
> When you increase "steepness in y", gradient descent diverges unless you lower the learning rate, and with a low learning rate it barely moves in the $x$ direction.
Adam moves with similar steps in both directions, regardless of steepness.
(In this slider, Adam's learning rate is multiplied by 2.5 to make its steps easier to see.)

## 10.7 実装: AdamでGPTを学習する - Implementation: training the GPT with Adam

0.1のコードと同じAdamを実装して，第9章と同じGPTを学習させます。**2〜4分かかります。**
> Let's implement Adam exactly as in 0.1 and train the same GPT as in Chapter 9. **It takes 2–4 minutes.**
""")

code(r'''
def train_adam(model, params, num_steps=1000, learning_rate=0.01, beta1=0.85, beta2=0.99, eps_adam=1e-8):
    m = [0.0] * len(params)   # 10.3 傾きの移動平均だよ
    v = [0.0] * len(params)   # 10.4 傾きの2乗の移動平均だよ
    history = []
    for step in range(num_steps):
        loss = name_loss(model, docs[step % len(docs)])
        loss.backward()
        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))   # 10.5 バイアス補正だよ
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0
        history.append(loss.data)
        if (step + 1) % 100 == 0:
            print(f'step {step + 1:4d} / {num_steps} | 直近100回の平均loss {sum(history[-100:]) / 100:.4f}', end='\r')
    print()
    return history

gpt_adam_state = new_gpt_state()   # 第9章と同じ最初の値からスタートするよ
gpt_adam = make_gpt(gpt_adam_state)
gpt_adam_history = train_adam(gpt_adam, all_params(gpt_adam_state))
print(f'テストloss: {test_loss(gpt_adam):.4f}  (勾配降下法: {test_loss(gpt_sgd):.4f} / 数えるbigram: {BIGRAM_TEST_LOSS:.4f})')
print('作った名前:', ' '.join(generate(gpt_adam, 12)))
plot_history({'GPT (勾配降下法)': gpt_sgd_history, 'GPT (Adam)': gpt_adam_history})
''')

md(r"""
今回の小さなモデルでは差はわずかですが，Adamのほうがテストlossが少し低くなりました。
Adamの本当のよさは，**学習率の選び方にあまり悩まなくていい** ことです。勾配降下法では，学習率を大きくしすぎると発散し(第7章)，小さすぎるとなかなか進まないので，モデルごとにちょうどいい値を探す必要がありました(このNotebookでも2.0，0.5，0.3と使い分けています)。
Adamは歩幅をパラメーターごとに自動で調整するので，0.01のような決まった値で，いろいろなモデルをうまく学習できます。パラメーターが何十億個もある本物のLLMでは，この差がとても大きくなります。
> With this small model the difference is small, but Adam got a slightly lower test loss.
Adam's real strength is that **you do not have to worry much about choosing the learning rate**. With gradient descent, a learning rate that is too large diverges (Chapter 7) and one that is too small barely moves, so we had to search for a good value for each model (this Notebook uses 2.0, 0.5 and 0.3).
Adam adjusts the step size for each parameter automatically, so a fixed value like 0.01 trains many kinds of models well. In real LLMs with billions of parameters, this difference becomes very large.

これで，**0.1のGPTと完全に同じもの** ができあがりました！
> Now we have built **exactly the same thing as the GPT in 0.1**!

## 10.8 0.1のコードとの対応 - Matching with the code in 0.1

```python
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)                                # 10.3 傾きの移動平均バッファだよ
v = [0.0] * len(params)                                # 10.4 傾き2乗の移動平均バッファだよ
m[i] = beta1 * m[i] + (1 - beta1) * p.grad             # 10.3 傾きの移動平均を更新するよ
v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2        # 10.4 傾きの2乗の移動平均を更新するよ
m_hat = m[i] / (1 - beta1 ** (step + 1))               # 10.5 バイアス補正だよ
v_hat = v[i] / (1 - beta2 ** (step + 1))
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)     # 10.4 歩幅をそろえて進むよ
```

## 練習問題 - Exercises

1. $\beta = 0.5$ で，値が $4,\ 8$ の順に来たとき，指数移動平均 $m$ はいくつになるでしょうか？($m$ は0から始めます)
2. あるパラメーターの傾きがずっと $0.001$ だったとき，勾配降下法とAdamでは1ステップの歩幅はどう違うでしょうか？(学習率は $0.01$ とします)

<details><summary>答え - Answers</summary>

1. 1回目 $0.5 \times 0 + 0.5 \times 4 = 2$，2回目 $0.5 \times 2 + 0.5 \times 8 = 5$ です。
2. 勾配降下法は $0.01 \times 0.001 = 0.00001$ しか進みません。Adamは $m \approx 0.001$，$\sqrt{v} \approx 0.001$ なので，$0.01 \times \frac{0.001}{0.001} = 0.01$ 進みます。傾きが小さくても，ちゃんと学習率くらいの歩幅で進めます。
</details>
""")
