from nbbuild import md, code

md(r"""
---
# 第7章 ニューラルネットワーク: MLPとReLU - Chapter 7: Neural networks: MLP and ReLU

第6章のモデルは「埋め込み → 線形変換」だけでした。
この章では，線形変換と **ReLU** という簡単な関数を組み合わせて，もっと複雑な関係を表せる **MLP(多層パーセプトロン)** を作ります。
0.1のコードの `# 2) MLP block` の部分です。
> The model in Chapter 6 was just "embedding → linear transformation".
In this chapter, we combine linear transformations with a simple function called **ReLU** to build an **MLP (multi-layer perceptron)**, which can express more complex relationships.
This is the `# 2) MLP block` part of the code in 0.1.

**この章で使う数学**: 一次関数，関数の合成，グラフ(数学I)
> **Math in this chapter**: linear functions, composition of functions, graphs (Math I)

## 7.1 線形変換を重ねても線形変換 - Stacking linear transformations gives a linear transformation

線形変換を2回重ねれば，もっと複雑なことができそうです。でも，実はそうはなりません。
1つの数で考えると，一次関数 $y = 2x + 1$ に，さらに一次関数 $z = 3y - 4$ を重ねても
> Stacking two linear transformations seems like it could do more complex things. But in fact it does not.
Think with single numbers: if we apply $z = 3y - 4$ after $y = 2x + 1$,

$$z = 3(2x + 1) - 4 = 6x - 1$$

となり，結局ただの一次関数です。**直線をいくつ重ねても直線** なのです。ベクトルと行列でも同じことが言えます。
> it is still just a linear function. **No matter how many straight lines you stack, you get a straight line.** The same is true for vectors and matrices.

曲がったグラフを表すには，**直線ではない関数** を間にはさむ必要があります。
> To express curved graphs, we need to insert **a function that is not a straight line** in between.

## 7.2 ReLU: 折れ線の関数 - ReLU: a bent-line function

**ReLU(Rectified Linear Unit)** は，一番シンプルな「直線ではない関数」です。
> **ReLU (Rectified Linear Unit)** is the simplest "function that is not a straight line":

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & (x > 0) \\ 0 & (x \le 0) \end{cases}$$

マイナスは0にして，プラスはそのまま通す，というだけです。グラフは原点で折れ曲がった折れ線になります。
傾きは，$x > 0$ なら1，$x < 0$ なら0です(第5章の `Value.relu` の局所的な傾き `float(self.data > 0)` はこれです)。
> It just turns negatives into 0 and lets positives through. The graph is a line bent at the origin.
The slope is 1 when $x > 0$ and 0 when $x < 0$ (this is the local slope `float(self.data > 0)` in `Value.relu` from Chapter 5).
""")

code(r'''
#@title ReLU関数のグラフ { display-mode: "form" }
xs = [i / 10 for i in range(-30, 31)]
ys = [max(0, x) for x in xs]
fig, ax = plt.subplots(figsize=(5.5, 3.4))
ax.plot(xs, ys, lw=2.8, color='#4f46e5', label='y = ReLU(x) = max(0, x)')
ax.axhline(0, color='#64748b', lw=1.0)
ax.axvline(0, color='#64748b', lw=1.0)
ax.plot([0], [0], 'o', color='#ef4444', ms=6)
ax.set_title('ReLU 関数の形 (原点で折れ曲がるよ)', fontsize=12, fontweight='bold')
ax.set_xlabel('x', fontsize=11); ax.set_ylabel('y', fontsize=11)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.show()
''')

md(r"""
## 7.3 折れ線を足し合わせると，どんな形でも作れる - Adding bent lines can make any shape

ReLUをずらしたり，傾きを変えたりしたものをいくつも足し合わせると，**いろいろな形の折れ線** が作れます。
折れ曲がる点を増やしていけば，曲線にもどんどん近づけられます。
> By shifting ReLUs, changing their slopes and adding many of them together, we can make **bent lines of many shapes**.
With more bending points, we can get closer and closer to any curve.

### [アニメーション] ReLUを足して放物線を作る - [Animation] Making a parabola by adding ReLUs
""")

code(r'''
#@title アニメーション: ReLUの足し合わせ { display-mode: "form" }
def make_scene(mn, T):
    class ReluSum(mn.Scene):
        def construct(self):
            relu = lambda x: max(0.0, x)
            ax = mn.Axes(x_range=[-2.5, 2.5, 1], y_range=[-0.5, 5, 1], x_length=7, y_length=5, tips=False).shift(mn.LEFT * 2 + mn.DOWN * 0.3)
            target = ax.plot(lambda x: x**2, x_range=[-2.2, 2.2], color=mn.GRAY, stroke_width=3).set_stroke(opacity=0.6)
            title = T('ReLU を足し合わせて y = x² に近づける', font_size=32).to_edge(mn.UP)
            self.play(mn.Write(title), mn.Create(ax), mn.Create(target))
            terms = [('4 − 3(x+2)', lambda x: 4 - 3 * (x + 2)),
                     ('+ 2·ReLU(x+1)', lambda x: 2 * relu(x + 1)),
                     ('+ 2·ReLU(x)', lambda x: 2 * relu(x)),
                     ('+ 2·ReLU(x−1)', lambda x: 2 * relu(x - 1))]
            fs, labels = [], mn.VGroup()
            graph = None
            for i, (text, fn) in enumerate(terms):
                fs.append(fn)
                cur = list(fs)
                g = ax.plot(lambda x: sum(f(x) for f in cur), x_range=[-2.2, 2.2], color=mn.YELLOW, stroke_width=5)
                lab = T(text, font_size=28, color=mn.YELLOW if i == 0 else mn.BLUE)
                labels.add(lab); labels.arrange(mn.DOWN, aligned_edge=mn.LEFT).to_edge(mn.RIGHT).shift(mn.LEFT * 0.3)
                if graph is None:
                    graph = g
                    self.play(mn.Create(graph), mn.FadeIn(lab))
                else:
                    self.play(mn.Transform(graph, g), mn.FadeIn(lab), run_time=1.2)
                self.wait(0.4)
            note = T('折れ曲がる点を増やすほど，曲線に近づくよ', font_size=26).to_edge(mn.DOWN)
            self.play(mn.Write(note))
            self.wait(1.5)
    return ReluSum

show_anim(make_scene)
''')

md(r"""
### [スライダー] ReLUを組み合わせて，グレーの曲線に合わせよう - [Slider] Combine ReLUs to match the gray curve

3つのReLUの **傾き $w$** と **折れ曲がる位置 $k$**，全体の **高さ $b$** を動かして，黄色の折れ線をグレーの曲線にできるだけ近づけてみましょう。
右上の「ずれ」(平均二乗誤差: ずれの2乗の平均)が小さいほど上手です。0.02を切れたら上級者！
> Move the **slope $w$** and **bending position $k$** of the three ReLUs and the overall **height $b$** to make the yellow line as close as possible to the gray curve.
The smaller the "error" (mean squared error: the average of the squared differences) at the top right, the better. Below 0.02 and you are an expert!

$$y = b + w_1\,\text{ReLU}(x - k_1) + w_2\,\text{ReLU}(x - k_2) + w_3\,\text{ReLU}(x - k_3)$$

> 実は，この「ずれを小さくするように $w$，$k$，$b$ を調整する」作業こそ，コンピューターが勾配降下法で自動でやっていることです。
> In fact, this task of "adjusting $w$, $k$ and $b$ to make the error small" is exactly what the computer does automatically with gradient descent.
""")

code(r'''
#@title [スライダー] ReLUフィッティングゲーム { display-mode: "form" }
def relu_game(b=0.0, w1=0.5, k1=-1.0, w2=0.5, k2=0.0, w3=0.5, k3=1.0):
    target = lambda x: max(0.0, x) ** 2 / 2
    xs = [i / 20 for i in range(-60, 61)]
    ys = [b + w1 * max(0, x - k1) + w2 * max(0, x - k2) + w3 * max(0, x - k3) for x in xs]
    err = sum((y - target(x)) ** 2 for x, y in zip(xs, ys)) / len(xs)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, [target(x) for x in xs], color='#94a3b8', lw=7, alpha=0.5, label='目標の曲線')
    ax.plot(xs, ys, color='#f59e0b', lw=2.5, label='3つのReLUの合成')
    for k in (k1, k2, k3): ax.axvline(k, color='#3b82f6', ls=':', alpha=0.6)
    ax.set_xlim(-3, 3); ax.set_ylim(-1, 5)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    badge = '  (素晴らしい！クリア！)' if err < 0.02 else ''
    ax.set_title(f'目標とのずれ(平均二乗誤差) = {err:.4f}{badge}', fontsize=12, fontweight='bold',
                 color='#15803d' if err < 0.02 else '#1e293b')
    plt.tight_layout()
    plt.show()

w = lambda v, d: widgets.FloatSlider(value=v, min=-3, max=3, step=0.05, description=d)
interact(relu_game, b=w(0.0, '底上げ b:'), w1=w(0.5, '傾き w1:'), k1=w(-1.0, '位置 k1:'),
         w2=w(0.5, '傾き w2:'), k2=w(0.0, '位置 k2:'), w3=w(0.5, '傾き w3:'), k3=w(1.0, '位置 k3:'));
''')

md(r"""
<details><summary>ヒント - Hint</summary>

目標の曲線は，$x < 0$ では0，$x > 0$ ではだんだん急になっていきます。
$k_1 = 0$ から始めて，右に行くほど傾きが増えるように，$k_2$，$k_3$ で傾きを足していきましょう。
例えば $b = 0,\ w_1 = 0.5,\ k_1 = 0.1,\ w_2 = 1,\ k_2 = 1,\ w_3 = 1,\ k_3 = 2$ あたりから調整してみてください。
> The target is 0 for $x < 0$ and gets steeper for $x > 0$.
Start from $k_1 = 0$ and add slope with $k_2$ and $k_3$ so the slope increases to the right.
</details>

## 7.4 MLP: 広げて，ReLU，縮める - MLP: expand, ReLU, shrink

0.1のコードのMLPは，次の3ステップです。
> The MLP in 0.1 has these 3 steps:

1. 線形変換 `mlp_fc1` で，16成分のベクトルを **64成分に広げる**(64行の行列)
2. 64個それぞれに **ReLU** をかける
3. 線形変換 `mlp_fc2` で，**16成分に縮める**
> 1. The linear transformation `mlp_fc1` **expands** the 16-component vector **to 64 components** (a 64-row matrix).
> 2. Apply **ReLU** to each of the 64.
> 3. The linear transformation `mlp_fc2` **shrinks it back to 16 components**.

`mlp_fc1` の64個の行は，それぞれが「ある特徴を調べる物差し」です(第6章)。
内積がプラスなら(その特徴が当てはまるなら)ReLUを通り，マイナスなら0になります。
つまり64個の「特徴を見つけたら反応するセンサー」のようなもので，その反応を組み合わせて次の計算に渡すのです。
> Each of the 64 rows of `mlp_fc1` is "a ruler that checks for a certain feature" (Chapter 6).
If the dot product is positive (the feature applies), it passes through ReLU; if negative, it becomes 0.
So it is like 64 "sensors that react when they find a feature", and their reactions are combined and passed to the next step.

## 7.5 実装: MLPを加えたモデル - Implementation: a model with an MLP

第6章のモデルにMLPを加えて，`n_embd = 16` で学習させてみます。**1〜2分かかります。**
> Let's add an MLP to the model from Chapter 6 and train it with `n_embd = 16`. **It takes 1–2 minutes.**
""")

code(r'''
random.seed(0)
n_embd = 16
mlp_state = {'wte': matrix(vocab_size, n_embd),
             'mlp_fc1': matrix(4 * n_embd, n_embd),   # 16次元 → 64次元に広げるよ
             'mlp_fc2': matrix(n_embd, 4 * n_embd),   # 64次元 → 16次元に縮めるよ
             'lm_head': matrix(vocab_size, n_embd)}
mlp_params = all_params(mlp_state)
print('パラメーター数:', len(mlp_params))

def mlp_model(tokens):
    logits_list = []
    for token in tokens:
        x = mlp_state['wte'][token]
        x = linear(x, mlp_state['mlp_fc1'])   # 1. 広げる
        x = [xi.relu() for xi in x]           # 2. ReLU
        x = linear(x, mlp_state['mlp_fc2'])   # 3. 縮める
        logits_list.append(linear(x, mlp_state['lm_head']))
    return logits_list

mlp_history = train_sgd(mlp_model, mlp_params, num_steps=1000, learning_rate=0.5)
print(f'テストloss: {test_loss(mlp_model):.4f}  (数えるbigram: {BIGRAM_TEST_LOSS:.4f})')
print('作った名前:', ' '.join(generate(mlp_model, 12)))
plot_history({'埋め込みモデル (n_embd=2)': emb_history, 'MLPモデル (n_embd=16)': mlp_history})
''')

md(r"""
作った名前はまだ1文字のものが多く，テストlossも数えるbigramモデルに届いていません。
どんなに複雑な計算をしても，**入力が直前の1文字だけ** なら，第2章の「数える」方法以上のことはほとんどできないのです。
> Many of the names it makes are still one character long, and the test loss has not reached the counting bigram model.
No matter how complex the calculation, if **the only input is the one previous character**, it can hardly do better than the "counting" method in Chapter 2.

> **学習率について**: この学習では学習率を0.5にしています。試しに2や3にすると，途中でlossが爆発して `math domain error` というエラーで止まってしまいます(確率が0になって $\log 0$ を計算しようとするため)。
> この「学習が不安定になる」問題は，第9章と第10章で解決します。
> **About the learning rate**: We use a learning rate of 0.5 here. If you try 2 or 3, the loss explodes partway and training stops with a `math domain error` (the probability becomes 0 and it tries to calculate $\log 0$).
> We will solve this "unstable training" problem in Chapters 9 and 10.

いよいよ次の章で，**前の文字をすべて見る** 仕組み，Attentionを作ります。
> In the next chapter, we finally build Attention, the mechanism that **looks at all the previous characters**.

## 7.6 0.1のコードとの対応 - Matching with the code in 0.1

```python
# 2) MLP block
x_residual = x                                  # 第9章で解説するよ
x = rmsnorm(x)                                  # 第9章で解説するよ
x = linear(x, state_dict[f'layer{li}.mlp_fc1'])  # 7.4 広げるよ
x = [xi.relu() for xi in x]                     # 7.2 ReLUだよ
x = linear(x, state_dict[f'layer{li}.mlp_fc2'])  # 7.4 縮めるよ
x = [a + b for a, b in zip(x, x_residual)]      # 第9章で解説するよ
```

## 練習問題 - Exercises

1. $\text{ReLU}(3)$，$\text{ReLU}(-2)$，$\text{ReLU}(0)$ の値はいくつでしょうか？
2. $y = \text{ReLU}(x) - \text{ReLU}(x - 1)$ のグラフを描いてみましょう(スライダーで $w_1 = 1, k_1 = 0, w_2 = -1, k_2 = 1, w_3 = 0$ にすると確かめられます)。
3. `mlp_model` のパラメーター数が4352個になることを確かめてみましょう。

<details><summary>答え - Answers</summary>

1. 3，0，0 です。
2. $x \le 0$ で0，$0 \le x \le 1$ で $y = x$，$x \ge 1$ で1になる「階段のような」折れ線です。
3. `wte` $72 \times 16 = 1152$，`mlp_fc1` $64 \times 16 = 1024$，`mlp_fc2` $16 \times 64 = 1024$，`lm_head` $72 \times 16 = 1152$ です。合計4352個になります。
</details>
""")
