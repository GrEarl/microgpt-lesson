from nbbuild import md, code

md(r"""
---
# 第5章 微分を自動で: 計算グラフと連鎖律 ★ - Chapter 5: Automatic derivatives: computation graphs and the chain rule ★

前の章の最後で，数値微分では「パラメーターの数だけlossを計算し直す」必要があり，遅すぎることがわかりました。
この章では，**1回の計算で，すべてのパラメーターの傾きを一気に求める方法**(**誤差逆伝播法**，backpropagation)を学びます。
0.1のコードの `class Value` の部分です。
> At the end of the last chapter, we saw that numerical differentiation needs "as many loss calculations as there are parameters", which is too slow.
In this chapter, we learn **a way to find the slopes of all the parameters at once in one pass** (**backpropagation**).
This is the `class Value` part of the code in 0.1.

> ★ この章は少し難しいです。「lossから逆向きにたどると，全部の傾きが一気にわかる」ということだけつかめれば，先に進んでも大丈夫です。
> ★ This chapter is a bit hard. If you get the idea "going backward from the loss gives all the slopes at once", you can move on.

**この章で使う数学**: 微分(第4章)，合成関数の微分(連鎖律，数学IIIの内容を直感的に)
> **Math in this chapter**: derivatives (Chapter 4), the derivative of composite functions (the chain rule, a Math III topic, explained intuitively)

## 5.1 計算を図にする: 計算グラフ - Drawing a calculation: computation graphs

次の計算を考えます。
> Consider this calculation:

$$L = (a \times b + c)^2$$

これを，1回の計算(掛け算，足し算，2乗)ごとに分けて，矢印でつなぐと次のようになります。このような図を **計算グラフ** といいます。
> If we split it into single operations (multiply, add, square) and connect them with arrows, we get the figure below. Such a figure is called a **computation graph**.

```
a ──┐
    ×── d = a×b ──┐
b ──┘             +── e = d + c ── ^2 ── L = e²
c ────────────────┘
```

$a = 2,\ b = -3,\ c = 10$ なら，$d = -6$，$e = 4$，$L = 16$ です。左から右に計算していくことを **順伝播**(forward pass)といいます。
> If $a = 2,\ b = -3,\ c = 10$, then $d = -6$, $e = 4$, $L = 16$. Calculating from left to right is called the **forward pass**.

知りたいのは，「$a$ を少し増やすと $L$ はどれくらい変わるか」，つまり $L$ を $a$ で微分した傾きです($b$，$c$ についても同じです)。
> What we want to know is "how much $L$ changes when we increase $a$ a little", that is, the slope of $L$ with respect to $a$ (and the same for $b$ and $c$).

## 5.2 1つ1つの計算の傾き - The slope of each single operation

まず，1回の計算だけに注目したときの傾き(**局所的な傾き**)を考えます。
> First, think about the slope of just one operation (the **local slope**).

| 計算 | $x$ を少し増やすと… | 局所的な傾き |
|---|---|---|
| $x + y$ | 増やした分だけ増える | **$1$** |
| $x \times y$ | 増やした分の $y$ 倍だけ増える | **$y$** |
| $x^n$ | 第4章の公式 | **$n x^{n-1}$** |
| $e^x$ | 第4章の公式 | **$e^x$** |
| $\log x$ | 第4章の公式 | **$\dfrac{1}{x}$** |

例えば $x \times y$ で $x$ を $h$ だけ増やすと $(x + h) \times y = xy + h y$ なので，$h$ の $y$ 倍だけ増えます。だから傾きは $y$ です。
> For example, if we increase $x$ by $h$ in $x \times y$, we get $(x + h) \times y = xy + hy$, so it grows by $y$ times $h$. That is why the slope is $y$.

## 5.3 連鎖律: 傾きは掛け算でつながる - The chain rule: slopes connect by multiplication

計算がつながっているとき，全体の傾きはどうなるでしょうか？ 歯車で考えてみます。
> When operations are connected, what is the overall slope? Let's think with gears.

- 歯車Aを1回転させると，歯車Bは3回転する
- 歯車Bを1回転させると，歯車Cは2回転する
> - Turning gear A once turns gear B 3 times.
> - Turning gear B once turns gear C 2 times.

すると，歯車Aを1回転させると，歯車Cは $3 \times 2 = 6$ 回転しますね。
傾きも同じで，**つながった計算の傾きは，それぞれの傾きの掛け算** になります。これを **連鎖律** といいます。
> Then turning gear A once turns gear C $3 \times 2 = 6$ times.
Slopes work the same way: **the slope of connected operations is the product of each slope**. This is called the **chain rule**.

確かめてみましょう。$y = (2x + 1)^2$ を展開すると $4x^2 + 4x + 1$ なので，第4章の公式から $y' = 8x + 4$ です。
連鎖律で考えると，$u = 2x + 1$ とおいて
> Let's check. Expanding $y = (2x + 1)^2$ gives $4x^2 + 4x + 1$, so by the formulas in Chapter 4, $y' = 8x + 4$.
With the chain rule, let $u = 2x + 1$:

- $u$ を $x$ で微分: $2$
- $y = u^2$ を $u$ で微分: $2u = 2(2x + 1)$
- 掛け算: $2(2x + 1) \times 2 = 8x + 4$

同じ答えになりました！
> The same answer!

## 5.4 逆向きにたどる - Going backward

連鎖律を使うと，$L$ から出発して **右から左へ** 局所的な傾きを掛けていけば，すべての変数の傾きがわかります。
これを **逆伝播**(backward pass)といいます。
> With the chain rule, if we start from $L$ and multiply local slopes **from right to left**, we get the slopes of all the variables.
This is called the **backward pass**.

1. $L$ の $L$ に対する傾きは $1$(自分自身なので)
2. $L = e^2$ なので，$e$ の傾き $= 2e \times 1 = 8$
3. $e = d + c$ なので，$d$ の傾き $= 1 \times 8 = 8$，$c$ の傾き $= 1 \times 8 = 8$
4. $d = a \times b$ なので，$a$ の傾き $= b \times 8 = -24$，$b$ の傾き $= a \times 8 = 16$
> 1. The slope of $L$ with respect to itself is $1$.
> 2. Since $L = e^2$, the slope for $e$ is $2e \times 1 = 8$.
> 3. Since $e = d + c$, the slope for $d$ is $1 \times 8 = 8$, and for $c$ it is $1 \times 8 = 8$.
> 4. Since $d = a \times b$, the slope for $a$ is $b \times 8 = -24$, and for $b$ it is $a \times 8 = 16$.

**1回たどるだけで，すべての傾きがわかりました。** パラメーターが5000個あっても，逆向きに1回たどるだけで済みます。
> **One pass gave us all the slopes.** Even with 5000 parameters, one backward pass is enough.

### [アニメーション] 順伝播と逆伝播 - [Animation] Forward pass and backward pass
""")

code(r'''
#@title アニメーション: 計算グラフの順伝播と逆伝播 { display-mode: "form" }
def make_scene(mn, T):
    class BackpropGraph(mn.Scene):
        def construct(self):
            title = T('L = (a × b + c)²', font_size=34).to_edge(mn.UP)
            self.play(mn.Write(title))
            pos = {'a': [-5, 2.1, 0], 'b': [-5, 0, 0], 'c': [-5, -2.1, 0], 'd': [-2, 1.05, 0], 'e': [1, 0, 0], 'L': [4.5, 0, 0]}
            ops = {'d': '×', 'e': '+', 'L': '²'}
            val = {'a': 2, 'b': -3, 'c': 10, 'd': -6, 'e': 4, 'L': 16}
            grad = {'L': 1, 'e': 8, 'd': 8, 'c': 8, 'a': -24, 'b': 16}
            nodes = {}
            for k, p in pos.items():
                circ = mn.Circle(0.45, color=mn.BLUE if k in 'abc' else mn.TEAL).move_to(p)
                nodes[k] = mn.VGroup(circ, T(k, font_size=30).move_to(p))
            edges = [('a', 'd'), ('b', 'd'), ('d', 'e'), ('c', 'e'), ('e', 'L')]
            arrows = {e: mn.Arrow(nodes[e[0]].get_center(), nodes[e[1]].get_center(), buff=0.5, color=mn.GRAY) for e in edges}
            oplabels = [T(ops[k], font_size=26, color=mn.TEAL).next_to(nodes[k], mn.UP, buff=0.1) for k in ops]
            self.play(*[mn.FadeIn(n) for n in nodes.values()], *[mn.GrowArrow(a) for a in arrows.values()], *[mn.FadeIn(o) for o in oplabels])
            fw = T('順伝播: 左から右へ値を計算するよ', font_size=28, color=mn.YELLOW).to_edge(mn.DOWN)
            self.play(mn.FadeIn(fw))
            vlabels = {}
            for k in ['a', 'b', 'c', 'd', 'e', 'L']:
                v = T(f'{val[k]}', font_size=26, color=mn.YELLOW).next_to(nodes[k], mn.DOWN, buff=0.12)
                vlabels[k] = v
                self.play(mn.FadeIn(v, shift=mn.UP * 0.2), mn.Indicate(nodes[k], color=mn.YELLOW), run_time=0.45)
            bw = T('逆伝播: 右から左へ傾きを掛けていくよ', font_size=28, color=mn.RED).to_edge(mn.DOWN)
            self.play(mn.Transform(fw, bw))
            for k in ['L', 'e', 'd', 'c', 'a', 'b']:
                g = T(f'傾き {grad[k]}', font_size=24, color=mn.RED).next_to(vlabels[k], mn.DOWN, buff=0.1)
                incoming = [arrows[e] for e in edges if e[0] == k]
                anims = [mn.FadeIn(g, shift=mn.LEFT * 0.2), mn.Indicate(nodes[k], color=mn.RED)]
                anims += [mn.ShowPassingFlash(a.copy().set_color(mn.RED).rotate(mn.PI), time_width=0.6) for a in incoming]
                self.play(*anims, run_time=0.6)
            self.wait(2)
    return BackpropGraph

show_anim(make_scene)
''')

md(r"""
### 1つの変数が2回以上使われるとき - When one variable is used more than once

$L = a \times a$ のように，同じ変数が2か所で使われることもあります。このときは，**それぞれの道から来た傾きを足し合わせます**。
$a \times a$ の1つ目の $a$ から来る傾きは $a$，2つ目から来る傾きも $a$ なので，合計 $2a$ です。$a^2$ を微分した $2a$ と一致しますね。
> The same variable can be used in two places, like $L = a \times a$. Then we **add up the slopes coming from each path**.
The slope from the first $a$ is $a$, and from the second is also $a$, so the total is $2a$. This matches $2a$, the derivative of $a^2$.

## 5.5 実装: Valueクラス - Implementation: the Value class

これをプログラムにしたのが，0.1のコードの `Value` クラスです。
`Value` は1つの数を包む「箱」で，次の4つを覚えています。
> The `Value` class in 0.1 is this idea as a program.
A `Value` is a "box" that wraps one number and remembers these 4 things:

| 名前 | 意味 |
|---|---|
| `data` | 順伝播で計算した値 |
| `grad` | 逆伝播で求めた，lossに対する傾き |
| `_children` | この値を作るのに使った値(矢印の元) |
| `_local_grads` | それぞれの `_children` に対する局所的な傾き(5.2の表) |

`+` や `*` で計算するたびに，新しい `Value` が「自分を何から作ったか」と「局所的な傾き」を覚えていくので，計算が終わると自動的に計算グラフができあがっています。
> Every time we calculate with `+` or `*`, a new `Value` remembers "what it was made from" and "the local slopes", so the computation graph is built automatically by the time the calculation is done.
""")

code(r'''
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 順伝播で計算した値だよ
        self.grad = 0                   # 逆伝播で求める傾きだよ
        self._children = children       # 自分を作るのに使った値だよ
        self._local_grads = local_grads # それぞれに対する局所的な傾きだよ

    def __add__(self, other):  # x + y の局所的な傾きは (1, 1) だよ
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):  # x × y の局所的な傾きは (y, x) だよ
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))  # n x^(n-1)
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))                           # 1/x
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))                   # e^x
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))                   # 第7章で使うよ

    # 以下は，引き算や割り算などを上の計算の組み合わせで表すためのものだよ
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # 1) 計算グラフを「子が必ず親より前に来る」順番に並べるよ(トポロジカルソート)
        topo, visited = [], set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 2) 自分自身の傾きは1。そこから逆順にたどって，連鎖律で傾きを掛けて足し込むよ
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad   # 5.4の「足し合わせる」は += でやっているよ
''')

md(r"""
`backward` の中の「並べ替え」は，「ある値の傾きを使う前に，その値を使ったすべての計算からの傾きが足し込まれている」ようにするためのものです。
5.1の例で確かめてみましょう。
> The "sorting" inside `backward` makes sure that "before we use the slope of a value, the slopes from all the operations that used it have been added in".
Let's check with the example from 5.1.
""")

code(r'''
a, b, c = Value(2.0), Value(-3.0), Value(10.0)
L = (a * b + c) ** 2
L.backward()
print('L =', L.data)
print('傾き: a =', a.grad, ' b =', b.grad, ' c =', c.grad)

# 数値微分(第4章)でも確かめてみるよ
h = 1e-6
Lf = lambda a, b, c: (a * b + c) ** 2
print('数値微分: a =', round((Lf(2 + h, -3, 10) - Lf(2, -3, 10)) / h, 3),
      ' b =', round((Lf(2, -3 + h, 10) - Lf(2, -3, 10)) / h, 3),
      ' c =', round((Lf(2, -3, 10 + h) - Lf(2, -3, 10)) / h, 3))
''')

md(r"""
### [スライダー] 値を変えて傾きの変化を見よう - [Slider] Change the values and watch the slopes

$a, b, c$ を動かすと，計算グラフの各ノードの値(黒)と傾き(赤)が変わります。
例えば $b = 0$ にすると $a$ の傾きが0になります。「$b$ が0なら，$a$ をいくら変えても $a \times b$ は0のまま」だからです。
> Moving $a, b, c$ changes the value (black) and slope (red) of each node in the computation graph.
For example, when $b = 0$, the slope for $a$ becomes 0, because "if $b$ is 0, $a \times b$ stays 0 no matter how you change $a$".
""")

code(r'''
#@title [スライダー] 計算グラフの値と傾き { display-mode: "form" }
def plot_graph(a=2.0, b=-3.0, c=10.0):
    A, B, C = Value(a), Value(b), Value(c)
    D = A * B; E = D + C; L = E ** 2
    L.backward()
    nodes = {'a': (0, 2, A), 'b': (0, 1, B), 'c': (0, 0, C), 'd = a×b': (1.6, 1.5, D), 'e = d+c': (3.2, 1, E), 'L = e²': (4.8, 1, L)}
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for (s, t) in [('a', 'd = a×b'), ('b', 'd = a×b'), ('d = a×b', 'e = d+c'), ('c', 'e = d+c'), ('e = d+c', 'L = e²')]:
        ax.annotate('', xy=nodes[t][:2], xytext=nodes[s][:2],
                    arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.8, shrinkA=30, shrinkB=30))
    for name, (x, y, v) in nodes.items():
        is_leaf = name in 'abc'
        bg = '#eff6ff' if is_leaf else '#f0fdf4'
        edge = '#3b82f6' if is_leaf else '#10b981'
        ax.text(x, y + 0.05, f'{name}\n値: {v.data:.2f}', ha='center', va='center', fontsize=9.5, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=bg, edgecolor=edge, lw=1.5))
        ax.text(x, y - 0.22, f'傾き: {v.grad:.2f}', ha='center', va='center', color='#dc2626', fontsize=10, fontweight='bold')
    ax.set_xlim(-0.7, 5.5); ax.set_ylim(-0.6, 2.6); ax.axis('off')
    ax.set_title('計算グラフの値と傾きの可視化', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

sl = lambda v: widgets.FloatSlider(value=v, min=-5.0, max=12.0, step=0.5)
interact(plot_graph, a=sl(2.0), b=sl(-3.0), c=sl(10.0));
''')

md(r"""
## 5.6 数値微分と比べてみよう - Compare with numerical differentiation

第4章の「最初の文字の確率」の学習を，`Value` を使ってやり直してみます。
今度はlossを1回計算して `backward()` を1回呼ぶだけで，72個すべての傾きが求まります。
> Let's redo "learning the probabilities of the first character" from Chapter 4, this time with `Value`.
Now we only calculate the loss once and call `backward()` once to get all 72 slopes.
""")

code(r'''
import time

def first_char_loss_value(z):
    p = softmax(z)  # 0.1と同じ，Value用のsoftmaxだよ
    return sum(c * -p[i].log() for i, c in enumerate(first_counts) if c > 0) * (1 / n_names)

z_v = [Value(0.0) for _ in range(vocab_size)]
t0 = time.time()
for step in range(201):
    loss = first_char_loss_value(z_v)
    loss.backward()                    # 1回の逆伝播で全部の傾きが求まるよ
    for p in z_v:
        p.data -= 10.0 * p.grad
        p.grad = 0                     # 次のステップのために傾きを0に戻すよ
    if step % 40 == 0:
        print(f'step {step:3d}: loss = {loss.data:.4f}')
print(f'かかった時間: {time.time() - t0:.2f}秒')

t0 = time.time()
numerical_gradient(first_char_loss, [0.0] * vocab_size)
t_num = time.time() - t0
t0 = time.time()
first_char_loss_value([Value(0.0) for _ in range(vocab_size)]).backward()
t_back = time.time() - t0
print(f'1ステップの傾きを求める時間  数値微分: {t_num*1000:.1f}ミリ秒 / 逆伝播: {t_back*1000:.1f}ミリ秒')
print(f'数値微分は lossを{vocab_size + 1}回計算 / 逆伝播は 順伝播1回 + 逆伝播1回')
''')

md(r"""
今回はパラメーターが72個しかないので，時間の差はそれほど大きくないかもしれません(`Value` は1つ1つの計算を記録するぶん，1回あたりは遅いのです)。
でも，数値微分はパラメーターの数に比例して遅くなるのに対し，逆伝播はパラメーターが増えても「順伝播1回 + 逆伝播1回」のままです。
パラメーターが5000個，本物のLLMのように数千億個になると，この差は決定的になります。
> Since there are only 72 parameters here, the time difference may not be very big (`Value` records every single operation, so each step is slower).
But numerical differentiation gets slower in proportion to the number of parameters, while backpropagation stays at "one forward pass + one backward pass" even as parameters increase.
With 5000 parameters, or hundreds of billions as in real LLMs, this difference is decisive.

> PyTorchなどの本物の機械学習ライブラリも，やっていることはこの `Value` と同じです。ただし，1つ1つの数ではなく，たくさんの数をまとめて(行列として)一気に計算するので，ずっと速いのです。
> Real machine learning libraries like PyTorch do the same thing as this `Value`. But they calculate many numbers together at once (as matrices), not one number at a time, so they are much faster.

## 練習問題 - Exercises

1. $L = (a + b) \times b$ で $a = 1,\ b = 2$ のとき，$a$ と $b$ の傾きを手で計算してみましょう。そのあと `Value` で確かめてみましょう。
2. `p.grad = 0` を消すとどうなるでしょうか？

<details><summary>答え - Answers</summary>

1. $L = ab + b^2$ なので，$a$ の傾き $= b = 2$，$b$ の傾き $= a + 2b = 5$ です。
   計算グラフでは，$b$ は「$a + b$」と「$\times b$」の2か所で使われるので，$1 \times b$(=2)と $(a + b)$(=3)を足して5になります。
2. 前のステップの傾きがどんどん足し込まれて，正しい傾きにならなくなります(`+=` で足しているからです)。
</details>
""")
