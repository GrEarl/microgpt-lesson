from nbbuild import md, code

md(r"""
---
# 第4章 lossを小さくする: 勾配降下法 - Chapter 4: Making the loss smaller: gradient descent

前の章で，予想の外れ具合をlossという1つの数で表せるようになりました。
学習とは，**lossが一番小さくなるようにパラメーター(モデルの中の数)を調整すること** です。
この章では，数学Iの二次関数から出発して，その調整の方法である **勾配降下法** を学びます。
> In the last chapter, we learned to express how wrong the predictions are as one number, the loss.
Learning means **adjusting the parameters (the numbers inside the model) so that the loss becomes as small as possible**.
In this chapter, starting from quadratic functions in Math I, we learn the method for this adjustment: **gradient descent**.

**この章で使う数学**: 二次関数(数学I)，変化の割合，微分(数学IIの内容をここで学びます)
> **Math in this chapter**: quadratic functions (Math I), rate of change, derivatives (a Math II topic, learned here)

## 4.1 復習: 二次関数の最小値 - Review: the minimum of a quadratic function

$f(x) = x^2 - 6x + 10$ の最小値を求めましょう。数学Iで習った **平方完成** を使うと
> Let's find the minimum of $f(x) = x^2 - 6x + 10$. Using **completing the square** from Math I:

$$f(x) = (x - 3)^2 + 1$$

となるので，$x = 3$ のとき最小値 $1$ をとります。グラフの頂点が $(3, 1)$ ですね。
> So the minimum value is $1$ at $x = 3$. The vertex of the graph is $(3, 1)$.

## 4.2 式がわからないときはどうする？ - What if we don't know the formula?

二次関数なら平方完成で一発ですが，GPTのlossは **5000個以上のパラメーターを持つ，とても複雑な関数** です。平方完成のような方法は使えません。
わかるのは，「今のパラメーターでlossを計算するといくつになるか」だけです。
> For a quadratic function, completing the square solves it at once. But the loss of a GPT is **a very complicated function with more than 5000 parameters**. We cannot use a method like completing the square.
All we know is "what the loss is for the current parameters".

これは，**霧の中で山を下りる** のに似ています。遠くは見えませんが，足元の地面が「どちらに傾いているか」はわかります。
それなら，**低くなる方向に少しずつ進めば**，いつか谷底にたどり着けるはずです。
そのために必要なのが，「傾き」を求める方法です。
> This is like **walking down a mountain in fog**. You cannot see far, but you can feel which way the ground at your feet is sloping.
Then, if you **keep taking small steps in the downhill direction**, you should eventually reach the bottom of the valley.
What we need for this is a way to find the "slope".

## 4.3 傾き: 変化の割合 - Slope: the rate of change

中学や数学Iで習った **変化の割合** を思い出しましょう。$x$ が $a$ から $b$ まで変わるとき
> Remember the **rate of change** from junior high and Math I. When $x$ changes from $a$ to $b$:

$$\text{変化の割合} = \frac{y\text{ の増加量}}{x\text{ の増加量}} = \frac{f(b) - f(a)}{b - a}$$

これはグラフ上の2点を結んだ直線の傾きです。
> This is the slope of the straight line connecting two points on the graph.
""")

code(r'''
def f(x):
    return x**2 - 6*x + 10

def rate_of_change(f, a, b):
    return (f(b) - f(a)) / (b - a)

print('x: 0 → 1 の変化の割合:', rate_of_change(f, 0, 1))
print('x: 4 → 5 の変化の割合:', rate_of_change(f, 4, 5))
''')

md(r"""
$x = 0$ 付近では傾きがマイナス(右下がり)，$x = 4$ 付近ではプラス(右上がり)です。
つまり，谷底 $x = 3$ は「傾きがマイナスからプラスに変わるところ」にあります。
> Near $x = 0$ the slope is negative (going down), and near $x = 4$ it is positive (going up).
So the bottom of the valley at $x = 3$ is "where the slope changes from negative to positive".

## 4.4 2点を近づける: 微分 - Bringing the two points closer: derivatives

でも，2点の間の変化の割合は「その区間の平均の傾き」にすぎません。知りたいのは **ちょうどその点での傾き** です。
そこで，2点目を $a + h$ として，$h$ をどんどん0に近づけてみます。
> But the rate of change between two points is only "the average slope over that interval". What we want is **the slope exactly at that point**.
So let the second point be $a + h$, and make $h$ closer and closer to 0.

### [アニメーション] 2点を近づけると接線になる - [Animation] Bringing two points together gives a tangent line
""")

code(r'''
#@title アニメーション: 割線から接線へ { display-mode: "form" }
def make_scene(mn, T):
    class SecantToTangent(mn.Scene):
        def construct(self):
            ax = mn.Axes(x_range=[-0.5, 3, 1], y_range=[-0.5, 6, 1], x_length=7, y_length=5.5, tips=False).shift(mn.LEFT * 1.8 + mn.DOWN * 0.3)
            graph = ax.plot(lambda x: x**2, x_range=[-0.3, 2.45], color=mn.BLUE)
            lab = T('y = x²', font_size=28, color=mn.BLUE).next_to(ax.c2p(2.3, 5.3), mn.LEFT)
            self.play(mn.Create(ax), mn.Create(graph), mn.FadeIn(lab))
            a = 1.0
            h = mn.ValueTracker(1.3)
            p = mn.Dot(ax.c2p(a, a**2), color=mn.YELLOW)
            q = mn.always_redraw(lambda: mn.Dot(ax.c2p(a + h.get_value(), (a + h.get_value())**2), color=mn.RED))
            def secant():
                hv = h.get_value(); s = ((a + hv)**2 - a**2) / hv
                return ax.plot(lambda x: a**2 + s * (x - a), x_range=[-0.3, 2.8], color=mn.RED)
            line = mn.always_redraw(secant)
            def info_mob():
                hv = h.get_value()
                return mn.VGroup(T(f'幅 h = {hv:.3f}', font_size=28),
                                 T(f'傾き = {((a + hv)**2 - a**2) / hv:.3f}', font_size=28, color=mn.RED),
                                 ).arrange(mn.DOWN, aligned_edge=mn.LEFT).to_corner(mn.UR).shift(mn.DOWN * 0.8)
            info = info_mob()
            self.play(mn.FadeIn(p), mn.FadeIn(q), mn.Create(line), mn.FadeIn(info))
            for hv in [1.0, 0.7, 0.5, 0.3, 0.15, 0.05, 0.01, 0.001]:
                self.play(h.animate.set_value(hv), run_time=0.6)
                self.play(mn.Transform(info, info_mob()), run_time=0.3)
            ans = T('x = 1 での傾き(微分係数) = 2', font_size=28, color=mn.YELLOW).to_edge(mn.DOWN)
            self.play(mn.Write(ans))
            self.wait(1.5)
    return SecantToTangent

show_anim(make_scene)
''')

md(r"""
$h$ を0に近づけていくと，2点を結ぶ直線は，その点でグラフに接する直線(**接線**)に近づきます。
この接線の傾きを **微分係数** といい，$f'(a)$ と書きます。
> As $h$ approaches 0, the line through the two points approaches the line that touches the graph at that point (the **tangent line**).
The slope of this tangent line is called the **derivative** at $a$, written $f'(a)$:

$$f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}$$

($\lim_{h \to 0}$ は「$h$ を限りなく0に近づけたときに近づく値」という意味です)
> ($\lim_{h \to 0}$ means "the value it approaches as $h$ gets infinitely close to 0".)

$f(x) = x^2$ で計算してみましょう。展開すると
> Let's calculate it for $f(x) = x^2$. Expanding:

$$\frac{(a + h)^2 - a^2}{h} = \frac{2ah + h^2}{h} = 2a + h$$

$h$ を0に近づけると $2a$ に近づくので，$f'(a) = 2a$ です。$a = 1$ なら傾きは2で，アニメーションと一致しますね。
> As $h$ approaches 0, this approaches $2a$, so $f'(a) = 2a$. When $a = 1$ the slope is 2, matching the animation.

### [スライダー] 点と h を動かして傾きを見よう - [Slider] Move the point and h to see the slope
""")

code(r'''
#@title [スライダー] 割線の傾きと接線の微分係数 { display-mode: "form" }
def plot_secant(a=1.0, h=1.0):
    g = lambda x: x**2
    xs = [i / 50 for i in range(-150, 151)]
    s = (g(a + h) - g(a)) / h
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(xs, [g(x) for x in xs], lw=2.5, color='#4f46e5', label='y = x²')
    ax.plot(xs, [g(a) + s * (x - a) for x in xs], color='#ef4444', lw=1.8, label=f'2点を結ぶ直線 (傾き: {s:.3f})')
    ax.plot(xs, [g(a) + 2 * a * (x - a) for x in xs], color='#10b981', ls='--', lw=2.0, label=f'接線 2a (傾き: {2 * a:.3f})')
    ax.plot([a, a + h], [g(a), g(a + h)], 'o', color='#1e293b', ms=6)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 7.5)
    ax.legend(loc='upper center', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    ax.set_title(f'x = {a:.1f} における割線と接線の比較 (h = {h:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_secant,
         a=widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1, description='点 a:'),
         h=widgets.FloatSlider(value=1.0, min=0.01, max=2.0, step=0.01, description='幅 h:'));
''')

md(r"""
$a$ を変えれば，その点での傾きもわかります。$a$ に対して傾き $f'(a)$ を対応させる関数を **導関数** といい，$f'(x)$ と書きます。「導関数を求めること」を **微分する** といいます。
同じように計算すると，次の公式が得られます。
> We can find the slope at any point $a$. The function that maps $a$ to its slope $f'(a)$ is called the **derivative function**, written $f'(x)$. Finding it is called **differentiating**.
Calculating in the same way, we get these formulas:

| 関数 | 導関数 | ひとこと |
|---|---|---|
| $c$ (定数) | $0$ | 平らなので傾き0 |
| $x^n$ | $n x^{n-1}$ | $x^2 \to 2x$，$x^3 \to 3x^2$ |
| $k f(x)$ | $k f'(x)$ | 定数倍はそのまま |
| $f(x) + g(x)$ | $f'(x) + g'(x)$ | 足し算は別々に微分 |
| $e^x$ | $e^x$ | 微分しても変わらない！ |
| $\log x$ | $\dfrac{1}{x}$ | |

最後の2つ($e^x$ と $\log x$)は数学IIIの内容なので，ここでは「そうなる」と受け入れて，コンピューターで確かめるだけにしましょう。$e$ が特別な数なのは，$e^x$ が微分しても形が変わらないからです。
> The last two ($e^x$ and $\log x$) are Math III topics, so let's just accept them here and check them with a computer. $e$ is special because $e^x$ keeps its form when differentiated.
""")

code(r'''
def numerical_derivative(f, a, h=1e-6):
    return (f(a + h) - f(a)) / h

a = 1.5
print(f'f(x) = x^2 - 6x + 10: 数値で {numerical_derivative(f, a):.4f}  公式 2x - 6 = {2*a - 6:.4f}')
print(f'e^x:                 数値で {numerical_derivative(math.exp, a):.4f}  公式 e^x = {math.exp(a):.4f}')
print(f'log x:               数値で {numerical_derivative(math.log, a):.4f}  公式 1/x = {1/a:.4f}')
''')

md(r"""
## 4.5 勾配降下法: 傾きの逆向きに少しずつ進む - Gradient descent: small steps against the slope

傾きがわかれば，下る方向がわかります。
- 傾きがプラス(右上がり) → 左に進むと下がる
- 傾きがマイナス(右下がり) → 右に進むと下がる
> Once we know the slope, we know which way is down.
> - Positive slope (going up to the right) → moving left goes down.
> - Negative slope (going down to the right) → moving right goes down.

どちらの場合も「**傾きと逆の向きに進む**」と下がります。これを式にしたのが **勾配降下法** です。
> In both cases, **moving against the slope** goes down. Writing this as a formula gives **gradient descent**:

$$x \leftarrow x - \eta \, f'(x)$$

$\eta$(イータ)は **学習率** で，1回にどれくらい進むかを決めます。$\leftarrow$ は「右辺の値で $x$ を更新する」という意味です。
> $\eta$ (eta) is the **learning rate**, which decides how far to move each time. $\leftarrow$ means "update $x$ with the value on the right".
""")

code(r'''
def df(x):
    return 2*x - 6  # f(x) = x^2 - 6x + 10 の導関数だよ

x = 0.0
learning_rate = 0.1
for step in range(31):
    if step % 5 == 0:
        print(f'step {step:2d}: x = {x:.4f}, f(x) = {f(x):.4f}')
    x = x - learning_rate * df(x)
''')

md(r"""
平方完成をしなくても，$x = 3$，最小値 $1$ にたどり着きました！
> Without completing the square, we reached $x = 3$ and the minimum value $1$!

### [アニメーション] 坂を下るボール - [Animation] A ball rolling down the slope
""")

code(r'''
#@title アニメーション: 勾配降下法 { display-mode: "form" }
def make_scene(mn, T):
    class GradientDescent(mn.Scene):
        def construct(self):
            fx = lambda x: (x - 3)**2 + 1
            ax = mn.Axes(x_range=[-0.5, 6.5, 1], y_range=[0, 12, 2], x_length=8, y_length=4.8, tips=False).shift(mn.DOWN * 0.1)
            graph = ax.plot(fx, x_range=[-0.1, 6.1], color=mn.BLUE)
            title = T('勾配降下法: x ← x − η f′(x)', font_size=32).to_edge(mn.UP)
            self.play(mn.Write(title), mn.Create(ax), mn.Create(graph))
            x, lr = 0.1, 0.15
            ball = mn.Dot(ax.c2p(x, fx(x)), color=mn.YELLOW, radius=0.12)
            self.play(mn.FadeIn(ball))
            for i in range(10):
                s = 2 * (x - 3)
                lo, hi = x - 1, x + 1
                if s != 0:
                    t_top = x + (11.5 - fx(x)) / s
                    if s > 0: hi = min(hi, t_top)
                    else: lo = max(lo, t_top)
                tan = ax.plot(lambda t: fx(x) + s * (t - x), x_range=[lo, hi], color=mn.RED)
                lab = T(f'傾き {s:+.2f}', font_size=26, color=mn.RED).next_to(ax.c2p(x, fx(x)), mn.UR, buff=0.2)
                self.play(mn.Create(tan), mn.FadeIn(lab), run_time=0.35)
                x = x - lr * s
                self.play(ball.animate.move_to(ax.c2p(x, fx(x))), mn.FadeOut(tan), mn.FadeOut(lab), run_time=0.45)
            note = T('傾きが0に近づくと，歩幅も自然に小さくなるよ', font_size=26).to_edge(mn.DOWN)
            self.play(mn.Write(note))
            self.wait(1.5)
    return GradientDescent

show_anim(make_scene)
''')

md(r"""
## 4.6 学習率の大きさ - How big should the learning rate be?

学習率が小さすぎると，なかなか谷底にたどり着きません。逆に大きすぎると，谷底を飛び越えてしまいます。
スライダーで確かめてみましょう。この関数では，学習率が1を超えると，谷底からどんどん遠ざかってしまいます(**発散**)。
> If the learning rate is too small, it takes a long time to reach the bottom. If it is too large, it jumps over the bottom.
Check it with the slider. For this function, when the learning rate is larger than 1, it moves farther and farther away from the bottom (**divergence**).

### [スライダー] 学習率とステップ数 - [Slider] Learning rate and number of steps
""")

code(r'''
#@title [スライダー] 学習率を変えて坂を下る { display-mode: "form" }
def plot_descent(learning_rate=0.1, steps=10, start=0.0):
    x, path = start, [start]
    for _ in range(steps):
        x = x - learning_rate * df(x)
        path.append(x)
    xs = [i / 20 for i in range(-40, 161)]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(xs, [f(t) for t in xs], lw=2.5, color='#3b82f6', label='f(x) = (x-3)² + 1')
    shown = [p for p in path if -2 <= p <= 8]
    ax.plot(shown, [f(p) for p in shown], 'o-', color='#ef4444', ms=5, lw=1.5, alpha=0.8, label='探索ルート')
    if -2 <= path[-1] <= 8:
        ax.plot([path[-1]], [f(path[-1])], 'o', color='#eab308', ms=11, mec='#a16207', mew=2, label='現在地')
    ax.set_xlim(-2, 8)
    ax.set_ylim(-1, 30)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right')
    if not (-2 <= path[-1] <= 8) or f(path[-1]) > 30:
        ax.text(0.5, 0.85, '発散！(画面の外に飛んでいる)', transform=ax.transAxes, ha='center',
                color='#dc2626', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='#fee2e2', edgecolor='#ef4444', boxstyle='round,pad=0.5'))
    ax.set_title(f'{steps}ステップ後: x = {path[-1]:.4g}, f(x) = {f(path[-1]):.4g}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_descent,
         learning_rate=widgets.FloatSlider(value=0.1, min=0.01, max=1.1, step=0.01, description='学習率 η:'),
         steps=widgets.IntSlider(value=10, min=0, max=50, step=1, description='ステップ数:'),
         start=widgets.FloatSlider(value=0.0, min=-1, max=7, step=0.5, description='スタート:'));
''')

md(r"""
学習率0.9くらいにすると，谷底をはさんで左右にジグザグしながら近づく様子が見られます。1を超えると発散します。
0.1のコードでは，学習率を **最初は大きく，だんだん小さく** していました(`lr_t = learning_rate * (1 - step / num_steps)`)。
最初は大きく進んで早く谷に近づき，最後は小さく進んで谷底を飛び越えないようにするためです。
> Around 0.9, you can see it zigzag left and right across the bottom while getting closer. Above 1, it diverges.
In 0.1, the learning rate was **large at first and gradually smaller** (`lr_t = learning_rate * (1 - step / num_steps)`).
This is to move fast toward the valley at first, and take small steps at the end so as not to jump over the bottom.

## 4.7 変数が2つ以上あるとき - When there are two or more variables

GPTのパラメーターは5000個以上あります。変数が2つの場合で考えてみましょう。
> A GPT has more than 5000 parameters. Let's think about the case with two variables:

$$f(x, y) = (x - 1)^2 + 2(y + 2)^2$$

変数が2つあっても，**「他の変数を定数だと思って，1つの変数だけで微分する」** ことができます。これを **偏微分** といいます。
> Even with two variables, we can **treat the other variables as constants and differentiate with respect to just one**. This is called **partial differentiation**.

- $y$ を定数とみて $x$ で微分: $2(x - 1)$
- $x$ を定数とみて $y$ で微分: $4(y + 2)$
> - Treat $y$ as a constant and differentiate by $x$: $2(x - 1)$
> - Treat $x$ as a constant and differentiate by $y$: $4(y + 2)$

この2つの傾きを並べたもの $\big(2(x-1),\ 4(y+2)\big)$ を **勾配(gradient)** といいます。勾配降下法は，$x$ と $y$ をそれぞれ自分の傾きの逆向きに動かします。
変数が5000個あっても，やることは同じです。
> The pair of these two slopes $\big(2(x-1),\ 4(y+2)\big)$ is called the **gradient**. Gradient descent moves $x$ and $y$ each against its own slope.
Even with 5000 variables, we do the same thing.

### [スライダー] 等高線の上を下る - [Slider] Going down on a contour map

下の図は，$f(x, y)$ の値を地図の **等高線** のように表したものです。中心(×印)が谷底です。
> The figure below shows the values of $f(x, y)$ like **contour lines** on a map. The center (×) is the bottom of the valley.
""")

code(r'''
#@title [スライダー] 2変数の勾配降下法 { display-mode: "form" }
import numpy as np
def f2(x, y): return (x - 1)**2 + 2 * (y + 2)**2

def plot_descent2(learning_rate=0.1, steps=20):
    x, y = -3.0, 1.5
    path = [(x, y)]
    for _ in range(steps):
        gx, gy = 2 * (x - 1), 4 * (y + 2)
        x, y = x - learning_rate * gx, y - learning_rate * gy
        path.append((x, y))
    X, Y = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-5, 3, 200))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cs = ax.contour(X, Y, f2(X, Y), levels=20, cmap='viridis')
    px = [p[0] for p in path if abs(p[0]) < 10 and abs(p[1]) < 10]
    py = [p[1] for p in path if abs(p[0]) < 10 and abs(p[1]) < 10]
    ax.plot(px, py, 'o-', color='#ef4444', ms=4.5, lw=1.5, label='探索ルート')
    ax.plot([1], [-2], 'x', color='#0f172a', ms=12, mew=3, label='谷底 (目標地点)')
    ax.set_xlim(-4, 4); ax.set_ylim(-5, 3); ax.set_aspect('equal')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper left')
    if not (-4 <= path[-1][0] <= 4 and -5 <= path[-1][1] <= 3):
        ax.text(0.5, 0.08, '発散！(画面の外に飛んでいる)', transform=ax.transAxes, ha='center',
                color='#dc2626', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='#fee2e2', edgecolor='#ef4444', boxstyle='round,pad=0.5'))
    ax.set_title(f'{steps}ステップ後: f = {f2(*path[-1]):.4g}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_descent2,
         learning_rate=widgets.FloatSlider(value=0.1, min=0.01, max=0.55, step=0.01, description='学習率 η:'),
         steps=widgets.IntSlider(value=20, min=0, max=60, step=1, description='ステップ数:'));
''')

md(r"""
学習率を0.25より大きくすると，$y$ の方向だけジグザグし始め，0.5を超えると発散します。$y$ の方向のほうが坂が急だからです。
方向によって坂の急さが違うと，1つの学習率ではうまくいかないことがあります。この問題は第10章のAdamで解決します。
> When the learning rate is larger than 0.25, it starts to zigzag only in the $y$ direction, and above 0.5 it diverges, because the slope is steeper in that direction.
When steepness differs by direction, a single learning rate may not work well. We will solve this with Adam in Chapter 10.

## 4.8 実装: 勾配降下法で確率を学習する - Implementation: learning probabilities with gradient descent

いよいよ勾配降下法で，名前について何かを学習させてみましょう。
まずは簡単な例として，「**名前の最初の文字** の確率」を学習させます。
72個のスコア $z_0, z_1, \ldots, z_{71}$ をパラメーターとし，softmaxで確率に変えて，lossを計算します。
> Now let's use gradient descent to learn something about names.
As a simple example, we learn "the probabilities of **the first character of a name**".
We use 72 scores $z_0, z_1, \ldots, z_{71}$ as parameters, turn them into probabilities with softmax, and calculate the loss.

微分の公式を使って傾きを求めるのは大変なので，4.4でやったように **数値微分**(ほんの少し $h$ だけ動かして変化の割合を計算する)で傾きを求めます。
> Finding the slopes with formulas would be hard, so we find them by **numerical differentiation** (move by a tiny $h$ and calculate the rate of change), as in 4.4.
""")

code(r'''
first_counts = counts[BOS]   # 第2章の表から「BOSの次(=最初の文字)」の回数を取り出すよ
n_names = sum(first_counts)

def first_char_loss(z):
    p = softmax_numbers(z)
    return sum(c * -math.log(p[i]) for i, c in enumerate(first_counts) if c > 0) / n_names

def numerical_gradient(loss_fn, z, h=1e-5):
    base = loss_fn(z)
    grad = []
    for i in range(len(z)):           # パラメーターを1個ずつ少しだけ動かして…
        z_moved = z.copy()
        z_moved[i] += h
        grad.append((loss_fn(z_moved) - base) / h)  # …変化の割合を求めるよ
    return grad

z = [0.0] * vocab_size   # 最初はすべて同じスコア(=でたらめな予想)だよ
learning_rate = 10.0
n_evals = 0
for step in range(201):
    grad = numerical_gradient(first_char_loss, z)
    n_evals += len(z) + 1
    z = [zi - learning_rate * gi for zi, gi in zip(z, grad)]
    if step % 40 == 0:
        print(f'step {step:3d}: loss = {first_char_loss(z):.4f}')
print('lossを計算した回数:', n_evals)
''')

md(r"""
lossが下がりました。学習した確率を，第2章で数えて求めた相対度数と比べてみましょう。
> The loss went down. Let's compare the learned probabilities with the relative frequencies we counted in Chapter 2.
""")

code(r'''
#@title 学習した確率と数えた確率の比較 { display-mode: "form" }
learned = softmax_numbers(z)
counted = [c / n_names for c in first_counts]
top = sorted(range(vocab_size), key=lambda i: -counted[i])[:15]
xs = range(len(top))
fig, ax = plt.subplots(figsize=(8.5, 3.4))
ax.bar([i - 0.2 for i in xs], [counted[i] for i in top], width=0.4, label='数えた相対度数(第2章)', color='#0d9488', edgecolor='#0f766e')
ax.bar([i + 0.2 for i in xs], [learned[i] for i in top], width=0.4, label='勾配降下法で学習した確率', color='#6366f1', edgecolor='#4f46e5')
ax.set_xticks(list(xs), [tok_str(i) for i in top])
ax.set_ylabel('確率')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
ax.set_title('名前の最初の文字')
plt.tight_layout()
plt.show()
''')

md(r"""
ほぼ同じになりました！ 勾配降下法でlossを小さくすると，数えて求めた確率と同じ答えにたどり着くのです。
数えることができない複雑なモデルでも，lossさえ計算できれば，同じ方法で学習できます。これが機械学習の基本的な考え方です。
> They are almost the same! Making the loss smaller with gradient descent leads to the same answer as counting.
Even for complicated models where we cannot count, we can learn in the same way as long as we can calculate the loss. This is the basic idea of machine learning.

**でも問題があります。** 72個のパラメーターの傾きを求めるのに，1ステップごとにlossを73回計算しました。
0.1のGPTにはパラメーターが5000個以上あるので，1ステップごとにGPT全体の計算を5000回以上やり直すことになります。これではとても遅すぎます。
次の章では，**1回の計算で全部のパラメーターの傾きを一気に求める方法** を学びます。
> **But there is a problem.** To find the slopes of 72 parameters, we calculated the loss 73 times per step.
The GPT in 0.1 has more than 5000 parameters, so each step would need more than 5000 full GPT calculations. That is far too slow.
In the next chapter, we learn **a way to find the slopes of all the parameters at once, in one pass**.

## 4.9 0.1のコードとの対応 - Matching with the code in 0.1

```python
lr_t = learning_rate * (1 - step / num_steps)        # 4.6 学習率をだんだん小さくするよ
p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)   # 4.5 傾きの逆向きに進むよ(Adamという改良版。第10章)
```

## 練習問題 - Exercises

1. $f(x) = 3x^2 + 2x$ の導関数を求めてみましょう。
2. $f(x) = (x + 2)^2$ で，$x = 0$，学習率0.25から勾配降下法を1ステップ進めると，$x$ はいくつになるでしょうか？
3. [スライダー] (4.6)で，学習率をちょうど1にするとどうなるでしょうか？ なぜそうなるか考えてみましょう。

<details><summary>答え - Answers</summary>

1. $f'(x) = 6x + 2$ です。
2. $f'(x) = 2(x + 2)$ なので $f'(0) = 4$ です。$x = 0 - 0.25 \times 4 = -1$ になります。
3. 谷底をはさんで同じ2点($x = 0$ と $x = 6$)を行ったり来たりし続けます。$x - 1 \cdot 2(x - 3) = 6 - x$ なので，谷底 $3$ からの距離が変わらないからです。
</details>
""")
