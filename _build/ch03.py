from nbbuild import md, code

md(r"""
---
# 第3章 どれくらい外れた？: 損失とsoftmax - Chapter 3: How wrong was it? Loss and softmax

学習とは，「予想の外れ具合」を少しずつ小さくしていくことです。
そのためにはまず，外れ具合を **1つの数** で表す必要があります。それが0.1で表示されていた **loss(損失)** です。
この章では，lossの正体と，スコアを確率に変える **softmax** を学びます。
> Learning means making "how wrong the predictions are" smaller, little by little.
To do that, we first need to express how wrong they are as **one number**. That is the **loss** shown in 0.1.
In this chapter, we learn what the loss really is, and **softmax**, which turns scores into probabilities.

**この章で使う数学**: 指数・対数(数学IIの内容を，数学Iの知識からここで学びます)
> **Math in this chapter**: exponents and logarithms (Math II topics, learned here starting from Math I)

## 3.1 予想の良さを数で表したい - We want to measure how good a prediction is

第2章のbigramモデルが「さくら」という名前をどれくらい予想できるかを考えます。
モデルが「さくら」を作る確率は，4回のくじ引きがすべて当たる確率なので，それぞれの確率の **積** になります。
> Let's think about how well the bigram model from Chapter 2 predicts the name "さくら".
The probability that the model makes "さくら" is the probability of winning all 4 draws, so it is the **product** of each probability:

$$P(\text{さくら}) = P(\text{さ}\mid\text{BOS}) \times P(\text{く}\mid\text{さ}) \times P(\text{ら}\mid\text{く}) \times P(\text{BOS}\mid\text{ら})$$

この確率が大きいほど，良いモデルだといえます。計算してみましょう。
> The larger this probability, the better the model. Let's calculate it.
""")

code(r'''
def bigram_prob(name):
    ids = tokenize(name)
    p = 1.0
    for a, b in zip(ids, ids[1:]):
        p *= next_probs(a)[b]
    return p

print('P(さくら) =', bigram_prob('さくら'))
print('P(はなこ) =', bigram_prob('はなこ'))
''')

md(r"""
では，学習データ全部の名前を作る確率はどうでしょうか？ 全部の積を計算してみます。
> Then what about the probability of making all the names in the training data? Let's multiply them all.
""")

code(r'''
total = 1.0
for name in docs:
    total *= bigram_prob(name)
print('全部の名前の確率の積 =', total)
''')

md(r"""
0.0になってしまいました！ 小さい数を3000回以上掛けると，コンピューターが表せる一番小さい数よりも小さくなってしまうのです。
**掛け算を足し算に変える道具** があれば，この問題を解決できます。それが **対数(log)** です。
対数を理解するために，まず指数から始めましょう。
> It became 0.0! Multiplying small numbers more than 3000 times gives a number smaller than the smallest number a computer can represent.
A **tool that turns multiplication into addition** would solve this problem. That tool is the **logarithm (log)**.
To understand logarithms, let's start with exponents.

## 3.2 指数: 累乗を広げる - Exponents: extending powers

中学で習った累乗 $2^3 = 2 \times 2 \times 2 = 8$ を思い出しましょう。右上の小さい数を **指数** といいます。
累乗には次の **指数法則** があります。
> Remember powers from junior high: $2^3 = 2 \times 2 \times 2 = 8$. The small number at the upper right is the **exponent**.
Powers follow the **laws of exponents**:

$$a^m \times a^n = a^{m+n}, \qquad (a^m)^n = a^{mn}$$

この法則が成り立つように考えると，指数は0や負の数，分数にも広げられます。
> If we want these laws to keep working, exponents can be extended to 0, negative numbers and fractions:

| 指数 | 考え方 | 例 |
|---|---|---|
| 0 | $2^3 \div 2 = 2^2$，$2^2 \div 2 = 2^1$，… と1つ減らすと2で割る。だから $2^0 = 2^1 \div 2 = 1$ | $2^0 = 1$ |
| 負の数 | さらに2で割っていく | $2^{-1} = \frac{1}{2}$，$2^{-2} = \frac{1}{4}$ |
| 分数 | $2^{\frac12} \times 2^{\frac12} = 2^1 = 2$ だから，2乗して2になる数 | $2^{\frac12} = \sqrt{2} \approx 1.414$ |
""")

code(r'''
print(2**3, 2**0, 2**-1, 2**-2, 2**0.5)
print('指数法則の確認: 2**3 * 2**4 =', 2**3 * 2**4, ' 2**7 =', 2**7)
''')

md(r"""
### [スライダー] 指数関数のグラフ - [Slider] The graph of an exponential function

$y = a^x$ のグラフを，$a$ を変えながら見てみましょう。
$a > 1$ なら右上がり，$0 < a < 1$ なら右下がりです。**どんな $x$ でも $y$ は必ず正** で，$x = 0$ のときはいつも $y = 1$ になります。
> Let's look at the graph of $y = a^x$ while changing $a$.
If $a > 1$ it goes up to the right; if $0 < a < 1$ it goes down. **For any $x$, $y$ is always positive**, and at $x = 0$, $y$ is always 1.
""")

code(r'''
#@title [スライダー] 指数関数 y = a^x { display-mode: "form" }
def plot_exp(a=2.0):
    xs = [i / 20 for i in range(-60, 61)]
    ys = [a**x for x in xs]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, ys, lw=2.5, color='#4f46e5', label=f'y = {a:.1f}^x')
    ax.axhline(0, color='#64748b', lw=1.0)
    ax.axvline(0, color='#64748b', lw=1.0)
    ax.plot([0], [1], 'o', color='#ef4444', ms=7)
    ax.annotate('(0, 1) を必ず通るよ', (0, 1), xytext=(15, 10), textcoords='offset points',
                fontsize=9.5, fontweight='bold', color='#ef4444',
                arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.2))
    ax.set_ylim(-0.5, 9)
    ax.set_xlim(-3, 3)
    ax.set_title(f'指数関数 y = {a:.1f}^x のグラフ', fontsize=12, fontweight='bold')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()

interact(plot_exp, a=widgets.FloatSlider(value=2.0, min=0.2, max=4.0, step=0.1, description='底 a:'));
''')

md(r"""
## 3.3 対数: 掛け算を足し算に変える - Logarithms: turning multiplication into addition

「2を何乗すると8になるか？」の答えは3です。これを
> "To what power must we raise 2 to get 8?" The answer is 3. We write this as

$$\log_2 8 = 3$$

と書き，**2を底とする8の対数** といいます。つまり $\log_a M = p$ と $a^p = M$ は同じ意味です。
> and call it **the logarithm of 8 to the base 2**. In other words, $\log_a M = p$ and $a^p = M$ mean the same thing.

指数法則 $a^m \times a^n = a^{m+n}$ を対数の言葉で言いかえると，次のとても大事な性質になります。
> If we rewrite the law of exponents $a^m \times a^n = a^{m+n}$ in terms of logarithms, we get this very important property:

$$\log_a (M \times N) = \log_a M + \log_a N$$

**掛け算の対数は，対数の足し算になる** のです。
> **The log of a product is the sum of the logs.**
""")

code(r'''
print('log2(8)  =', math.log2(8))
print('log2(4 × 8) =', math.log2(4 * 8), ' log2(4) + log2(8) =', math.log2(4) + math.log2(8))
''')

md(r"""
### [アニメーション] 掛け算が足し算になる - [Animation] Multiplication becomes addition
""")

code(r'''
#@title アニメーション: 対数の性質 { display-mode: "form" }
def make_scene(mn, T):
    class LogScene(mn.Scene):
        def construct(self):
            title = T('log: 掛け算 → 足し算', font_size=36).to_edge(mn.UP)
            self.play(mn.Write(title))
            nums = [1, 2, 4, 8, 16, 32, 64]
            top = mn.VGroup(*[T(str(n), font_size=36) for n in nums]).arrange(mn.RIGHT, buff=0.9).shift(mn.UP * 1)
            bot = mn.VGroup(*[T(str(i), font_size=36, color=mn.YELLOW) for i in range(len(nums))]).arrange(mn.RIGHT, buff=0.9).shift(mn.DOWN * 1)
            for t, b in zip(top, bot): b.set_x(t.get_x())
            lt = T('2の累乗', font_size=26).next_to(top, mn.LEFT, buff=0.6)
            lb = T('底2の log', font_size=26, color=mn.YELLOW).next_to(bot, mn.LEFT, buff=0.6)
            self.play(mn.FadeIn(top), mn.FadeIn(lt))
            arrows = mn.VGroup(*[mn.Arrow(t.get_bottom(), b.get_top(), buff=0.15, color=mn.GRAY) for t, b in zip(top, bot)])
            self.play(mn.LaggedStart(*[mn.GrowArrow(a) for a in arrows], lag_ratio=0.1), mn.FadeIn(bot), mn.FadeIn(lb))
            m1 = T('4 × 8 = 32', font_size=34).to_edge(mn.DOWN).shift(mn.UP * 0.9 + mn.LEFT * 3)
            m2 = T('2 + 3 = 5', font_size=34, color=mn.YELLOW).to_edge(mn.DOWN).shift(mn.UP * 0.9 + mn.RIGHT * 3)
            self.play(mn.Indicate(top[2]), mn.Indicate(top[3]))
            self.play(mn.Write(m1), mn.Indicate(top[5]))
            self.play(mn.Indicate(bot[2]), mn.Indicate(bot[3]))
            self.play(mn.Write(m2), mn.Indicate(bot[5]))
            note = T('上で掛け算すると，下では足し算になるよ', font_size=28).to_edge(mn.DOWN)
            self.play(mn.Write(note))
            self.wait(1.5)
    return LogScene

show_anim(make_scene)
''')

md(r"""
## 3.4 特別な数 e と自然対数 - The special number e and the natural logarithm

AIの世界では，底として $2$ や $10$ ではなく，**$e = 2.71828\ldots$** という特別な数がよく使われます。
$e$ は，例えば次のような「複利」の計算から出てきます。
> In AI, the special number **$e = 2.71828\ldots$** is often used as the base, instead of 2 or 10.
$e$ appears, for example, in this "compound interest" calculation:

1年で100%の利息がつく銀行に1円を預けると，1年後には2円になります。
では，半年ごとに50%ずつ利息がつくなら $(1 + \frac12)^2 = 2.25$ 円，毎月なら $(1 + \frac{1}{12})^{12} \approx 2.61$ 円…と，細かく分けるほど増えます。
でも無限に増えるわけではなく，ある数に近づいていきます。その数が $e$ です。
> If you put 1 yen in a bank that pays 100% interest per year, you have 2 yen after a year.
If it pays 50% every half year, you get $(1 + \frac12)^2 = 2.25$ yen; every month, $(1 + \frac{1}{12})^{12} \approx 2.61$ yen. The finer we split, the more we get.
But it does not grow forever. It gets closer to a certain number. That number is $e$.

### [スライダー] (1 + 1/n)^n が e に近づく - [Slider] (1 + 1/n)^n approaches e
""")

code(r'''
#@title [スライダー] 複利とネイピア数 e { display-mode: "form" }
def plot_e(n=12):
    ks = list(range(1, n + 1))
    vals = [(1 + 1 / k)**k for k in ks]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(ks, vals, 'o-', color='#0284c7', ms=3.5, lw=1.5, label='(1 + 1/n)^n')
    ax.axhline(math.e, color='#ef4444', ls='--', lw=1.5, label=f'e = {math.e:.5f}')
    ax.set_xlabel('分ける回数 n', fontsize=11)
    ax.set_ylabel('(1 + 1/n)^n の値', fontsize=11)
    ax.set_ylim(1.9, 2.85)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.set_title(f'n = {n} のとき値は {(1 + 1 / n)**n:.5f}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_e, n=widgets.IntSlider(value=12, min=1, max=500, step=5, description='分ける回数 n:'));
''')

md(r"""
$e$ を底とする対数を **自然対数** といい，単に $\log x$ と書くことが多いです。Pythonでは `math.log(x)` が自然対数，`math.exp(x)` が $e^x$ です。
$e$ が特別な理由は第4章でわかります($e^x$ は微分しても形が変わらない，という不思議な性質を持っています)。
> The logarithm with base $e$ is called the **natural logarithm**, and is often written simply as $\log x$. In Python, `math.log(x)` is the natural log and `math.exp(x)` is $e^x$.
You will see why $e$ is special in Chapter 4 (it has the strange property that $e^x$ does not change its form when you differentiate it).

## 3.5 損失: -log p - Loss: -log p

いよいよlossです。モデルが正解のTokenにつけた確率を $p$ とすると，1回の予想のlossは
> Now for the loss. If $p$ is the probability the model gave to the correct token, the loss of one prediction is

$$\text{loss} = -\log p$$

です。なぜこの形なのでしょうか？ $p$ を動かして確かめてみましょう。
> Why this form? Let's move $p$ and see.
""")

code(r'''
#@title [スライダー] 損失 loss = -log p { display-mode: "form" }
def plot_loss(p=0.3):
    xs = [i / 1000 for i in range(5, 1001)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(xs, [-math.log(x) for x in xs], lw=2.5, color='#e11d48', label='loss = -log(p)')
    ax.plot([p], [-math.log(p)], 'o', color='#1e293b', ms=8)
    ax.axvline(p, color='#94a3b8', ls=':', lw=1.2)
    ax.set_xlabel('正解文字に与えた確率 p', fontsize=11)
    ax.set_ylabel('loss = -log p', fontsize=11)
    ax.set_title(f'正解の確率 p = {p:.2f} のとき loss = {-math.log(p):.3f}', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.2, 5.5)
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()

interact(plot_loss, p=widgets.FloatSlider(value=0.3, min=0.01, max=1.0, step=0.01, description='正解確率 p:'));
''')

md(r"""
- $p = 1$(自信満々で正解)なら loss $= 0$
- $p$ が小さい(正解をほとんど予想できていない)ほど，lossはどんどん大きくなる
> - If $p = 1$ (correct with full confidence), the loss is 0.
> - The smaller $p$ is (the model barely expected the correct answer), the larger the loss.

さらに，対数の性質のおかげで，**確率の積の -log は，1つ1つの -log の和** になります。
0.0になってしまった積の代わりに和を使えば，計算できるようになります。
比べやすいように，最後に予想の回数で割って **平均** にします。これが0.1で表示されていたlossです。
> Also, thanks to the property of logs, **the -log of a product of probabilities is the sum of each -log**.
If we use the sum instead of the product that became 0.0, we can calculate it.
To make it easy to compare, we divide by the number of predictions to get the **average**. This is the loss shown in 0.1.

$$\text{loss} = \frac{1}{n}\sum_{i=1}^{n} \left(-\log p_i\right)$$

($\sum$ は「全部足す」という記号です。$\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n$)
> ($\sum$ is a symbol meaning "add them all up". $\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n$)
""")

code(r'''
def bigram_loss(names):
    total, n = 0.0, 0
    for name in names:
        ids = tokenize(name)
        for a, b in zip(ids, ids[1:]):
            total += -math.log(next_probs(a)[b])
            n += 1
    return total / n

print('bigramモデルの平均loss:', round(bigram_loss(docs), 4))
print('完全あてずっぽう(72文字ぜんぶ等確率 1/72)のloss:', round(-math.log(1 / vocab_size), 4))
''')

md(r"""
でたらめに予想すると loss は $-\log \frac{1}{72} = \log 72 \approx 4.28$ です。学習を始めたばかりのモデルはほぼでたらめなので，lossはこのあたりから始まります。
数えるだけのbigramモデルは約2.73です。0.1のGPTの最後のloss(直近50回の平均)と比べてみてください。ほとんど同じくらいですね。
実は今回のデータは2〜3文字の短い名前がほとんどなので，直前の1文字を見るだけでもかなり予想できてしまうのです。GPTの本当の強さは第8章で確かめます。
> A random guess gives a loss of $-\log \frac{1}{72} = \log 72 \approx 4.28$. A model that has just started learning is almost random, so its loss starts around here.
The counting bigram model gets about 2.73. Compare it with the last loss of the GPT in 0.1 (the average of the last 50 steps). They are about the same.
In fact, most names in this data are only 2 or 3 characters long, so just looking at the previous character already predicts quite well. We will check the real strength of GPT in Chapter 8.

## 3.6 softmax: スコアを確率に変える - Softmax: turning scores into probabilities

bigramモデルは回数を数えて確率を作りました。でも，これから作るニューラルネットワークは，次の文字ごとに **スコア(logit)** という数を出します。
スコアはマイナスにもなるし，全部足しても1になりません。これを確率に変える必要があります。確率に変えるための条件は3つです。
> The bigram model made probabilities by counting. But the neural networks we will build output a number called a **score (logit)** for each next character.
Scores can be negative, and they do not add up to 1. We need to turn them into probabilities. There are 3 conditions:

1. すべて0以上
2. 全部足すと1
3. スコアが大きいものほど確率も大きい(大小関係が変わらない)
> 1. All are 0 or more.
> 2. They add up to 1.
> 3. A larger score gives a larger probability (the order does not change).

「合計で割る」だけでは，マイナスがあると1.を満たせません。そこで，**先に $e^x$ で全部を正の数に変えてから，合計で割ります**。
$y = e^x$ は必ず正で，右上がりなので，1.と3.を満たします。これが **softmax** です。
> Just "dividing by the total" fails condition 1 when there are negative numbers. So **we first turn everything into positive numbers with $e^x$, and then divide by the total**.
$y = e^x$ is always positive and goes up to the right, so it satisfies 1 and 3. This is **softmax**:

$$p_i = \frac{e^{z_i}}{e^{z_1} + e^{z_2} + \cdots + e^{z_n}}$$
""")

code(r'''
def softmax_numbers(scores, temperature=1.0):
    scores = [s / temperature for s in scores]  # temperature(温度)は次の節で説明するよ
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]    # 最大値を引いておく理由は下で説明するよ
    total = sum(exps)
    return [e / total for e in exps]

scores = [2.0, 1.0, -0.5, 0.3]
probs = softmax_numbers(scores)
print('入力スコア:', scores)
print('変換後の確率:', [round(p, 3) for p in probs], '| 合計 =', round(sum(probs), 6))
''')

md(r"""
**最大値を引く理由**: $e^{1000}$ のような大きな数はコンピューターで表せず，エラーになってしまいます。
でも指数法則から $e^{z - m} = \dfrac{e^z}{e^m}$ なので，全部から同じ $m$ を引いても，分子と分母の両方が $e^m$ で割られるだけで，**確率は変わりません**。
> **Why subtract the maximum**: A huge number like $e^{1000}$ cannot be represented and causes an error.
But by the law of exponents, $e^{z - m} = \dfrac{e^z}{e^m}$, so subtracting the same $m$ from all scores just divides both the numerator and the denominator by $e^m$. **The probabilities do not change.**
""")

code(r'''
try:
    math.exp(1000)
except OverflowError as err:
    print('math.exp(1000) はエラーだよ:', err)

print('最大値を引くので安全に計算できるよ:', softmax_numbers([1000, 999, 998]))
''')

md(r"""
### [アニメーション] softmaxの流れ - [Animation] How softmax works
""")

code(r'''
#@title アニメーション: softmax { display-mode: "form" }
def make_scene(mn, T):
    class SoftmaxScene(mn.Scene):
        def construct(self):
            scores = [2.0, 1.0, -0.5, 0.3]
            exps = [math.exp(s) for s in scores]
            probs = [e / sum(exps) for e in exps]
            labels = ['く', 'ち', 'な', 'と']
            title = T('softmax: スコア → 確率', font_size=36).to_edge(mn.UP)
            self.play(mn.Write(title))
            base = mn.DOWN * 1.5
            axis = mn.Line(mn.LEFT * 4 + base, mn.RIGHT * 4 + base, color=mn.GRAY)
            xs = [-2.7, -0.9, 0.9, 2.7]
            names = mn.VGroup(*[T(l, font_size=30).move_to([x, -2.0, 0]) for x, l in zip(xs, labels)])
            def bars(vals, scale, color):
                g = mn.VGroup()
                for x, v in zip(xs, vals):
                    h = max(abs(v) * scale, 0.02)
                    r = mn.Rectangle(width=1.0, height=h, fill_color=color, fill_opacity=0.8, stroke_width=0)
                    r.move_to([x, base[1] + (h / 2 if v >= 0 else -h / 2), 0])
                    g.add(r)
                return g
            def nums(vals, fmt):
                return mn.VGroup(*[T(fmt.format(v), font_size=24).move_to([x, 2.05, 0]) for x, v in zip(xs, vals)])
            step = T('① スコア (マイナスもあるよ)', font_size=26, color=mn.YELLOW).next_to(title, mn.DOWN, buff=0.2)
            b = bars(scores, 0.9, mn.BLUE); n = nums(scores, '{:.1f}')
            self.play(mn.Create(axis), mn.FadeIn(names), mn.FadeIn(step), mn.GrowFromEdge(b, mn.DOWN), mn.FadeIn(n))
            self.wait(1)
            step2 = T('② e^x で全部プラスに変換するよ', font_size=26, color=mn.YELLOW).next_to(title, mn.DOWN, buff=0.2)
            self.play(mn.Transform(step, step2), mn.Transform(b, bars(exps, 0.45, mn.GREEN)), mn.Transform(n, nums(exps, '{:.2f}')))
            self.wait(1)
            step3 = T('③ 合計で割る → 全部足すとキッカリ1になるよ', font_size=26, color=mn.YELLOW).next_to(title, mn.DOWN, buff=0.2)
            self.play(mn.Transform(step, step3), mn.Transform(b, bars(probs, 5.0, mn.ORANGE)), mn.Transform(n, nums(probs, '{:.2f}')))
            self.wait(2)
    return SoftmaxScene

show_anim(make_scene)
''')

md(r"""
### [スライダー] スコアを動かして確率の変化を見よう - [Slider] Move the scores and watch the probabilities
""")

code(r'''
#@title [スライダー] softmax { display-mode: "form" }
def plot_softmax(く=2.0, ち=1.0, な=-0.5, と=0.3):
    scores = [く, ち, な, と]
    probs = softmax_numbers(scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax1.bar(['く', 'ち', 'な', 'と'], scores, color='#3b82f6', edgecolor='#1d4ed8')
    ax1.axhline(0, color='#64748b', lw=1.0)
    ax1.set_ylim(-5.5, 5.5)
    ax1.set_title('スコア (logit)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('スコア値')

    bars2 = ax2.bar(['く', 'ち', 'な', 'と'], probs, color='#f97316', edgecolor='#ea580c')
    ax2.set_ylim(0, 1.1)
    ax2.set_title('softmax後の確率分布 (合計 = 1.0)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('確率')
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f'{h:.2f}', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.show()

sl = lambda v: widgets.FloatSlider(value=v, min=-5, max=5, step=0.1)
interact(plot_softmax, く=sl(2.0), ち=sl(1.0), な=sl(-0.5), と=sl(0.3));
''')

md(r"""
スコアを全部同じだけ増やしても確率が変わらないことや，1つだけ大きくするとその確率が急に1に近づくことを確かめてみましょう。
> Check that adding the same amount to all scores does not change the probabilities, and that making one score large quickly pushes its probability toward 1.

## 3.7 temperature: 選び方の自由さ - Temperature: how freely to choose

0.1のコードには `temperature = 0.5` という設定がありました。softmaxの前に，スコアを **temperature $T$ で割る** のです。
> The code in 0.1 had the setting `temperature = 0.5`. Before softmax, we **divide the scores by the temperature $T$**:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

- $T$ が小さい → スコアの差が大きくなり，一番スコアが高いものばかり選ぶ(まじめ・同じ答えばかり)
- $T = 1$ → そのままの確率
- $T$ が大きい → スコアの差が小さくなり，どれも同じくらいの確率になる(自由・でたらめ)
> - Small $T$ → the differences between scores grow, so it almost always picks the top one (serious, always the same answer).
> - $T = 1$ → the original probabilities.
> - Large $T$ → the differences shrink, so everything gets similar probabilities (free, random).

bigramモデルでも試せます。確率 $p$ の対数 $\log p$ をスコアとして使えば，$T = 1$ のときに元の確率に戻ります。
> We can try it with the bigram model. If we use $\log p$ as the score, we get back the original probabilities when $T = 1$.

### [スライダー] temperatureを変えて名前を作ろう - [Slider] Make names while changing the temperature
""")

code(r'''
#@title [スライダー] temperature (生成の温度感) { display-mode: "form" }
log_counts = [[math.log(c) if c > 0 else float('-inf') for c in row] for row in counts]

def bigram_generate_t(temperature, rng):
    token, out = BOS, []
    while len(out) < 10:
        token = rng.choices(range(vocab_size), weights=softmax_numbers(log_counts[token], temperature))[0]
        if token == BOS:
            break
        out.append(itos[token])
    return ''.join(out)

def show_temperature(temperature=1.0):
    p = softmax_numbers(log_counts[BOS], temperature)
    top = sorted(range(vocab_size), key=lambda b: -p[b])[:15]
    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.bar([tok_str(b) for b in top], [p[b] for b in top], color='#8b5cf6', edgecolor='#6d28d9')
    ax.set_ylim(0, max(0.2, max([p[b] for b in top]) * 1.15))
    ax.set_title(f'T = {temperature:.1f} のときの「最初の文字」の確率(上位15個)', fontsize=12, fontweight='bold')
    ax.set_ylabel('確率')
    plt.tight_layout()
    plt.show()
    rng = random.Random(0)
    print('作った名前:', ' '.join(bigram_generate_t(temperature, rng) for _ in range(12)))

interact(show_temperature, temperature=widgets.FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1, description='温度 T:', continuous_update=False));
''')

md(r"""
ChatGPTなどでも，同じ質問をしたのに毎回少し違う答えが返ってくるのは，このようにくじ引きで次のTokenを選んでいるからです。
> The reason ChatGPT gives slightly different answers to the same question is that it chooses the next token by drawing lots like this.

## 3.8 0.1のコードとの対応 - Matching with the code in 0.1

```python
def softmax(logits):                                  # 3.6 softmax関数だよ
    max_val = max(val.data for val in logits)         # 数値安定化のため最大値を引くよ
    exps = [(val - max_val).exp() for val in logits]  # e^x を計算するよ
    total = sum(exps)
    return [e / total for e in exps]                  # 合計で割って確率化するよ

loss_t = -probs[target_id].log()                      # 3.5 1文字の損失 loss = -log p だよ
loss = (1 / n) * sum(losses)                          # 3.5 全ステップの平均損失だよ
probs = softmax([l / temperature for l in logits])    # 3.7 temperatureで割ってからサンプリングするよ
```

0.1のコードの `softmax` は `val.data` や `.exp()` のように書かれていますが，やっていることは `softmax_numbers` と同じです(この書き方の理由は第5章でわかります)。
> The `softmax` in 0.1 is written with `val.data` and `.exp()`, but it does the same thing as `softmax_numbers` (you will see why it is written this way in Chapter 5).

## 練習問題 - Exercises

1. $\log_2 \frac{1}{8}$ の値を答えましょう。
2. 3つのスコアがすべて同じとき，softmaxの確率はどうなるでしょうか？
3. 正解の確率が $p = 0.5$ のとき，lossはいくつになるでしょうか？(`math.log` で計算してみましょう)

<details><summary>答え - Answers</summary>

1. $-3$($2^{-3} = \frac{1}{8}$ だから)
2. すべて $\frac{1}{3}$
3. $-\log 0.5 = \log 2 \approx 0.693$
</details>
""")
