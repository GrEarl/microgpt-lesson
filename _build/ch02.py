from nbbuild import md, code

md(r"""
---
# 第2章 数えて名前を作る: bigramモデル - Chapter 2: Making names by counting

いきなりGPTを作る前に，**数えるだけ** で名前を作るモデルを作ってみましょう。
これは0.1のコードには直接出てきませんが，「次の文字を確率で選ぶ」というLLMの一番大事な考え方が，とてもシンプルな形で入っています。
> Before building a GPT, let's build a model that makes names **just by counting**.
It is not in the code in 0.1, but it contains the most important idea of LLMs, "choose the next character by probability", in a very simple form.

**この章で使う数学**: 度数と相対度数(数学I「データの分析」)，確率・条件付き確率(数学A「場合の数と確率」)
> **Math in this chapter**: frequency and relative frequency (Math I), probability and conditional probability (Math A)

## 2.1 次に来る文字を数える - Counting which character comes next

名前を作るときに一番知りたいのは，「ある文字の次に，どの文字が来やすいか」です。
例えば「さ」の次には「く」「ち」「と」などが来そうですね。それなら，学習データの中で実際に数えてしまえばいいのです。
> What we most want to know when making names is "which character is likely to come after a certain character".
For example, after "さ", characters like "く", "ち" and "と" seem likely. Then we can simply count them in the training data.

隣り合う2つのTokenの組を **bigram(バイグラム)** と呼びます。「さくら」なら，次の4組です。
> A pair of two neighboring tokens is called a **bigram**. For "さくら", there are these 4 pairs:
""")

code(r'''
ids = tokenize('さくら')
for a, b in zip(ids, ids[1:]):  # zip(ids, ids[1:]) で「1つ目と2つ目」「2つ目と3つ目」…の組が作れるよ
    print(f'{tok_str(a):>3} → {tok_str(b)}')
''')

md(r"""
すべての名前について，すべての組を数えて，**72×72の表** に記録します。
`counts[a][b]` は「Token `a` の次にToken `b` が来た回数」です。
> For every name, we count every pair and record it in a **72×72 table**.
`counts[a][b]` is "the number of times token `b` came right after token `a`".
""")

code(r'''
counts = [[0] * vocab_size for _ in range(vocab_size)]
for name in docs:
    ids = tokenize(name)
    for a, b in zip(ids, ids[1:]):
        counts[a][b] += 1

a = stoi['さ']
top = sorted(range(vocab_size), key=lambda b: -counts[a][b])[:8]
print('「さ」の次に来た文字ベスト8:')
for b in top:
    print(f'  {tok_str(b):>3}: {counts[a][b]}回')
''')

md(r"""
### [アニメーション] 数え上げの様子 - [Animation] How counting works
""")

code(r'''
#@title アニメーション: bigramを数える { display-mode: "form" }
def make_scene(mn, T):
    class BigramCount(mn.Scene):
        def construct(self):
            title = T('bigram を数える', font_size=36).to_edge(mn.UP)
            self.play(mn.Write(title))
            table, rows = {}, {}
            start = mn.UP * 2 + mn.RIGHT * 1.5
            def row_mob(pair, n):
                return T(f'{pair[0]} → {pair[1]} : {n}回', font_size=28)
            for name in ['さくら', 'さき', 'ゆき']:
                word = ['BOS'] + list(name) + ['BOS']
                boxes = mn.VGroup(*[mn.VGroup(mn.Square(1.0, color=mn.BLUE), T(w, font_size=24 if w == 'BOS' else 36)) for w in word]).arrange(mn.RIGHT, buff=0.15).to_edge(mn.LEFT).shift(mn.RIGHT * 0.3)
                for g in boxes: g[1].move_to(g[0])
                self.play(mn.FadeIn(boxes))
                for i in range(len(word) - 1):
                    frame = mn.SurroundingRectangle(mn.VGroup(boxes[i], boxes[i + 1]), color=mn.YELLOW, buff=0.08)
                    pair = (word[i], word[i + 1])
                    self.play(mn.Create(frame), run_time=0.3)
                    table[pair] = table.get(pair, 0) + 1
                    new = row_mob(pair, table[pair])
                    if pair in rows:
                        new.move_to(rows[pair], aligned_edge=mn.LEFT).set_color(mn.YELLOW)
                        self.play(mn.Transform(rows[pair], new), run_time=0.5)
                    else:
                        new.move_to(start + mn.DOWN * 0.6 * len(rows), aligned_edge=mn.LEFT)
                        rows[pair] = new
                        self.play(mn.FadeIn(new, shift=mn.LEFT * 0.3), run_time=0.4)
                    self.play(mn.FadeOut(frame), run_time=0.2)
                self.play(mn.FadeOut(boxes))
            note = T('同じ組が出てきたら，回数が増えていくよ', font_size=26).to_edge(mn.DOWN)
            self.play(mn.Write(note))
            self.wait(1.5)
    return BigramCount

show_anim(make_scene)
''')

md(r"""
表全体を色の濃さで見てみましょう(72×72だと細かすぎるので，よく出てくる25文字とBOSだけにしています)。
**行が「前の文字」，列が「次の文字」** です。色が明るいところほど，よく出てくる組です。
> Let's look at the whole table by color (72×72 is too detailed, so we show only the 25 most frequent characters and BOS).
**Rows are the "previous character" and columns are the "next character"**. Brighter cells are more frequent pairs.
""")

code(r'''
#@title 数え上げ表のヒートマップ { display-mode: "form" }
freq = sorted(range(len(uchars)), key=lambda i: -sum(counts[i]))[:25]
show = [BOS] + sorted(freq)
grid = [[counts[a][b] for b in show] for a in show]
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(grid, cmap='viridis', aspect='auto')
ax.set_xticks(range(len(show)), [tok_str(i) for i in show], fontsize=9)
ax.set_yticks(range(len(show)), [tok_str(i) for i in show], fontsize=9)
ax.set_xlabel('次の文字 (next)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel('前の文字 (previous)', fontsize=11, fontweight='bold', labelpad=8)
ax.set_title('文字のつながり出現回数 (bigram ヒートマップ)', fontsize=13, fontweight='bold', pad=12)
cb = fig.colorbar(im, ax=ax, label='出現回数')
cb.ax.tick_params(labelsize=9)
plt.tight_layout()
plt.show()
''')

md(r"""
BOSの行(一番上)は「名前の最初の文字」，BOSの列(一番左)は「名前の最後の文字」を表しています。
「こ」「み」「な」などで終わる名前が多いことが，一番左の列からわかりますね。
> The BOS row (top) shows "the first character of names", and the BOS column (left) shows "the last character of names".
From the leftmost column, you can see that many names end with "こ", "み", "な" and so on.

## 2.2 回数から確率へ - From counts to probabilities

数学Iの「データの分析」では，度数を全体の合計で割ったものを **相対度数** と呼びました。
相対度数は0以上1以下で，全部足すと1になります。これはそのまま **確率** として使えます。
> In Math I, a frequency divided by the total is called the **relative frequency**.
It is between 0 and 1, and all of them add up to 1. We can use it directly as a **probability**.

ただし，ここで知りたいのは「前の文字が『さ』だったときに，次が『く』である確率」です。
これは数学Aの **条件付き確率** です。
> But what we want is "the probability that the next character is 'く' when the previous one was 'さ'".
This is a **conditional probability** from Math A.

$$P(\text{次が「く」} \mid \text{前が「さ」}) = \frac{n(\text{「さ」→「く」})}{n(\text{「さ」→ 何か})}$$

数学Aの記号で書くと $P_A(B) = \dfrac{n(A \cap B)}{n(A)}$ です。$A$ は「前が『さ』」，$B$ は「次が『く』」という事象です。
> In Math A notation, this is $P_A(B) = \dfrac{n(A \cap B)}{n(A)}$, where $A$ is "the previous one is 'さ'" and $B$ is "the next one is 'く'".
""")

code(r'''
def next_probs(a):
    total = sum(counts[a])
    return [c / total for c in counts[a]]

p = next_probs(stoi['さ'])
print('P(次が「く」| 前が「さ」) =', round(p[stoi['く']], 4))
print('全文字の確率の合計 =', round(sum(p), 6), '(きっちり 1.0 になるよ)')
''')

md(r"""
### [スライダー] 前の文字を選んで，次の文字の確率を見よう - [Slider] Choose the previous character and see the next-character probabilities
""")

code(r'''
#@title [スライダー] 条件付き確率の棒グラフ { display-mode: "form" }
def plot_next(prev='さ'):
    a = BOS if prev == 'BOS' else stoi[prev]
    p = next_probs(a)
    top = sorted(range(vocab_size), key=lambda b: -p[b])[:15]
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    bars = ax.bar([tok_str(b) for b in top], [p[b] for b in top], color='#3b82f6', edgecolor='#1d4ed8', width=0.6)
    ax.set_ylim(0, max(0.2, max([p[b] for b in top]) * 1.15))
    ax.set_title(f'前の文字が「{prev}」のとき，次の文字の確率(上位15個)', fontsize=12, fontweight='bold')
    ax.set_ylabel('確率', fontsize=10)
    for bar in bars:
        h = bar.get_height()
        if h > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=8.5, color='#1e293b')
    plt.tight_layout()
    plt.show()

interact(plot_next, prev=widgets.Dropdown(options=['BOS'] + uchars, value='さ', description='前の文字:'));
''')

md(r"""
## 2.3 確率にしたがってくじを引く - Drawing lots according to probabilities

確率がわかったら，その確率にしたがって次の文字を選びます。
これは「当たりやすさが違うくじ」を引くのと同じです。例えば「く」のくじが30枚，「ち」のくじが10枚入った箱から1枚引くようなものです。
Pythonでは `random.choices` がこれをやってくれます(`weights` に回数をそのまま渡しても大丈夫です)。
> Once we know the probabilities, we choose the next character according to them.
This is like drawing lots where each lot has a different chance. For example, drawing one ticket from a box with 30 "く" tickets and 10 "ち" tickets.
In Python, `random.choices` does this (you can pass the counts directly as `weights`).

### [スライダー] くじを引く回数を増やすと？ - [Slider] What happens when we draw more times?

引く回数を増やすほど，実際に出た割合(オレンジ)が確率(青)に近づいていく様子を確かめてみましょう。
> Check that the more times you draw, the closer the actual ratio (orange) gets to the probability (blue).
""")

code(r'''
#@title [スライダー] くじ引きのシミュレーション { display-mode: "form" }
def simulate(prev='さ', draws=10):
    a = BOS if prev == 'BOS' else stoi[prev]
    p = next_probs(a)
    rng = random.Random(0)
    got = rng.choices(range(vocab_size), weights=counts[a], k=draws)
    top = sorted(range(vocab_size), key=lambda b: -p[b])[:10]
    x = range(len(top))
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    ax.bar([i - 0.2 for i in x], [p[b] for b in top], width=0.4, label='理論確率', color='#3b82f6', edgecolor='#1d4ed8')
    ax.bar([i + 0.2 for i in x], [got.count(b) / draws for b in top], width=0.4, label=f'実際の割合 ({draws}回引いた結果)', color='#f97316', edgecolor='#ea580c')
    ax.set_xticks(list(x), [tok_str(b) for b in top])
    ax.set_ylim(0, max(0.2, max([p[b] for b in top]) * 1.25))
    ax.set_title(f'「{prev}」の次の文字のくじ引き実験 (サンプル数: {draws})', fontsize=12, fontweight='bold')
    ax.set_ylabel('確率 / 割合', fontsize=10)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.show()

interact(simulate,
         prev=widgets.Dropdown(options=['BOS'] + uchars, value='さ', description='前の文字:'),
         draws=widgets.IntSlider(value=10, min=1, max=2000, step=10, description='引く回数:'));
''')

md(r"""
## 2.4 実装: 数えるだけで名前を作る - Implementation: making names just by counting

BOSからスタートして，「次の文字を確率で選ぶ」をBOSが出るまで繰り返せば，名前ができあがります。
> Start from BOS and repeat "choose the next character by probability" until BOS comes out. Then you have a name.
""")

code(r'''
def bigram_generate():
    token, out = BOS, []
    while True:
        token = random.choices(range(vocab_size), weights=counts[token])[0]
        if token == BOS:
            return ''.join(out)
        out.append(itos[token])

random.seed(1)
print('--- bigramモデルが作った名前 15選 ---')
for i in range(15):
    name = bigram_generate()
    note = '学習データにある名前' if name in docs else '新しい名前！'
    print(f'sample {i+1:2d}: {name:<6} ({note})')
''')

md(r"""
微分もニューラルネットワークも使わずに，それらしい名前ができました！
これが **言語モデル** の一番シンプルな形です。LLMも本質的には「次のTokenの確率を求めて，くじを引く」ことを繰り返しています。
> Without any calculus or neural networks, we made plausible names!
This is the simplest form of a **language model**. In essence, LLMs also repeat "find the probabilities of the next token and draw a lot".

## 2.5 数えるだけのモデルの限界 - The limits of counting

でも，このモデルには大きな弱点があります。**直前の1文字しか見ていない** のです。
例えば「こ」の次にBOSが来る(=名前が終わる)確率はとても高いので，「こ」1文字だけの名前を作ってしまうことがあります。
学習データの名前の長さと，このモデルが作った名前の長さを比べてみましょう。
> But this model has a big weakness: **it only looks at the one previous character**.
For example, the probability that BOS comes after "こ" (the name ends) is very high, so it sometimes makes a one-character name "こ".
Let's compare the lengths of names in the training data with the lengths of names this model makes.
""")

code(r'''
#@title 名前の長さの比較 { display-mode: "form" }
random.seed(2)
made = [bigram_generate() for _ in range(3000)]
lengths = range(1, 9)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.bar([l - 0.2 for l in lengths], [sum(len(n) == l for n in docs) / len(docs) for l in lengths], width=0.4, label='学習データ (正解)', color='#10b981', edgecolor='#059669')
ax.bar([l + 0.2 for l in lengths], [sum(len(n) == l for n in made) / len(made) for l in lengths], width=0.4, label='bigramモデルの出力', color='#6366f1', edgecolor='#4f46e5')
ax.set_xlabel('名前の文字数', fontsize=11, fontweight='bold')
ax.set_ylabel('割合', fontsize=10)
ax.set_title('名前の文字数分布の比較', fontsize=12, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.show()
print('生成された1文字ネームの例:', sorted({n for n in made if len(n) == 1}))
''')

md(r"""
学習データには1文字の名前が1つもないのに，bigramモデルは1文字の名前や長すぎる名前を作ってしまいます。
「今は何文字目か」「前に何の文字があったか」を知らないからです。
> The training data has no one-character names at all, but the bigram model makes one-character names and names that are too long.
This is because it does not know "which position we are at" or "what characters came before".

もっと前の文字まで見たいなら，3文字の組(trigram)，4文字の組…と数えればいいように思えます。
でも，組の種類は $72^2 = 5184$，$72^3 = 373248$，… と爆発的に増えて，ほとんどの組は一度も出てこなくなってしまいます。
> If we want to look further back, it seems we could count 3-character groups (trigrams), 4-character groups, and so on.
But the number of possible groups explodes: $72^2 = 5184$, $72^3 = 373248$, ... and most groups never appear even once.

そこで，**数える代わりに「学習」で確率を求める** 方法が必要になります。これが次の章からのテーマです。
その第一歩として，次の章では「モデルの予想がどれくらい良いか」を測る方法を学びます。
> So we need a way to **find probabilities by "learning" instead of counting**. This is the theme from the next chapter.
As a first step, the next chapter shows how to measure "how good the model's predictions are".

## 練習問題 - Exercises

1. [スライダー] で前の文字を「BOS」にすると，何がわかるでしょうか？
2. 「っ」の次に来る文字には，どんな特徴があるでしょうか？ スライダーで確かめてみましょう。
3. $P(\text{次が「こ」} \mid \text{前が「み」})$ を，`counts` を使って計算してみましょう。

<details><summary>答え - Answers</summary>

1. 名前の最初の文字の確率がわかります(「あ」「ゆ」「ち」などが上位に来ます)。
2. 「か」「こ」「ぽ」の3つだけです。小さい「っ」の後には，か行やぱ行のように，つまる音の後に来られる音しか来ません。ただし「っ」はデータに4回しか出てこないので，この確率はあまり当てになりません。
3. `counts[stoi['み']][stoi['こ']] / sum(counts[stoi['み']])` で計算できます。
</details>
""")
