from nbbuild import md, code

md(r"""
---
# 第8章 前の文字に注目する: Attention - Chapter 8: Paying attention to previous characters

ここまでのモデルは，どれも **直前の1文字しか見ていません** でした。
この章では，**それまでのすべての文字の中から，今どれに注目すべきかを自分で決める** 仕組み，**Attention(アテンション，注意機構)** を作ります。
AttentionはGPTの心臓部であり，ChatGPTなどが長い文章を理解できるのもこの仕組みのおかげです。
> All our models so far **only looked at the one previous character**.
In this chapter, we build **Attention**, a mechanism that **decides by itself which of all the previous characters to focus on**.
Attention is the heart of GPT, and it is thanks to this mechanism that ChatGPT and others can understand long texts.

**この章で使う数学**: 内積(第6章)，softmax(第3章)，加重平均(数学I「データの分析」の平均の発展)
> **Math in this chapter**: dot product (Chapter 6), softmax (Chapter 3), weighted average (an extension of the average in Math I)

## 8.1 前の文字をすべて使いたい - We want to use all the previous characters

「ゆ」の次に来る文字を予想するとき，それが名前の1文字目の「ゆ」なのか(「ゆき」「ゆみ」…)，「まゆ」「あゆ」のように2文字目の「ゆ」なのかで，答えは変わるはずです(2文字目なら，そこで名前が終わることも多いでしょう)。
前の文字を全部使いたいのですが，名前によって文字数が違います。文字数が変わっても使える方法が必要です。
> When predicting the character after "ゆ", the answer should depend on whether it is the first character of the name ("ゆき", "ゆみ", ...) or the second, as in "まゆ" or "あゆ" (then the name often ends there).
We want to use all the previous characters, but names have different lengths. We need a method that works for any length.

アイデアは，**前の文字のベクトルの「平均」をとる** ことです。平均なら，何個あっても1つのベクトルにまとめられます。
ただし，ただの平均ではなく，**大事な文字ほど重くした平均** にします。
> The idea is to **take the "average" of the vectors of the previous characters**. An average can combine any number of vectors into one.
But instead of a plain average, we use **an average where more important characters count more**.

## 8.2 加重平均 - Weighted average

小テストを3回受けて，60点，80点，90点だったとします。普通の平均は $\frac{60 + 80 + 90}{3} \approx 76.7$ 点です。
これは，3回それぞれに $\frac13$ ずつの **重み** をつけて足したものと同じです。
> Suppose you took 3 quizzes and scored 60, 80 and 90. The plain average is $\frac{60 + 80 + 90}{3} \approx 76.7$.
This is the same as giving each quiz a **weight** of $\frac13$ and adding them up.

もし「最近のテストほど大事」として，重みを $0.2,\ 0.3,\ 0.5$ にすると
> If "more recent quizzes are more important" and we use weights $0.2,\ 0.3,\ 0.5$:

$$0.2 \times 60 + 0.3 \times 80 + 0.5 \times 90 = 81 \text{ 点}$$

これを **加重平均** といいます。**重みはすべて0以上で，合計が1** になるようにします。
…この条件，どこかで見ましたね？ そう，**softmaxの出力** です(第3章)。
> This is called a **weighted average**. **All weights are 0 or more, and they add up to 1.**
...Have you seen these conditions before? Yes, they are **the output of softmax** (Chapter 3).

## 8.3 Query・Key・Value: 図書館で本を探す - Query, Key, Value: finding books in a library

では，重み(どの文字にどれくらい注目するか)はどうやって決めるのでしょう？ Attentionは，図書館で本を探すのに似た方法を使います。
> Then how do we decide the weights (how much to focus on each character)? Attention uses a method similar to finding books in a library.

| 名前 | 図書館でいうと | 役割 |
|---|---|---|
| **Query(クエリ)** $q$ | 探している本の条件(検索ワード) | 今の文字が「どんな情報を探しているか」 |
| **Key(キー)** $k$ | 本の背表紙のラベル | それぞれの文字が「どんな情報を持っているか」の見出し |
| **Value(バリュー)** $v$ | 本の中身 | それぞれの文字が実際に渡す情報 |

手順は次のとおりです。
> The steps are:

1. 各文字のベクトル $x$ から，線形変換(第6章)で $q$，$k$，$v$ の3つのベクトルを作る
2. 今の文字の $q$ と，前の各文字の $k$ の **内積** を計算する。内積は「似ている度合い」なので(第6章)，探している条件に合う文字ほどスコアが大きくなる
3. スコアを **softmax** で重みに変える(合計1)
4. 各文字の $v$ を，その重みで **加重平均** する
> 1. From each character's vector $x$, make three vectors $q$, $k$ and $v$ with linear transformations (Chapter 6).
> 2. Calculate the **dot product** of the current character's $q$ with each previous character's $k$. Since the dot product means "how similar" (Chapter 6), characters that match what we are looking for get larger scores.
> 3. Turn the scores into weights with **softmax** (they add up to 1).
> 4. Take the **weighted average** of each character's $v$ with those weights.

$$\text{出力} = \sum_{t} (\text{重み})_t \, v_t, \qquad (\text{重み})_t = \text{softmax}\!\left(\frac{q \cdot k_t}{\sqrt{d}}\right)$$

$\sqrt{d}$ で割る理由は8.5で説明します。
> We explain why we divide by $\sqrt{d}$ in 8.5.

### [アニメーション] Attentionの流れ - [Animation] How Attention works
""")

code(r'''
#@title アニメーション: Attention { display-mode: "form" }
def make_scene(mn, T):
    class AttentionScene(mn.Scene):
        def construct(self):
            title = T('Attention: どの文字に注目する？', font_size=34).to_edge(mn.UP)
            self.play(mn.Write(title))
            words = ['BOS', 'は', 'な']
            weights = [0.1, 0.65, 0.25]
            vals = [[0.2, -0.5, 0.1], [0.9, 0.4, -0.3], [-0.2, 0.7, 0.5]]
            boxes = mn.VGroup(*[mn.VGroup(mn.RoundedRectangle(width=1.4, height=0.9, corner_radius=0.15, color=mn.BLUE), T(w, font_size=26 if w == 'BOS' else 36)) for w in words]).arrange(mn.RIGHT, buff=1.2).shift(mn.UP * 1.6 + mn.LEFT * 2.8)
            for g in boxes: g[1].move_to(g[0])
            self.play(mn.FadeIn(boxes))
            q = mn.VGroup(T('Query: 「な」の次に', font_size=24, color=mn.YELLOW), T('来る文字を知りたい', font_size=24, color=mn.YELLOW)).arrange(mn.DOWN, aligned_edge=mn.LEFT).next_to(boxes[-1], mn.RIGHT, buff=0.4)
            self.play(mn.Indicate(boxes[-1], color=mn.YELLOW), mn.FadeIn(q))
            arcs = []
            for b, w in zip(boxes, weights):
                if b is boxes[-1]:
                    arc = mn.Circle(0.25, color=mn.YELLOW, stroke_width=2 + 12 * w).next_to(b, mn.DOWN, buff=0.1)
                else:
                    arc = mn.CurvedArrow(boxes[-1].get_bottom() + mn.DOWN * 0.1, b.get_bottom() + mn.DOWN * 0.1, angle=-mn.PI / 2, color=mn.YELLOW, stroke_width=2 + 12 * w)
                arcs.append(arc)
            lab = T('q と各 k の内積 → softmax → 重み', font_size=24).to_edge(mn.DOWN)
            self.play(*[mn.Create(a) for a in arcs], mn.FadeIn(lab))
            wl = mn.VGroup(*[T(f'{w:.2f}', font_size=28, color=mn.YELLOW).next_to(b, mn.UP, buff=0.15) for b, w in zip(boxes, weights)])
            self.play(mn.FadeIn(wl))
            def bars(v, color, scale=1.0):
                g = mn.VGroup()
                for i, e in enumerate(v):
                    h = abs(e) * scale + 0.02
                    r = mn.Rectangle(width=0.25, height=h, fill_color=color, fill_opacity=0.9, stroke_width=0)
                    r.move_to([i * 0.32, (h / 2 if e >= 0 else -h / 2), 0])
                    g.add(r)
                return g
            vb = [bars(v, mn.GREEN).move_to(b.get_center() + mn.DOWN * 3.0) for v, b in zip(vals, boxes)]
            vlabs = mn.VGroup(*[T(f'Value({w})', font_size=20, color=mn.GREEN).next_to(v, mn.DOWN, buff=0.15) for v, w in zip(vb, words)])
            self.play(*[mn.FadeIn(v) for v in vb], mn.FadeIn(vlabs))
            out = [sum(w * v[j] for w, v in zip(weights, vals)) for j in range(3)]
            ob = bars(out, mn.ORANGE).move_to(mn.RIGHT * 4.3 + mn.DOWN * 1.4)
            olab = T('出力 = 重みつき平均', font_size=24, color=mn.ORANGE).next_to(ob, mn.UP, buff=0.4)
            calc = T('0.10×BOS + 0.65×は + 0.25×な', font_size=20, color=mn.ORANGE).next_to(ob, mn.DOWN, buff=0.4)
            copies = [v.copy() for v in vb]
            self.play(*[c.animate.move_to(ob).set_opacity(0.4) for c in copies], mn.FadeIn(olab), run_time=1.5)
            self.play(*[mn.FadeOut(c) for c in copies], mn.FadeIn(ob), mn.FadeIn(calc))
            self.wait(2)
    return AttentionScene

show_anim(make_scene)
''')

md(r"""
### 小さな例で計算してみよう - Let's calculate a small example

ベクトルが2成分の小さな例で，Attentionの計算を1ステップずつやってみます。
> Let's do the Attention calculation step by step with a small example where vectors have 2 components.
""")

code(r'''
q = [1.0, 0.5]                                # 今の文字のQueryだよ
keys_toy = [[0.2, -1.0], [1.0, 0.8], [-0.5, 0.3]]  # 3つの文字のKeyだよ
values_toy = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]  # 3つの文字のValueだよ
d = len(q)

scores = [sum(qi * ki for qi, ki in zip(q, k)) / d**0.5 for k in keys_toy]  # 2. 内積(÷√d)
weights = softmax_numbers(scores)                                            # 3. softmax
output = [sum(w * v[j] for w, v in zip(weights, values_toy)) for j in range(d)]  # 4. 加重平均
print('スコア:', [round(s, 3) for s in scores])
print('重み  :', [round(w, 3) for w in weights], ' 合計', round(sum(weights), 3))
print('出力  :', [round(o, 3) for o in output])
''')

md(r"""
$q$ と一番似ている(同じ向きの)2つ目のKeyに，一番大きな重みがつきました。出力も2つ目のValue $(0, 1)$ に近くなっています。
> The second key, which is the most similar to $q$ (points in the same direction), got the largest weight. The output is also close to the second value $(0, 1)$.

### [スライダー] Queryの向きを変えて，注目先が変わる様子を見よう - [Slider] Change the query's direction and watch the focus change

Queryの向きを回すと，一番似ているKeyが変わり，重みが移っていきます。Queryを長くすると内積が大きくなり，softmaxの結果がはっきり(1つに集中)します。
> As you rotate the query, the most similar key changes and the weights shift. Making the query longer makes the dot products larger, so the softmax result becomes sharper (focused on one).
""")

code(r'''
#@title [スライダー] QueryとKeyの内積 → 重み { display-mode: "form" }
def plot_query(angle=30, length=1.5):
    qv = [length * math.cos(math.radians(angle)), length * math.sin(math.radians(angle))]
    ks = [[1.5, 0.3], [-0.4, 1.4], [-1.2, -0.9]]
    names, colors = ['Key 1', 'Key 2', 'Key 3'], ['#3b82f6', '#10b981', '#8b5cf6']
    sc = [sum(a * b for a, b in zip(qv, k)) / math.sqrt(2) for k in ks]
    w = softmax_numbers(sc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    for k, n, c in zip(ks, names, colors):
        ax1.annotate('', xy=k, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=c, lw=2.5))
        ax1.text(k[0] * 1.15, k[1] * 1.15, n, color=c, fontweight='bold')
    ax1.annotate('', xy=qv, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='#ef4444', lw=3.2))
    ax1.set_xlim(-4, 4); ax1.set_ylim(-4, 4); ax1.set_aspect('equal')
    ax1.set_title('Query と Key のベクトル配置', fontsize=12, fontweight='bold')

    bars = ax2.bar(names, w, color=colors, edgecolor='#1e293b', width=0.55)
    ax2.set_ylim(0, 1.15)
    ax2.set_title('各Keyへの注目度 (softmax後の重み)', fontsize=12, fontweight='bold')
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f'{h:.2f}', ha='center', fontsize=9.5, fontweight='bold')
    plt.tight_layout()
    plt.show()

interact(plot_query, angle=widgets.IntSlider(value=30, min=0, max=360, step=5, description='向き:'),
         length=widgets.FloatSlider(value=1.5, min=0.1, max=4, step=0.1, description='長さ:'));
''')

md(r"""
## 8.4 何文字目かを教える: 位置埋め込み - Telling the position: position embeddings

Attentionは加重平均なので，**文字の順番を区別できません**。「はな」と「なは」で，同じ文字の集まりなら同じ結果になってしまいます。
そこで，第6章で紹介した **位置埋め込み `wpe`** を使います。「0文字目」「1文字目」…にもそれぞれベクトルを用意して，文字のベクトルに **足し算** します。
こうすれば，同じ「な」でも，1文字目の「な」と2文字目の「な」は違うベクトルになります。
> Because Attention is a weighted average, **it cannot tell the order of characters**. "はな" and "なは" would give the same result, since they have the same characters.
So we use the **position embedding `wpe`** introduced in Chapter 6. We prepare a vector for "position 0", "position 1", ..., and **add** it to the character's vector.
This way, the same "な" gets a different vector at position 1 and at position 2.

```python
tok_emb = state_dict['wte'][token_id]  # 文字のベクトルだよ
pos_emb = state_dict['wpe'][pos_id]    # 位置のベクトルだよ
x = [t + p for t, p in zip(tok_emb, pos_emb)]  # 足し算するよ
```

## 8.5 √d で割る理由 - Why divide by √d

ベクトルの成分が多いほど，内積は多くの数の足し算になるので，値が大きくなりがちです。
スコアが大きすぎるとsoftmaxの結果が1つに集中しすぎてしまいます(スライダーでQueryを長くしたときと同じです)。
そこで，成分の数 $d$ の平方根 $\sqrt{d}$ で割って，スコアの大きさをそろえます。
これは第3章の **temperature を $T = \sqrt{d}$ にしている** のと同じことです。
> The more components a vector has, the more numbers are added in the dot product, so the value tends to be larger.
If the scores are too large, the softmax result concentrates too much on one (the same as making the query longer with the slider).
So we divide by $\sqrt{d}$, the square root of the number of components $d$, to keep the scores at a similar size.
This is the same as **setting the temperature from Chapter 3 to $T = \sqrt{d}$**.

## 8.6 未来の文字は見ない - Do not look at future characters

学習のとき，モデルは「次の文字」を予想します。もし次の文字そのものを見られたら，カンニングになってしまいます。
0.1のコードでは，1文字ずつ順番に処理しながら，**それまでに出てきた文字のKeyとValueだけ** を `keys` と `values` に追加していきます。
だから，Attentionは自然に「今と過去の文字」だけを見るようになっています。
> During training, the model predicts "the next character". If it could see the next character itself, that would be cheating.
In 0.1, the characters are processed one by one, and **only the keys and values of the characters seen so far** are added to `keys` and `values`.
So Attention naturally looks only at "the current and past characters".

## 8.7 マルチヘッド: いくつもの視点で注目する - Multi-head: attention from several viewpoints

0.1のコードでは `n_head = 4` で，16成分のベクトルを4成分ずつ **4つに分けて**，それぞれで別々にAttentionを計算しています。これを **マルチヘッドAttention** といいます。
例えば「1つ目のヘッドは直前の文字に注目」「2つ目のヘッドは最初の文字に注目」のように，**それぞれのヘッドが違う視点で注目できる** のです(どのヘッドが何に注目するかは，学習で自然に決まります)。
最後に4つの結果をつなげて，線形変換 `attn_wo` で混ぜ合わせます。
> In 0.1, `n_head = 4`: the 16-component vector is **split into 4 parts** of 4 components each, and Attention is calculated separately for each. This is called **multi-head Attention**.
For example, "head 1 focuses on the previous character" and "head 2 focuses on the first character": **each head can focus from a different viewpoint** (what each head focuses on is decided naturally by learning).
Finally, the 4 results are joined and mixed with the linear transformation `attn_wo`.

## 8.8 実装: Attentionを使ったモデル - Implementation: a model with Attention

0.1のコードとほぼ同じAttentionを実装して学習させます。**1〜2分かかります。**
あとで「どこに注目したか」を図にするために，重みを記録できるようにしておきます(`record`)。
> Let's implement Attention almost exactly as in 0.1 and train it. **It takes 1–2 minutes.**
We make it possible to record the weights (`record`) so we can draw "where it focused" later.
""")

code(r'''
random.seed(0)
n_embd, n_head = 16, 4
head_dim = n_embd // n_head
attn_state = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd),
              'attn_wq': matrix(n_embd, n_embd), 'attn_wk': matrix(n_embd, n_embd),
              'attn_wv': matrix(n_embd, n_embd), 'attn_wo': matrix(n_embd, n_embd),
              'lm_head': matrix(vocab_size, n_embd)}
attn_params = all_params(attn_state)
print('パラメーター数:', len(attn_params))

def attention(x, keys, values, wq, wk, wv, wo, record=None):
    q, k, v = linear(x, wq), linear(x, wk), linear(x, wv)   # 1. Query, Key, Value を作るよ
    keys.append(k)                                            # 8.6 今までの文字のKeyとValueだけがたまっていくよ
    values.append(v)
    x_attn = []
    for h in range(n_head):                                   # 8.7 ヘッドごとに計算するよ
        hs = h * head_dim
        q_h = q[hs:hs + head_dim]
        k_h = [ki[hs:hs + head_dim] for ki in keys]
        v_h = [vi[hs:hs + head_dim] for vi in values]
        attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]  # 2. 内積 ÷ √d
        attn_weights = softmax(attn_logits)                   # 3. softmax
        if record is not None:
            record[h].append([w.data for w in attn_weights])
        head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]  # 4. 加重平均
        x_attn.extend(head_out)
    return linear(x_attn, wo)                                 # 4つのヘッドの結果を混ぜるよ

def attn_model(tokens, record=None):
    keys, values, logits_list = [], [], []
    s = attn_state
    for pos_id, token in enumerate(tokens):
        x = [t + p for t, p in zip(s['wte'][token], s['wpe'][pos_id])]   # 8.4 文字 + 位置
        x = attention(x, keys, values, s['attn_wq'], s['attn_wk'], s['attn_wv'], s['attn_wo'], record)
        logits_list.append(linear(x, s['lm_head']))
    return logits_list

attn_history = train_sgd(attn_model, attn_params, num_steps=1000, learning_rate=0.5)
print(f'テストloss: {test_loss(attn_model):.4f}  (数えるbigram: {BIGRAM_TEST_LOSS:.4f})')
print('作った名前:', ' '.join(generate(attn_model, 12)))
plot_history({'MLPモデル': mlp_history, 'Attentionモデル': attn_history})
''')

md(r"""
ついに，**テストlossが数えるbigramモデルを下回りました！** 作った名前も，2〜3文字の名前らしいものが増えました。
前の文字をすべて見て，さらに「何文字目か」もわかるようになったので，「直前の1文字だけ」の限界を超えられたのです。
> Finally, **the test loss went below the counting bigram model!** The names it makes are now mostly 2–3 characters long and look more like names.
Because it looks at all the previous characters and also knows the position, it went beyond the limit of "only the one previous character".

### [スライダー] モデルがどこに注目しているか見てみよう - [Slider] See where the model is paying attention

名前を入力すると，4つのヘッドそれぞれの注目の重みが表示されます。
**行が「今の文字(Query)」，列が「注目された文字(Key)」** です。明るいほど強く注目しています。右上が空白なのは，未来の文字は見ないからです(8.6)。
> Type a name to see the attention weights of each of the 4 heads.
**Rows are the "current character (query)", and columns are the "character being attended to (key)"**. Brighter means stronger attention. The upper right is empty because the model does not look at future characters (8.6).
""")

code(r'''
#@title [スライダー] Attentionの重みのヒートマップ { display-mode: "form" }
def show_attention(name='さくら'):
    if not name or any(ch not in stoi for ch in name) or len(name) > block_size - 1:
        print(f'語彙にあるひらがなで，{block_size - 1}文字以内の名前を入れてね')
        return
    tokens = tokenize(name)[:-1]
    record = [[] for _ in range(n_head)]
    logits_list = attn_model(tokens, record)
    labels = [tok_str(t) for t in tokens]
    fig, axes = plt.subplots(1, n_head, figsize=(3.2 * n_head, 3.4))
    for h, ax in enumerate(axes):
        grid = [row + [float('nan')] * (len(tokens) - len(row)) for row in record[h]]
        im = ax.imshow(grid, cmap='viridis', vmin=0, vmax=1)
        ax.set_xticks(range(len(tokens)), labels, fontsize=9)
        ax.set_yticks(range(len(tokens)), labels, fontsize=9)
        ax.set_title(f'ヘッド {h + 1}', fontsize=11, fontweight='bold')
        ax.set_xlabel('注目された文字 (Key)', fontsize=9)
    axes[0].set_ylabel('今の文字 (Query)', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()
    for t, lg in zip(labels, logits_list):
        p = softmax_numbers([l.data for l in lg])
        top = sorted(range(vocab_size), key=lambda i: -p[i])[:5]
        print(f'「{t}」の次の予想: ' + '  '.join(f'{tok_str(i)} {p[i]:.2f}' for i in top))

interact(show_attention, name=widgets.Text(value='さくら', description='名前:'));
''')

md(r"""
ヘッドによって注目のしかたが違うことがわかります。どのヘッドがどんな役割を持つかは，学習をやり直すと変わります。
> You can see that each head pays attention differently. Which head plays which role changes if you train again.

> このモデルにはまだ第7章のMLPが入っていません。次の章では，AttentionとMLPを **何層も重ねても学習がうまくいく** ための工夫を学び，0.1のGPTを完成させます。
> This model does not have the MLP from Chapter 7 yet. In the next chapter, we learn tricks that make training work **even when Attention and MLP are stacked in many layers**, and complete the GPT from 0.1.

## 8.9 0.1のコードとの対応 - Matching with the code in 0.1

```python
q = linear(x, state_dict[f'layer{li}.attn_wq'])   # 8.3 Queryを作るよ
k = linear(x, state_dict[f'layer{li}.attn_wk'])   # 8.3 Keyを作るよ
v = linear(x, state_dict[f'layer{li}.attn_wv'])   # 8.3 Valueを作るよ
keys[li].append(k); values[li].append(v)          # 8.6 過去の文字だけを蓄積するよ
for h in range(n_head): ...                       # 8.7 マルチヘッド計算だよ
    attn_logits = [... / head_dim**0.5 ...]       # 8.5 内積 ÷ √d
    attn_weights = softmax(attn_logits)           # 8.3 softmaxで重み化するよ
    head_out = [sum(attn_weights[t] * v_h[t][j] ...)]  # 8.2 加重平均だよ
x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])   # 8.7 ヘッドの出力を混ぜるよ
```

## 練習問題 - Exercises

1. 重みが $0.5,\ 0.3,\ 0.2$，値が $10,\ 20,\ 50$ のときの加重平均を求めましょう。
2. Queryとすべての Key の内積が同じ値だったら，重みはどうなるでしょうか？
3. 位置埋め込みがないと，「はな」の最後の「な」と「なな」の最後の「な」で，Attentionの出力は同じになるでしょうか？ 違うでしょうか？

<details><summary>答え - Answers</summary>

1. $0.5 \times 10 + 0.3 \times 20 + 0.2 \times 50 = 21$ です。
2. すべて同じ重み(文字が $n$ 個なら $\frac1n$ ずつ)になり，ただの平均になります。
3. 違います。注目される文字の集まりが「BOS，は，な」と「BOS，な，な」で違うからです。ただし位置埋め込みがないと「はな」と「なは」のような並び替えは区別できません。
</details>
""")
