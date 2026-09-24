from nbbuild import md, code

md(r"""
---
# 第1章 文字を数字に: Tokenizer - Chapter 1: From characters to numbers

0.1のプログラムは，最初に「語彙の大きさ: 72」と表示しました。
この章では，この72という数が何なのか，そしてなぜ文字を数字に直す必要があるのかを学びます。
> The program in 0.1 first printed "vocabulary size: 72".
In this chapter, we will learn what this number 72 means and why we need to turn characters into numbers.

**この章で使う数学**: 集合(数学I「集合と命題」)，対応
> **Math in this chapter**: sets (Math I), correspondence (mapping)

## 1.1 コンピューターは数しか扱えない - Computers can only handle numbers

コンピューターの中では，文字も画像も音も，すべて数として扱われています。
実は文字には，世界共通の番号(Unicode)がすでに決められています。確かめてみましょう。
> Inside a computer, text, images and sound are all handled as numbers.
In fact, every character already has a worldwide number called Unicode. Let's check it.
""")

code(r'''
for ch in 'さくら':
    print(ch, '→', ord(ch))  # ord(): 文字のUnicode番号(コードポイント)を調べる関数だよ
''')

md(r"""
「さ」は12373番のようです。でも，Unicodeには約15万もの文字が登録されています。
私たちの学習データに出てくる文字はそのうちのごく一部なので，**実際に出てくる文字だけに0から順に番号を付け直す**ことにします。
この「文字 → 番号」の変換をする仕組みを **Tokenizer(トークナイザー)** と呼び，変換された番号を **Token(トークン)** と呼びます。
> "さ" is number 12373. But Unicode has about 150,000 characters.
Only a tiny part of them appear in our training data, so **we give new numbers, starting from 0, only to the characters that actually appear**.
The system that does this "character → number" conversion is called a **Tokenizer**, and the resulting numbers are called **Tokens**.

## 1.2 集合: 出てくる文字の集まり - Sets: the collection of characters that appear

数学Iで学ぶ **集合** とは，「ものの集まり」のことです。集合に入っている1つ1つのものを **要素** といいます。
集合には，次の2つの大事な性質があります。
> A **set** in Math I is "a collection of things". Each thing in a set is called an **element**.
Sets have two important properties:

1. 同じ要素は1回しか数えない(重複しない)
2. 要素の順番は関係ない
> 1. The same element is counted only once (no duplicates).
> 2. The order of elements does not matter.

集合 $A$ の要素の個数を $n(A)$ と書きます。Pythonにも `set` という集合があります。
> The number of elements of a set $A$ is written $n(A)$. Python also has sets, called `set`.
""")

code(r'''
A = set('さくら')
B = set('さつき')
print('A =', A, ' n(A) =', len(A))
print('B =', B, ' n(B) =', len(B))
print('A ∪ B (和集合) =', A | B)   # どちらかに入っている文字だよ
print('A ∩ B (共通部分) =', A & B)  # 両方に入っている文字だよ
print("set('ももこ') =", set('ももこ'))  # 「も」は1回しか入らないよ
''')

md(r"""
学習データのすべての名前をつなげて集合にすれば，「データに出てくる文字の集合」が作れます。
最後に `sorted()` で並べておくと，毎回同じ順番になって便利です(ひらがなはおおよそ五十音順に並びます)。
> If we join all the names in the training data and make a set, we get "the set of characters that appear in the data".
We sort it with `sorted()` so the order is always the same (hiragana comes out roughly in Japanese alphabetical order).
""")

code(r'''
uchars = sorted(set(''.join(docs)))
print('出てくる文字の数 n(U) =', len(uchars))
print(''.join(uchars))
''')

md(r"""
## 1.3 対応: 文字 ⇔ 番号 - Correspondence: character ⇔ number

次に，集合の要素1つ1つに番号を割り当てます。
「あ」→ 0，「い」→ 1，… のように，**1つの文字に1つの番号**，しかも **違う文字には違う番号** が対応するようにします。
このような対応を **1対1の対応** といいます。1対1なので，番号から文字へ逆向きに戻すこともできます。
> Next, we give a number to each element of the set.
Like "あ" → 0, "い" → 1, ..., **one character gets one number**, and **different characters get different numbers**.
This is called a **one-to-one correspondence**. Because it is one-to-one, we can also go back from numbers to characters.

- 文字 → 番号: **encode(エンコード)**
- 番号 → 文字: **decode(デコード)**
> - Character to number: **encode**
> - Number to character: **decode**
""")

code(r'''
stoi = {ch: i for i, ch in enumerate(uchars)}  # string to integer: 文字 → 番号の辞書だよ
itos = {i: ch for i, ch in enumerate(uchars)}  # integer to string: 番号 → 文字の辞書だよ

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

print('「さくら」をencode:', encode('さくら'))
print('元に戻すdecode:', decode(encode('さくら')))
''')

md(r"""
### [アニメーション] 文字が番号に変わる - [Animation] Characters become numbers
""")

code(r'''
#@title アニメーション: encode と decode { display-mode: "form" }
def make_scene(mn, T):
    class TokenMap(mn.Scene):
        def construct(self):
            title = T('Tokenizer: 文字 ⇔ 番号', font_size=36).to_edge(mn.UP)
            self.play(mn.Write(title))
            word = ['BOS'] + list('さくら') + ['BOS']
            ids = [BOS] + [uchars.index(c) for c in 'さくら'] + [BOS]
            top = mn.VGroup(*[mn.VGroup(mn.RoundedRectangle(width=1.3, height=1, corner_radius=0.15, color=mn.BLUE), T(w, font_size=30 if w == 'BOS' else 40)) for w in word]).arrange(mn.RIGHT, buff=0.4).shift(mn.UP * 1.2)
            for g in top: g[1].move_to(g[0])
            bottom = mn.VGroup(*[mn.VGroup(mn.RoundedRectangle(width=1.3, height=1, corner_radius=0.15, color=mn.YELLOW), T(str(i), font_size=36)) for i in ids]).arrange(mn.RIGHT, buff=0.4).shift(mn.DOWN * 1.4)
            for g in bottom: g[1].move_to(g[0])
            enc = T('encode', font_size=28, color=mn.BLUE).next_to(top, mn.LEFT).shift(mn.DOWN * 1.3)
            self.play(mn.LaggedStart(*[mn.FadeIn(g, shift=mn.DOWN * 0.3) for g in top], lag_ratio=0.2))
            self.play(mn.FadeIn(enc))
            arrows = []
            for a, b in zip(top, bottom):
                arr = mn.Arrow(a.get_bottom(), b.get_top(), buff=0.1, color=mn.GRAY)
                arrows.append(arr)
                self.play(mn.GrowArrow(arr), mn.FadeIn(b, shift=mn.DOWN * 0.3), run_time=0.5)
            note = T('BOS は「名前の始まり・終わり」の目印だよ', font_size=26).to_edge(mn.DOWN)
            self.play(mn.Indicate(top[0]), mn.Indicate(top[-1]), mn.Write(note))
            self.wait(0.5)
            dec = T('decode', font_size=28, color=mn.YELLOW).move_to(enc)
            self.play(*[mn.Rotate(a, angle=mn.PI) for a in arrows], mn.Transform(enc, dec))
            self.play(*[mn.Indicate(g) for g in top[1:-1]])
            self.wait(1)
    return TokenMap

show_anim(make_scene)
''')

md(r"""
## 1.4 特別なToken「BOS」 - The special token "BOS"

モデルに名前を作らせるには，「ここから名前が始まるよ」「ここで名前が終わるよ」という目印も必要です。
そこで，どの文字とも違う特別なTokenを1つ追加します。これが **BOS(Beginning of Sequence)** です。
番号は，文字の番号の次(71番)を使います。だから語彙の大きさは $71 + 1 = 72$ になるのです。
> To let the model make names, we also need marks that say "a name starts here" and "a name ends here".
So we add one special token that is different from every character. This is **BOS (Beginning of Sequence)**.
It uses the number right after the characters (71). That is why the vocabulary size is $71 + 1 = 72$.

名前は，前後をBOSではさんだToken列にして使います。
> Each name is turned into a token sequence with BOS on both ends.
""")

code(r'''
BOS = len(uchars)
vocab_size = len(uchars) + 1
print('BOS =', BOS, ' 語彙の大きさ =', vocab_size)

def tokenize(name):
    return [BOS] + encode(name) + [BOS]

print('さくら →', tokenize('さくら'))
''')

md(r"""
### [スライダー] 好きな名前をTokenにしてみよう - [Slider] Turn any name into tokens

下の入力欄にひらがなで名前を入れてみましょう。学習データに出てこない文字(カタカナや漢字など)は変換できないので，そのことも表示されます。
> Type a name in hiragana in the box below. Characters that do not appear in the training data (katakana, kanji, etc.) cannot be converted, and you will see that too.
""")

code(r'''
#@title [スライダー] 名前 → Token列 { display-mode: "form" }
def show_tokens(name='はなこ'):
    missing = [ch for ch in name if ch not in stoi]
    if missing:
        print('語彙にない文字があるよ:', ' '.join(missing))
        return
    ids = tokenize(name)
    print(' '.join(f'{tok_str(i):>4}' for i in ids))
    print(' '.join(f'{i:>4}' for i in ids))
    print(f'Token数: {len(ids)} (文字数 {len(name)} + BOS 2個)')
    print('学習データに', 'ある名前だよ' if name in docs else 'ない名前だよ')

interact(show_tokens, name=widgets.Text(value='はなこ', description='名前:'));
''')

md(r"""
## 1.5 0.1のコードとの対応 - Matching with the code in 0.1

0.1のプログラムでは，この章の内容はたった3行で書かれていました。
> In the program in 0.1, this chapter was written in just 3 lines:

```python
uchars = sorted(set(''.join(docs)))  # 1.2 集合で重複を排除してソートするよ
BOS = len(uchars)                    # 1.4 BOSの番号を決めるよ
vocab_size = len(uchars) + 1         # 1.4 語彙数を決めるよ
tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]  # 1.3 encode + BOS
```

`uchars.index(ch)` は「リストの中で `ch` が何番目にあるか」を調べる命令で，`stoi[ch]` と同じ結果になります。
> `uchars.index(ch)` finds "at which position `ch` is in the list", which gives the same result as `stoi[ch]`.

## 1.6 発展: 本物のLLMのTokenizer - Going further: tokenizers in real LLMs

ChatGPTなどの本物のLLMは，1文字ずつではなく「よく出てくる文字のかたまり」を1つのTokenにします(**BPE** という方法)。
例えば英語の "ing" や "the" は1つのTokenになることが多いです。語彙の大きさは数万〜数十万にもなります。
> Real LLMs like ChatGPT do not use one character per token. They turn "frequent chunks of characters" into one token (a method called **BPE**).
For example, "ing" or "the" in English are often a single token. The vocabulary size can be tens of thousands to hundreds of thousands.

日本語は英語に比べて1文字あたりのToken数が多くなりやすく，同じ内容でも日本語のほうがTokenを多く使うことがあります。
LLMの利用料金や「一度に読める文章の長さ」はToken数で決まるので，これはAIを使ううえで知っておくと役に立ちます。
> Japanese tends to need more tokens per character than English, so the same content can use more tokens in Japanese.
The price of using an LLM and "how much text it can read at once" are measured in tokens, so this is useful to know when using AI.

## 練習問題 - Exercises

1. `set('たまたま')` の要素の個数を予想してから，実行して確かめましょう。
2. `decode([BOS])` を実行するとどうなるでしょうか？ なぜそうなるか考えてみましょう。
3. 学習データに出てこないひらがなを探してみましょう(ヒント: 五十音表と `uchars` を見比べる)。

<details><summary>答え - Answers</summary>

1. 2個(「た」と「ま」)。同じ要素は1回しか数えないからです。
2. エラーになります。`itos` には文字の番号(0〜70)しか入っていなくて，BOS(71)は入っていないからです。
3. 例えば「ぱ」「ぴ」などの半濁音や，「ゐ」「ゑ」などです。実際に `[ch for ch in 'あいうえおかきくけこ...' if ch not in stoi]` のように調べてみましょう。
</details>
""")
