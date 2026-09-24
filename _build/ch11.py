from nbbuild import md, code

md(r"""
---
# 第11章 全部組み立てる: MicroGPTの完成 - Chapter 11: Putting it all together: MicroGPT complete

おつかれさまでした！ ここまでで，0.1のGPTのすべての部品を作りました。
この章では全体をふり返り，完成したモデルで遊びながら，私たちが普段使うLLMとのつながりを考えます。
この章では，第10章でAdamを使って学習したモデル(`gpt_adam`)を使います。
> Well done! By now, you have built every part of the GPT in 0.1.
In this chapter, we look back at the whole thing, play with the finished model, and think about how it connects to the LLMs we use every day.
This chapter uses the model trained with Adam in Chapter 10 (`gpt_adam`).

## 11.1 全体の流れ - The whole flow

### [アニメーション] 1文字予想するまでの流れ - [Animation] The flow of predicting one character
""")

code(r'''
#@title アニメーション: GPTの全体像 { display-mode: "form" }
def make_scene(mn, T):
    class PipelineScene(mn.Scene):
        def construct(self):
            title = T('GPT: 次の1文字を予想するまで', font_size=34).to_edge(mn.UP)
            self.play(mn.Write(title))
            steps = [('Token「は」', '第1章', mn.BLUE), ('埋め込み + 位置', '第6・8章', mn.BLUE), ('RMSNorm', '第9章', mn.TEAL),
                     ('Attention + 近道', '第8・9章', mn.YELLOW), ('MLP + 近道', '第7・9章', mn.YELLOW), ('lm_head → スコア', '第6章', mn.BLUE),
                     ('softmax → 確率', '第3章', mn.ORANGE), ('くじ引き →「な」', '第2・3章', mn.RED)]
            boxes = mn.VGroup()
            for name, ch, col in steps:
                b = mn.VGroup(mn.RoundedRectangle(width=2.7, height=0.95, corner_radius=0.15, color=col), T(name, font_size=22), T(ch, font_size=16, color=mn.GRAY))
                b[1].move_to(b[0]).shift(mn.UP * 0.12); b[2].move_to(b[0]).shift(mn.DOWN * 0.27)
                boxes.add(b)
            boxes.arrange_in_grid(rows=2, cols=4, buff=(0.65, 1.2)).shift(mn.DOWN * 0.3)
            arrows = mn.VGroup(*[mn.Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.08) for i in (0, 1, 2, 4, 5, 6)])
            arrows.add(mn.Arrow(boxes[3].get_bottom(), boxes[4].get_top(), buff=0.08, path_arc=0))
            for i, b in enumerate(boxes):
                self.play(mn.FadeIn(b, shift=mn.UP * 0.2), run_time=0.45)
                if i < len(boxes) - 1:
                    a = arrows[[0, 1, 2, 6, 3, 4, 5][i]]
                    self.play(mn.GrowArrow(a), run_time=0.25)
            train = T('学習: loss = −log p → 逆伝播(第5章) → Adam で更新(第10章)', font_size=24, color=mn.RED).to_edge(mn.DOWN)
            self.play(mn.Write(train))
            self.play(mn.Indicate(boxes[-1], color=mn.RED), run_time=1)
            self.wait(1.5)
    return PipelineScene

show_anim(make_scene)
''')

md(r"""
### 0.1のコードと章の対応表 - Which chapter explains each part of the code in 0.1

| 0.1のコード | 内容 | 章 |
|---|---|---|
| `uchars`，`BOS`，`vocab_size` | Tokenizer | 第1章 |
| `class Value` | 計算グラフと自動微分 | 第5章 |
| `matrix`，`state_dict` | パラメーター，正規分布の乱数 | 第6章，第9章 |
| `linear` | 線形変換(内積を並べる) | 第6章 |
| `softmax` | スコアを確率に | 第3章 |
| `rmsnorm` | 大きさをそろえる | 第9章 |
| `gpt` の `wte`，`wpe` | 埋め込み，位置埋め込み | 第6章，第8章 |
| `gpt` の `# 1) Multi-head Attention block` | Attention | 第8章 |
| `gpt` の `# 2) MLP block` | MLPとReLU | 第7章 |
| `x_residual` | 残差接続 | 第9章 |
| `loss_t = -probs[target_id].log()` | loss | 第3章 |
| `loss.backward()` | 逆伝播 | 第5章 |
| `lr_t`，`m`，`v`，`m_hat`，`v_hat` | 学習率の減衰，Adam | 第4章，第10章 |
| `temperature`，`random.choices` | temperature，くじ引き | 第3章，第2章 |

もう一度0.1のコードを読んでみてください。最初は呪文のように見えたコードが，1行ずつ意味のわかるものになっているはずです。
> Try reading the code in 0.1 again. Code that looked like a magic spell at first should now make sense line by line.

## 11.2 完成したモデルで名前を作ろう - Make names with the finished model

### [スライダー] temperatureと名前 - [Slider] Temperature and names

temperatureを変えながら，完成したGPTに名前を作らせてみましょう。学習データにある名前か，新しい名前かも表示します。
> Let's have the finished GPT make names while changing the temperature. It also shows whether each name is in the training data or new.
""")

code(r'''
#@title [スライダー] GPTで名前を作る { display-mode: "form" }
def show_names(temperature=0.5, seed=0):
    names = generate(gpt_adam, 20, temperature=temperature, seed=seed)
    new = [n for n in names if n not in docs]
    for i, n in enumerate(names):
        print(f'{n:<6}', '(学習データにある名前)' if n in docs else '(新しい名前！)', end='\n' if i % 2 else '    ')
    print(f'\n新しい名前: {len(new)} / {len(names)}')

interact(show_names,
         temperature=widgets.FloatSlider(value=0.5, min=0.1, max=2.0, step=0.1, description='T:', continuous_update=False),
         seed=widgets.IntSlider(value=0, min=0, max=20, description='乱数の種:', continuous_update=False));
''')

md(r"""
temperatureが低いと，学習データによくある名前ばかりになります。高くすると新しい名前が増えますが，だんだん名前らしくなくなっていきます。
> With a low temperature, you mostly get names that are common in the training data. With a high temperature, there are more new names, but they gradually stop looking like names.

### [スライダー] モデルの頭の中をのぞく - [Slider] Look inside the model's head

名前の最初の何文字かを入力すると，モデルが「次の文字」にどんな確率をつけているかが見られます。
> Type the first few characters of a name to see what probabilities the model gives to "the next character".
""")

code(r'''
#@title [スライダー] 次の文字の確率 { display-mode: "form" }
def show_next(prefix='ゆ', temperature=1.0):
    if any(ch not in stoi for ch in prefix) or len(prefix) > block_size - 1:
        print(f'語彙にあるひらがなで，{block_size - 1}文字以内で入れてね')
        return
    tokens = [BOS] + encode(prefix)
    logits = gpt_adam(tokens)[-1]
    p = softmax_numbers([l.data for l in logits], temperature)
    top = sorted(range(vocab_size), key=lambda i: -p[i])[:15]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.bar(['終わり' if i == BOS else tok_str(i) for i in top], [p[i] for i in top], color='#10b981', edgecolor='#047857')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'「{prefix}」の次の文字の確率 (T = {temperature:.1f})', fontsize=12, fontweight='bold')
    ax.set_ylabel('確率')
    plt.tight_layout()
    plt.show()
    in_data = sum(1 for d in docs if d.startswith(prefix))
    print(f'学習データで「{prefix}」から始まる名前: {in_data}個')

interact(show_next, prefix=widgets.Text(value='ゆ', description='最初の文字:'),
         temperature=widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='T:'));
''')

md(r"""
「ゆ」「ゆき」「さく」などを入れてみましょう。「ゆき」の次は「終わり」(BOS)の確率が高いはずです。
学習データに1つもない始まり方(例えば「ぬぬ」)を入れても，モデルは **何かしらの確率を必ず出します**。これが次の節のテーマです。
> Try "ゆ", "ゆき", "さく" and so on. After "ゆき", "end" (BOS) should have a high probability.
Even if you type a beginning that never appears in the training data (for example "ぬぬ"), the model **always outputs some probabilities**. This is the theme of the next section.

## 11.3 なぜAIはもっともらしい間違いを出力するのか - Why does AI output plausible mistakes?

ChatGPTなどのLLMは，ときどき事実と違うことを，とても自然な文章で答えることがあります。これを **ハルシネーション(幻覚)** といいます。
このNotebookで作ったGPTを思い出すと，その理由が見えてきます。
> LLMs like ChatGPT sometimes give answers that are not true, in very natural sentences. This is called **hallucination**.
Remembering the GPT we built in this Notebook helps us see why.

1. **GPTがやっているのは「次のTokenの確率を計算して，くじを引く」ことだけ** です(第2章，第3章)。「正しいかどうか」を確かめる仕組みはどこにもありません。
2. 学習でlossを小さくするとは，**学習データに「ありそうな」並び方をまねる** ことです。だから出力は「もっともらしく」なります。
3. 学習データにない状況でも(「ぬぬ」の例)，モデルは **必ず何かの確率を出して，何かを選びます**。「わかりません」と言うのも，結局は「わかりません」というTokenの並びが選ばれたときだけです。
4. 0.2で見たように，GPTは学習データにない名前(「新しい名前！」)も作りました。名前づくりならそれは創造性ですが，事実を答えるときには，それが **存在しない事実を作り出すこと** になります。
> 1. **All a GPT does is "calculate the probabilities of the next token and draw a lot"** (Chapters 2, 3). There is no mechanism anywhere that checks whether something is true.
> 2. Making the loss smaller means **imitating sequences that are "likely" in the training data**. That is why the output looks "plausible".
> 3. Even in situations not in the training data (the "ぬぬ" example), the model **always outputs some probabilities and picks something**. Saying "I don't know" only happens when the token sequence "I don't know" is chosen.
> 4. As we saw in 0.2, the GPT made names that were not in the training data ("new names!"). For making names, that is creativity; but when answering facts, it means **creating facts that do not exist**.

つまり，ハルシネーションは「バグ」というより，**GPTの仕組みそのものから自然に出てくる性質** なのです。
本物のLLMは，事後学習(Post-training)などで間違いを減らす工夫をしていますが，仕組み上ゼロにすることは難しいのです。
> In other words, hallucination is not so much a "bug" as **a property that naturally comes from how GPT works**.
Real LLMs use techniques such as post-training to reduce mistakes, but because of how they work, it is hard to make them zero.

## 11.4 AIにどう入力すればいいか - How should we give input to AI?

GPTは「それまでのTokenの並び(文脈)」から次のTokenの確率を決めていました(第8章のAttention)。
スライダーで「ゆ」と「ゆき」を入れたとき，次の文字の確率がまったく違ったように，**入力する文章(プロンプト)が変われば，出てくる答えの確率も変わります**。
> A GPT decides the probabilities of the next token from "the sequence of tokens so far (the context)" (Attention in Chapter 8).
Just as the probabilities of the next character were completely different for "ゆ" and "ゆき" in the slider, **if the input text (the prompt) changes, the probabilities of the answer change too**.

このことから，AIを使うときに役立つコツがわかります。
> This gives us useful tips for using AI:

- **状況や目的をくわしく書く**: 文脈が多いほど，Attentionが注目できる手がかりが増えて，求める答えの確率が高くなります。
- **答えの形を指定する・例を見せる**: 「箇条書きで」「この例のように」と書くと，その形に続くTokenの確率が高くなります。
- **大事な事実は自分で確かめる**: 11.3で見たように，もっともらしさと正しさは別物です。
- **同じ質問でも答えは毎回変わる**: くじ引き(temperature)で選んでいるからです。何回か聞いてみるのも1つの手です。
> - **Describe the situation and your goal in detail**: More context gives Attention more clues to focus on, making the answer you want more likely.
> - **Specify the format or show an example**: Writing "as a bulleted list" or "like this example" makes tokens that follow that format more likely.
> - **Check important facts yourself**: As we saw in 11.3, plausible and correct are different things.
> - **Answers change even for the same question**: Because it chooses by drawing lots (temperature). Asking a few times can help.

## 11.5 本物のLLMとの違い - Differences from real LLMs

私たちのMicroGPTと，ChatGPTなどの本物のLLMの **仕組みの本質は同じ** です。違うのは主に「規模」と「学習のさせ方」です。
> The **essence of the mechanism** is the same for our MicroGPT and real LLMs like ChatGPT. The main differences are "scale" and "how they are trained".

| 項目 | MicroGPT | 本物のLLM |
|---|---|---|
| パラメーター数 | 約5000個 | 数十億〜数千億個以上 |
| Token | ひらがな1文字 | BPEによる文字のかたまり(第1章) |
| 語彙の大きさ | 72 | 数万〜数十万 |
| 層の数 | 1 | 数十〜100以上 |
| 一度に見られる長さ | 6 Token | 数千〜数十万Token以上 |
| 学習データ | 女性名 約3000個 | インターネット上の大量の文章など |
| 学習 | 事前学習のみ | 事前学習 + 事後学習(指示に従う練習，人間の好みに合わせる調整など) |
| 計算 | Pythonで1つずつ | GPUで行列をまとめて一気に |

逆に言えば，**このNotebookで作ったものを大きくしていけば，本物のLLMにつながっている** ということです。
> Put the other way, **if you scale up what you built in this Notebook, it leads to real LLMs**.

## 11.6 もっと試してみよう - Try more

- 第9章の `n_layer = 1` を `2` にすると，層を2つ重ねたGPTになります(学習時間は約2倍)。lossはどう変わるでしょうか？
- `num_steps` を増やすと，lossはどこまで下がるでしょうか？ テストlossも下がり続けるでしょうか？
- 共通準備セルのデータを男性の名前(`first_name_man_org.csv`)に変えてみましょう(同じデータセットに入っています)。
> - Change `n_layer = 1` in Chapter 9 to `2` to make a GPT with two layers (training takes about twice as long). How does the loss change?
> - If you increase `num_steps`, how low does the loss go? Does the test loss keep going down too?
> - Try changing the data in the common setup cell to male names (`first_name_man_org.csv`, in the same dataset).

## おわりに - Closing

0.2で立てた3つの問い，
> The three questions from 0.2:

- 「なぜこのコードから，学習データにない名前が生成されたのか」
- 「私たちが普段使うAIがどう動いていて，どう入力すべきなのか」
- 「なぜAIはもっともらしい間違い(ハルシネーション)を出力するのか」

を，自分の言葉で説明できるようになったでしょうか？
ここまでたどり着いたあなたは，LLMの一番大事な仕組みを，数学から実装まで自分の手で作りきりました。ぜひ，その先の世界も探求してみてください。
> Can you now explain them in your own words?
Having come this far, you have built the most important mechanism of LLMs with your own hands, from the math to the code. Please go on and explore the world beyond.
""")
