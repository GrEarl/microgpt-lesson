from nbbuild import md

md(r"""
---
# 付録 - Appendix

## A. 用語集 - Glossary

| 用語 | 英語表記 | 意味 | 登場する章 |
|---|---|---|---|
| 大規模言語モデル | Large Language Model (LLM) | 大量の文章で学習し，次のTokenを予想するモデル | 第0章 |
| トークン | Token | モデルが扱う文字や単語のかけらの単位，とその番号 | 第1章 |
| トークナイザー | Tokenizer | 文字列とTokenを変換する仕組み | 第1章 |
| 語彙 | Vocabulary | Tokenの種類の全体 | 第1章 |
| 言語モデル | Language Model | 次に来るTokenの確率を求めるモデル | 第2章 |
| 条件付き確率 | Conditional Probability | ある条件のもとでの確率 $P_A(B)$ | 第2章 |
| 損失 (loss) | Loss | 予想の外れ具合を表す数。$-\log p$ の平均 | 第3章 |
| スコア (ロジット) | Logit / Score | softmaxに入れる前の数 | 第3章 |
| ソフトマックス | Softmax | スコアを確率に変える関数 | 第3章 |
| 温度 (temperature) | Temperature | 選び方の自由さを決める数 | 第3章 |
| パラメーター | Parameter | 学習で調整されるモデルの中の数 | 第4章 |
| 微分 | Derivative / Differentiation | ある点での傾きを求めること | 第4章 |
| 勾配 | Gradient | 各パラメーターについての傾きを並べたもの | 第4章 |
| 勾配降下法 | Gradient Descent | 傾きの逆向きに少しずつ進んで最小値を探す方法 | 第4章 |
| 学習率 | Learning Rate ($\eta$) | 1回に進む大きさ | 第4章 |
| 計算グラフ | Computation Graph | 計算を1つずつ矢印でつないだ図 | 第5章 |
| 連鎖律 | Chain Rule | つながった計算の傾きは掛け算になる，という法則 | 第5章 |
| 順伝播 / 逆伝播 | Forward Pass / Backward Pass (Backpropagation) | 値を計算する向き / 傾きを求める向き | 第5章 |
| ベクトル | Vector | 数の組 | 第6章 |
| 内積 | Dot Product | 成分どうしを掛けて足したもの。似ている度合い | 第6章 |
| 行列 | Matrix | 数を長方形に並べたもの | 第6章 |
| 線形変換 | Linear Transformation | 行列の各行とベクトルの内積を並べる変換 | 第6章 |
| 埋め込み | Embedding | Tokenをベクトルに変える表 | 第6章 |
| ニューラルネットワーク | Neural Network | 線形変換と非線形な関数を重ねたモデル | 第7章 |
| 活性化関数 | Activation Function | ReLUのように，間にはさむ直線でない関数 | 第7章 |
| 多層パーセプトロン | Multi-Layer Perceptron (MLP) | 広げる → ReLU → 縮める，の部品 | 第7章 |
| アテンション | Attention | 前のTokenのどれに注目するかを決めて加重平均する仕組み | 第8章 |
| クエリ / キー / バリュー | Query / Key / Value | 探す条件 / 見出し / 中身 | 第8章 |
| 位置埋め込み | Position Embedding | 何番目かを表すベクトル | 第8章 |
| マルチヘッド | Multi-Head Attention | いくつもの視点で同時にAttentionを計算すること | 第8章 |
| 正規化 | Normalization | 値の大きさをそろえること(RMSNormなど) | 第9章 |
| 残差接続 | Residual Connection | 入力を出力にそのまま足す近道 | 第9章 |
| 層 | Layer | Attentionブロック + MLPブロックのひとまとまり | 第9章 |
| 最適化手法 | Optimizer | パラメーターの更新のしかた(勾配降下法，Adamなど) | 第10章 |
| 移動平均 | Moving Average | 最近の値の平均 | 第10章 |
| 訓練 / 推論 | Training / Inference | 学習すること / 学習したモデルで予想すること | 第0・11章 |
| 事前学習 / 事後学習 | Pre-training / Post-training | 大量の文章で次のTokenを学ぶ / 指示に従うなどの調整 | 第11章 |
| ハルシネーション | Hallucination | もっともらしいが事実と違う出力 | 第11章 |
| プロンプト | Prompt | AIへの入力文 | 第11章 |

## B. 数学の公式まとめ - Math formulas

**確率**(第2章)
$$P_A(B) = \frac{n(A \cap B)}{n(A)}$$

**指数・対数**(第3章)
$$a^m \times a^n = a^{m+n}, \quad a^0 = 1, \quad a^{-n} = \frac{1}{a^n}$$
$$\log_a M = p \iff a^p = M, \quad \log_a(M \times N) = \log_a M + \log_a N$$

**loss と softmax**(第3章)
$$\text{loss} = \frac{1}{n}\sum_{i=1}^{n}\left(-\log p_i\right), \qquad p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

**微分**(第4章)
$$f'(a) = \lim_{h \to 0}\frac{f(a+h) - f(a)}{h}, \quad (x^n)' = n x^{n-1}, \quad (e^x)' = e^x, \quad (\log x)' = \frac{1}{x}$$

**勾配降下法**(第4章)
$$\theta \leftarrow \theta - \eta \, \frac{\partial L}{\partial \theta}$$

**連鎖律**(第5章): つながった計算の傾きは，それぞれの傾きの積

**内積**(第6章)
$$\vec{a} \cdot \vec{b} = a_1 b_1 + a_2 b_2 = |\vec{a}|\,|\vec{b}|\cos\theta$$

**ReLU**(第7章)
$$\text{ReLU}(x) = \max(0, x)$$

**Attention**(第8章)
$$\text{出力} = \sum_t \text{softmax}\!\left(\frac{q \cdot k_t}{\sqrt{d}}\right) v_t$$

**RMSNorm**(第9章)
$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}, \qquad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)}$$

**Adam**(第10章)
$$m \leftarrow \beta_1 m + (1-\beta_1) g, \quad v \leftarrow \beta_2 v + (1-\beta_2) g^2, \quad \theta \leftarrow \theta - \eta\,\frac{m / (1-\beta_1^t)}{\sqrt{v / (1-\beta_2^t)} + \varepsilon}$$

## C. 参考資料 - References

- Andrej Karpathy, [microgpt.py (GitHub Gist)](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): このNotebookの元になったコード
- Andrej Karpathy, [microgpt (解説ブログ，英語)](https://karpathy.github.io/2026/02/12/microgpt/)
- Andrej Karpathy, [Neural Networks: Zero to Hero (動画シリーズ，英語)](https://karpathy.ai/zero-to-hero.html): 第2章のbigramや第5章の自動微分を，さらにくわしく解説しています
- 3Blue1Brown, [Neural Networks (動画シリーズ，英語)](https://www.3blue1brown.com/topics/neural-networks) / [3Blue1Brown Japan (日本語版)](https://www.youtube.com/@3Blue1BrownJapan): ニューラルネットワークやTransformerを美しいアニメーションで解説しています(このNotebookのアニメーションにも使っているmanimは，3Blue1Brownが作ったライブラリが元になっています)
- shuheilocale, [japanese-personal-name-dataset](https://github.com/shuheilocale/japanese-personal-name-dataset): 学習データ(MITライセンス)
""")
