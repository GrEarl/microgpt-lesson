# Learn the Basics of Large Language Models with Karpathy's MicroGPT

高校1年生(数学IA履修程度)を対象に，大規模言語モデル(LLM)・GPTの仕組みを，数学とプログラミングの初歩から順番に作って学ぶ教材です。Google Colabで動かせます。
> This is learning material that builds up LLM/GPT from basic math and programming, one piece at a time. It runs on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GrEarl/microgpt-lesson/blob/main/Learn_the_Basics_of_Large_Language_Models_with_Karpathy%27s_MicroGPT_LLM%28%E5%A4%A7%E8%A6%8F%E6%A8%A1%E8%A8%80%E8%AA%9E%E3%83%A2%E3%83%87%E3%83%AB%29%E3%81%AE%E5%9F%BA%E6%9C%AC%E3%82%92%E4%BD%9C%E3%81%A3%E3%81%A6%E5%AD%A6%E3%81%BC%E3%81%86.ipynb)

## 使い方 - How to use

1. 上のバッジからGoogle Colabで開き，「ファイル → ドライブに保存」で自分のコピーを作ります
2. 最初に「共通準備」セルを実行し，そのあと上から順番に実行します(順番どおりに実行すれば，そのまま最後まで動きます)
3. [アニメーション]の動画やグラフは実行結果として保存済みなので，実行しなくても見られます
4. 学習データとグラフ用フォントは実行時に自動でダウンロードされます(インターネット接続が必要です)

> 1. Open it in Google Colab with the badge above, then use "File → Save a copy in Drive" to make your own copy.
> 2. Run the "common setup" cell first, then run the cells in order from top to bottom.
> 3. The [Animation] videos and figures are saved in the notebook, so you can view them without running anything.
> 4. The training data and the graph font are downloaded automatically when you run the cells.

## このリポジトリの構成 - Repository layout

- `*.ipynb`: 教材本体(Colab向けノートブック)
- `_build/`: ノートブックの生成スクリプト一式(教材の修正はここの章ファイル `ch00.py`〜`ch11.py`，`appendix.py` を行います)

```sh
py _build/nbbuild.py          # 章ファイルからノートブックを組み立て直す(出力なし)
py _build/nbbuild.py --run    # 組み立てたうえで全セルを実行し，実行結果(動画・グラフ)も再生成する
py _build/inspect_nb.py       # 実行後のエラーと各セルの出力を確認する
```

## クレジット - Credits

- Andrej Karpathy, [microgpt.py (GitHub Gist)](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95): この教材のコードの元になったプログラム
- shuheilocale, [japanese-personal-name-dataset](https://github.com/shuheilocale/japanese-personal-name-dataset): 学習データ(MIT License)
- [IBM Plex Sans JP](https://github.com/google/fonts/tree/main/ofl/ibmplexsansjp): グラフ・動画の日本語フォント(SIL Open Font License)
- manim / 3Blue1Brown: アニメーションの描画ライブラリ
