# LLM Study

高校1年生(数学IA履修程度)向けの日英併記LLM教材。Karpathy の microgpt をベースにした Colab ノートブック1本。

## 構成

- `Learn_the_Basics_of_...ipynb`：教材本体(Colab 向け)
- `_build/`：ノートブックの生成スクリプト
  - `nbbuild.py`：`ch00.py`〜`ch11.py`，`appendix.py` の `md()` / `code()` からノートブックを組み立てる
  - 先頭3セル(0 はじめに / 0.1 コード / 0.2 何が起きたのか)は既存ノートブックから引き継ぐ(`KEEP = 3`)
  - `inspect_nb.py`：実行後の出力(エラー・動画・画像の有無)を一覧表示する
  - `preview.py`：見た目の検査用。全セルを高速に実行し(学習は30ステップに短縮)，スライダーを既定値・最小・最大で描画した PNG を大量に出力する。`--anim ch04` で指定した章のアニメーションを再描画してコマ送り画像にする。`--sheets` で一覧画像を作る

## ビルドと検証

```
py _build/nbbuild.py            # 組み立てのみ(出力なし)
py _build/nbbuild.py --run      # 全セルを実行して出力も保存(約20分：学習とmanimの描画を含む)
py _build/inspect_nb.py         # エラー確認
py _build/preview.py --sheets       # 図のプレビューを _preview/ に PNG で出力(約1分)
```

`--run` には `nbclient` `nbformat` `ipykernel` `ipywidgets` と `manim`(+ffmpeg)が必要。
カーネル名は環境変数 `NB_KERNEL` で指定する(省略時 `python3`)。

## 方針

- 本文は日本語の段落 + `>` で平易な英文を併記する
- 章は「なぜ必要か → 数学 → 実装」の順。各章に【スライダー】(ipywidgets + matplotlib)と【アニメーション】(manim)を入れる
- グラフ・動画のコードは `#@title ... { display-mode: "form" }` で Colab 上では折りたたむ
- フォントは同梱の `IBMPlexSansJP-Regular.ttf` を使う(環境差の文字化け防止)。`font_file()` がダウンロードして登録する。`↔`(U+2194)など Plex にないグリフは使わない(⇔ はある)
- manim のシーンは `make_scene(mn, T)` を定義して `show_anim(make_scene)` で描画する。`from manim import *` はノートブックの関数名(`linear` など)を上書きするので使わない
- manim では LaTeX(`MathTex`，`Matrix`，`DecimalNumber`)を使わない(Colab で LaTeX のインストールが必要になるため)。文字は `T()`(`mn.Text` + 日本語フォント)で書く。下付き文字(₁ など)はフォントにないので使わない
- `always_redraw` の中で毎フレーム `Text` を作らない(遅く，長時間の実行で Pango が失敗する)。値の表示は区切りごとに `Transform` で更新する
- 本文中の数値(loss など)の主張は，`--run` の実際の出力と一致しているか確認する
