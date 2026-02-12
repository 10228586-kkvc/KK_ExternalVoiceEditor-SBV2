# KK_ExternalVoiceEditor-SBV2

[English README](README.en.md)

** 概要 **
- コイカツのキャラスタジオで、任意ボイスを使用するKK_ExternalVoiceLoaderのデータ編集補助（KK_ExternalVoiceLoaderは用意しなくてよい）
- AI（Style-Bert-VITS2）を使用して、任意のテキストの音声合成して音声ファイルを作成

** インストール **
1. 任意のフォルダ（フォルダ名は英数字）にダウンロードしたInstall-KK_ExternalVoiceEditor-SBV2.bat（Setup-Python.batも必要）を設置してダブルクリックする。
2. Environment setup is complete.と表示されたらウィンドウを閉じる。
3. 作成されたKK_ExternalVoiceEditor-SBV2フォルダ内にあるApp.batをダブルクリックする。

![KK_ExternalVoiceEditor-SBV2コイカツ外部音声](https://github.com/user-attachments/assets/316a9325-bddf-485f-aa74-97d41d3ecc1c)

** コイカツ外部音声の使い方 **
- モデル: 現在未使用（AI音声合成で生成したモデルの情報を選択の予定）
- 性格: コイカツの性格
- リロード: 選択したモデル、性格でボイス一覧へ反映
- ボイス一覧: クリックして選択
- △▲▼▽: 選択したボイスの並び順を変更、最上部・上部・下部・最下部へ移動
- 追加: ボイス情報を追加
- 変更: ボイス情報を変更
- 削除: ボイス情報を削除
- 音声: 選択したボイスを再生
- 出力パス: ボイス（wavファイル、oggファイル）、プラグイン（KK_ExternalVoiceLoader.dll、KKS_ExternalVoiceLoader.dll）、ボイスリストmod（KK_KKS_custom-voice-list-1.0.0.zipmod）を指定したパスにインストール
- 出力先: パス指定・コイカツ！・コイカツ！サンシャインの出力先選択（コイカツ！、コイカツ！サンシャインがインストールされていれば表示）
- メッセージ: 実行結果を表示
- 起動ボタン: コイカツ！・コイカツ！サンシャインを起動（コイカツ！、コイカツ！サンシャインがインストールされていれば表示）

![KK_ExternalVoiceEditor-SBV2音声合成](https://github.com/user-attachments/assets/b263ac21-1f38-4597-b51b-5ff8e8419ac1)

** 音声合成の使い方 **
- モデル一覧: 声優風学習済みモデル一覧
- モデルファイル: モデルファイル
- 設定保存: 設定をJson形式でファイルとして保存
- 設定開く: 保存したJson形式の設定ファイルを開く
- 設定初期化: 設定を初期化
- ロード: モデルファイルを読み込む
- テキスト: 音声合成するセリフのテキスト
- 音高(1以外では音質劣化): 音声の高さ
- 抑揚(1以外では音質劣化): 音声の抑揚
- 改行で分けて生成（分けたほうが感情が乗ります）: 
- 改行ごとに挟む無音の長さ（秒）: 
- アクセント調整（数値は 0=低 か1=高 のみ）: 
- アクセント調整を使う: 
- Language: テキストの言語（JP固定）
- 話者: モデルに複数の話者が入っている場合は選択
- SDP Ratio: 
- Noise: 
- Noise_W: 
- Length: 
- Assist textを使う: 
- Assist text: 
- Assist textの強さ: 
- スタイルの指定方法: プリセットから選ぶ: モデルに内蔵されたスタイルを使う、音声ファイルを入力: 音声ファイルからスタイルを作成して使う
- スタイル（{DEFAULT_STYLE}が平均スタイル）: モデルに複数のスタイルがある場合は選択
- スタイルの強さ（声が崩壊したら小さくしてください）: 重み
- 参照音声: 
- 音声合成: 音声合成の実行
- 音声プレーヤー: 合成した音声を再生するプレーヤー
- 性格: コイカツの性格
- カテゴリー: カスタム固定
- セリフ: 登録するタイトル
- 音声形式: ogg: 圧縮音声、wav: 無圧縮音声
- 外部音声へ送る: コイカツ外部音声へ合成した音声ファイルを送る
- テキスト例: リストをクリックするとテキストへ入力

** 注意事項 **
- 使用するボイスはUserDataフォルダの中に保存してください。（パスはUserDataフォルダ以降のフォルダ）
- ボイスはwavファイルかoggファイルを使用してください。
- 出力パスがパス指定の場合は、プラグイン（KK_ExternalVoiceLoader.dll、KKS_ExternalVoiceLoader.dll）はコピーされません。
- バックアップを保存したい場合は、app.dbファイルとUserDataフォルダをコピーしてください。この二つを上書きすれば復元できます。
- 音声はパスとファイル名が正しくないと再生できません。
- 出力先選択と起動ボタンはコイカツ！、コイカツ！サンシャイン、Koikatsu Partyがインストールされていれば表示されます。

** キャラスタジオ（コイカツ！・コイカツ！サンシャイン） **

![コイカツ！のキャラスタジオのボイスに任意の音声データを追加する。](https://github.com/user-attachments/assets/ad7565d3-6bad-4d4d-b05b-489bc206bffb)

** キャラスタジオ音声MODチュートリアル **
[![【コイカツのキャラスタジオ】でキャラが喋る！Timeline＋音声MODの神機能！AIで声優ボイスを再現！ボイス付きのシーンが作れる！【チュートリアル】](https://github.com/user-attachments/assets/9f6396b1-35fa-4822-90d7-27bb9bb046ba)](https://www.youtube.com/watch?v=Aw6TAnGvwCw)

** 関連情報 **
- 外部音声ファイル読み込み KK_ExternalVoiceLoader[[Download>https://github.com/10228586-kkvc/KK_ExternalVoiceLoader]]

- TimelineVoiceControl(Rikki-Koi-Plugins)[[Download>https://github.com/RikkiBalboa/Rikki-Koi-Plugins]]

- Timelineで音声を使う[[チュートリアル>https://www.youtube.com/watch?v=Aw6TAnGvwCw]]

- コイカツ本編セリフ一覧[[kk-studio-add-list>https://github.com/10228586-kkvc/kk-studio-add-list]]

- 音声MODについて[[コイカツ！MODスレ避難所（音声MOD等）>https://jbbs.shitaraba.net/game/61301/]]
