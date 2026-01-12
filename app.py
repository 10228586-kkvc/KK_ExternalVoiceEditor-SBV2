# ┌───────────────────────────────────────────
# │  KK_ExternalVoiceEditor v1.0.0 (2025.06.01)
# └───────────────────────────────────────────
# ==============================================================================
# pip install gradio
# pip install pandas

import gradio as gr
import locale
import sqlite3
import pandas as pd
import random
import string
import json
import os
import re
import winreg
import tempfile
import zipfile
import subprocess
import shutil
from pathlib import Path
import importlib
from huggingface_hub import HfApi, list_repo_files, hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
import gradio_tabs.kkeve as kkeve
import gradio_tabs.sbv2 as sbv2
# ==============================================================================

# ============================================================================ #
#                          [ モデルダウンロード関数 ]                          #
# ============================================================================ #
# リポジトリのcommit hash（SHA）を取得
api = HfApi()
def get_repository_sha(repo_id, repo_type="model"):
	try:
		info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
		return info.sha
	except (
		HfHubHTTPError,
		RepositoryNotFoundError,
		RevisionNotFoundError,
		ConnectionError,
		TimeoutError,
	) as e:
		# 接続不可・存在しない・一時的エラー → スキップ
		print(f"[SKIP] {repo_id}: {e}")
		return None

# ------------------------------------------------------------------------------
# モデルダウンロード
def download_all_files(repo_id: str, local_dir: str, revision: str = "main"):

	# 保存先ディレクトリを作り直す
	if os.path.exists(local_dir):
		shutil.rmtree(local_dir)
	os.makedirs(local_dir, exist_ok=True)

	# リポジトリ内のファイル一覧を取得
	file_list = list_repo_files(repo_id, revision=revision)
	#print(f"[{repo_id}@{revision}] に {len(file_list)} 件のファイルが見つかりました。")

	for file_path in file_list:
		# すでにモデル名のフォルダがあればダウンロードをスキップ
		base_name = os.path.basename(file_path)
		if base_name not in conf['model_target']:
			continue

		# 各ファイルをローカルに保存（フォルダ構造を再現）
		local_path = os.path.dirname(os.path.join(local_dir, file_path))
		print(f"モデルファイルチェック: {file_path}")
		cache_path = hf_hub_download(
			repo_id=repo_id,
			filename=file_path,
			revision=revision
		)
		#print(f"cache_path:{cache_path}  file_path:{file_path}\n")
		os.makedirs(local_path, exist_ok=True)
		shutil.copy(cache_path, local_path)

	print("モデルファイルチェック完了")

# ------------------------------------------------------------------------------
# 既存の name を取得
def get_existing_names(cursor):
	cursor.execute("SELECT name FROM model")
	rows = cursor.fetchall()
	return {row[0] for row in rows}

# データベースのmodelテーブルに登録が無ければ追加
def process_directory(base_dir, db_path):
	# データベースに接続
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	existing_names = get_existing_names(cursor)

	# ディレクトリ内のフォルダ取得
	for name in os.listdir(base_dir):
		dir_path = os.path.join(base_dir, name)
		if not os.path.isdir(dir_path):
			continue  # サブディレクトリでない場合は無視

		if name in existing_names:
			continue  # すでに登録済み

		config_path = os.path.join(dir_path, "config.json")
		if not os.path.exists(config_path):
			print(f"config.json が見つかりません: {config_path}")
			continue

		try:
			with open(config_path, "r", encoding="utf-8") as f:
				config = json.load(f)

			model_id    = config.get("id")
			sort        = config.get("sort")
			description = config.get("description")
			character   = config.get("character")

			if model_id is None:
				print(f"id が config.json に存在しません: {config_path}")
				continue

			# データベースに挿入
			cursor.execute(
				"INSERT INTO model (id, sort, name, description, character) VALUES (?, ?, ?, ?, ?)",
				(model_id, sort, name, description, character)
			)
			print(f"追加: {name}")

		except Exception as e:
			print(f"エラー発生 ({config_path}): {e}")

	# 保存して接続終了
	conn.commit()
	conn.close()

# ------------------------------------------------------------------------------
# データベースのmodelテーブルにしかないレコードを削除
def delete_nonexistent_dirs(base_dir, db_path):
	# データベース接続
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	# 実際に存在するディレクトリ名の一覧を取得
	actual_dirs = {name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))}

	# データベース上の name 一覧を取得
	cursor.execute("SELECT name FROM model WHERE id != '0000'")
	db_names = [row[0] for row in cursor.fetchall()]

	# 存在しないディレクトリに対応する name を削除対象に
	to_delete = [name for name in db_names if name not in actual_dirs]

	for name in to_delete:
		cursor.execute("DELETE FROM model WHERE name = ?", (name,))
		print(f"削除: {name}")

	# 変更を保存して接続終了
	conn.commit()
	conn.close()

# ============================================================================ #
#                           [ インターフェース関数 ]                           #
# ============================================================================ #

# キャッシュ取得
def get_cache(tab, field):
	global conf
	conn = sqlite3.connect(conf['db_path'])
	cursor = conn.cursor()
	cursor.execute(f"SELECT value FROM {conf['tbl_gradio']} WHERE tab = ? AND field = ?", (tab, field))
	result = cursor.fetchone()
	conn.close()
	if not result:
		return None
	return result[0]

# ------------------------------------------------------------------------------
# キャッシュ変更
def update_cache(tab, field, value):
	global conf
	# JSON文字列化（日本語OK）
	value_str = json.dumps(value, ensure_ascii=False)
	# 文字列の両端のダブルクォートを削除
	if isinstance(value, str):
		value_str = value_str.strip('"')

	conn = sqlite3.connect(conf['db_path'])
	cursor = conn.cursor()
	cursor.execute(f"UPDATE {conf['tbl_gradio']} SET value = ? WHERE tab = ? AND field = ?", (value_str, tab, field))
	conn.commit()
	conn.close()

# ------------------------------------------------------------------------------
# 言語別メッセージ取得
def get_message(target, id, cd, **kwargs):
	global conf
	conn = sqlite3.connect(conf['db_path'])
	cursor = conn.cursor()
	cursor.execute("SELECT value FROM language WHERE target = ? AND id = ? AND cd = ?", (target, id, cd))
	template = cursor.fetchone()[0]
	return template.format(**kwargs)

# ------------------------------------------------------------------------------
# 言語変更
def change_language(language):
	global conf

	# 言語が変わらない場合は何も変更しない
	if conf['language'] == language:
		return (
			conf['language'], 
			gr.update(), # title_markdown
			*[gr.update() for _ in conf["tab_list"]] # tabs
			#*[gr.TabItem.update() for _ in conf["tab_list"]] # tabs
		)

	# 言語更新
	update_cache("kkeve", "language", language)
	conf['language'] = language

	# タイトルの更新
	results = [
		conf['language'], 
		gr.update(value=f"# {get_message('kkeve', 'label_title', conf['language'])}")
	]

	# タブラベルの更新
	for tab in conf["tab_list"]:
		results.append(
			#gr.TabItem.update(label=get_message('kkeve', tab, conf['language']))
			gr.update(label=get_message('kkeve', tab, conf['language']))
		)

	return tuple(results)

# ------------------------------------------------------------------------------
# Gradioインターフェース
def create_interface():
	global conf
	kkeve.main(conf)
	sbv2.main(conf)

	# 言語
	conf['language'] = get_cache("kkeve", "language")
	if conf['language'] is None:
		conf['language'] = locale.getdefaultlocale()[0][:2]
		update_cache("kkeve", "language", conf['language'])



	with gr.Blocks(theme='NoCrypt/miku') as app:



		language_state = gr.State(value=conf["language"])# gr.Blocksより下に書く

		gr.HTML("""
			<style>
				#List {
					.cell-selected>div>button {display:none;}
					tr:has(td.cell-selected){background-color: var(--color-accent);}
				}
				#Model textarea:disabled, 
				#Id textarea:disabled, 
				#Output textarea:disabled{opacity: 0.3;}
				.lang-row {
					justify-content: space-between !important; /* 左右に分ける */
					align-items: center;
				}
				#language_dropdown {width: 150px !important;}
				.no-border,
				.no-border * {
					border: none !important;
					outline: none !important;
					box-shadow: none !important;
				}
			</style>
		""")
		with gr.Row(elem_classes="lang-row"):
			title_markdown = gr.Markdown(f"# {get_message('kkeve', 'label_title', conf['language'])}")

			# 言語
			language_dropdown = gr.Dropdown(
				choices=[("日本語", "ja"), ("English", "en")], 
				value=conf['language'], 
				label="Language / 言語"
			)

		tabs = {}
		with gr.Tabs():
			# conf["tab_list"]作成するタブリスト
			for tab in conf["tab_list"]:
				# get_message('kkeve', tab, conf['language'])の名前でタブ作成
				with gr.TabItem(get_message('kkeve', tab, conf['language'])) as t:

					# conf["tab_module"][tab]モジュール内のconf['tab_interface'][tab]関数実行（タブ内容出力）
					getattr(conf["tab_module"][tab], conf['tab_interface'][tab])(language_state)

				# タブの実態を格納
				tabs[tab] = t

		# ----------------------------------------------------------------------
		# イベントハンドラ
		outputs=[
			language_state, 
			title_markdown
		]

		for tab in conf["tab_list"]:
			outputs.append(tabs[tab])

		language_dropdown.change(
			fn=change_language,
			inputs=[language_dropdown], 
			outputs=outputs
		)
		return app

# ============================================================================ #
#                                [ メイン関数 ]                                #
# ============================================================================ #
def main(config):
	global conf
	conf = config

	# --------------------------------------------------------------------------
	# 前回のリポジトリハッシュ取得
	conf['repository_sha'] = get_cache("sbv2", "repository_sha")

	# 現在のリポジトリハッシュ取得
	sha = get_repository_sha(conf['model_repository'])

	# リポジトリの変更を検知したらモデルチェック
	if sha is not None and sha != conf['repository_sha']:

		# モデルダウンロード
		download_all_files(repo_id=conf['model_repository'], local_dir=conf['assets_root'])

		# データベースのmodelテーブルに登録が無ければ追加
		process_directory(conf['assets_root'], conf['db_path'])

		# データベースのmodelテーブルにしかないレコードを削除
		delete_nonexistent_dirs(conf['assets_root'], conf['db_path'])

		# リポジトリハッシュ保存
		update_cache("sbv2", "repository_sha", sha)

	# --------------------------------------------------------------------------
	app = create_interface()
	app.launch(inbrowser=True)

if __name__ == "__main__":

	# コンフィグファイルを読み込む
	with open('config.json', 'r', encoding='utf-8') as f:
		config = json.load(f)

	# タブキーとモジュールを繋ぐ
	config["tab_module"] = {
		"kkeve": kkeve, 
		"sbv2": sbv2 
	}

	main(config)
