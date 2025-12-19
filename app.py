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
import gradio_tabs.kkeve as kkeve
# ==============================================================================
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
	update_cache("inference", "language", language)
	conf['language'] = language

	# タイトルの更新
	results = [
		conf['language'], 
		gr.update(value=f"# {get_message('inference', 'label_title', conf['language'])}")
	]

	# タブラベルの更新
	for tab in conf["tab_list"]:
		results.append(
			#gr.TabItem.update(label=get_message('inference', tab, conf['language']))
			gr.update(label=get_message('inference', tab, conf['language']))
		)

	return tuple(results)

# ------------------------------------------------------------------------------
# Gradioインターフェース
def create_interface():
	global conf
	kkeve.main(conf)

	# 言語
	conf['language'] = get_cache("inference", "language")
	if conf['language'] is None:
		conf['language'] = locale.getdefaultlocale()[0][:2]
		update_cache("inference", "language", conf['language'])



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
			</style>
		""")
		with gr.Row(elem_classes="lang-row"):
			title_markdown = gr.Markdown(f"# {get_message('inference', 'label_title', conf['language'])}")

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
				# get_message('inference', tab, conf['language'])の名前でタブ作成
				with gr.TabItem(get_message('inference', tab, conf['language'])) as t:

					# conf["tab_module"][tab]モジュール内のconf['tab_interface'][tab]関数実行（タブ内容出力）
					getattr(conf["tab_module"][tab], conf['tab_interface'][tab])(language_state)

				# タブの実態を格納
				tabs[tab] = t

		'''

	config["tab_module"] = {
		"kkeve": kkeve
	}
	"tab_list": ["kkeve"], 
	"tab_interface": {"kkeve": "create_interface"}, 
INSERT INTO 'language' VALUES ("inference", "kkeve", "ja", "コイカツ外部音声");
INSERT INTO 'language' VALUES ("inference", "sbv2", "ja", "音声合成");
INSERT INTO 'language' VALUES ("inference", "kkeve", "en", "Koikatsu External Voice");
INSERT INTO 'language' VALUES ("inference", "sbv2", "en", "Text To Speech");

		with gr.Tabs():
			for tab in config["tab_list"]:
				with gr.Tab(get_message('inference', tab, conf['language'])):

		dispatcher = {
		    "hello": hello,
		    "bye": bye,
		}

		func_name = "hello"
		dispatcher[func_name]()   # 実行
		'''

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

	app = create_interface()
	app.launch(inbrowser=True)

if __name__ == "__main__":

	# コンフィグファイルを読み込む
	with open('config.json', 'r', encoding='utf-8') as f:
		config = json.load(f)

	# タブキーとモジュールを繋ぐ
	config["tab_module"] = {
		"kkeve": kkeve
	}

	main(config)
