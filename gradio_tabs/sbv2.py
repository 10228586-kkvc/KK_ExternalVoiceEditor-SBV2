# ┌──────────────────────────────────────
# │  KK_ExternalVoiceEditor v1.0.0 (2025.06.01)
# └──────────────────────────────────────
# ==============================================================================
import datetime
import json
from typing import Optional
from pathlib import Path
import gradio as gr

import locale
import sqlite3
import pandas as pd
import random
import string
import os
import re
import winreg
import tempfile
import zipfile
import subprocess
import shutil

from style_bert_vits2.constants import (
	DEFAULT_ASSIST_TEXT_WEIGHT, 
	DEFAULT_LENGTH, 
	DEFAULT_LINE_SPLIT, 
	DEFAULT_NOISE, 
	DEFAULT_NOISEW, 
	DEFAULT_SDP_RATIO, 
	DEFAULT_SPLIT_INTERVAL, 
	DEFAULT_STYLE, 
	DEFAULT_STYLE_WEIGHT, 
	GRADIO_THEME, 
	Languages, 
)
from style_bert_vits2.logging import logger
from style_bert_vits2.models.infer import InvalidToneError
from style_bert_vits2.nlp.japanese import pyopenjtalk_worker as pyopenjtalk
from style_bert_vits2.nlp.japanese.g2p_utils import g2kata_tone, kata_tone2phone_tone
from style_bert_vits2.nlp.japanese.normalizer import normalize_text
from style_bert_vits2.tts_model import TTSModelHolder

pyopenjtalk.initialize_worker()
# ==============================================================================



languages = [lang.value for lang in Languages]

initial_text = "こんにちは、初めまして。あなたの名前はなんていうの？"

examples = [
	[initial_text, "JP"],
	[
		"""あなたがそんなこと言うなんて、私はとっても嬉しい。
あなたがそんなこと言うなんて、私はとっても怒ってる。
あなたがそんなこと言うなんて、私はとっても驚いてる。
あなたがそんなこと言うなんて、私はとっても辛い。""",
		"JP",
	],
	[  # ChatGPTに考えてもらった告白セリフ
		"""私、ずっと前からあなたのことを見てきました。あなたの笑顔、優しさ、強さに、心惹かれていたんです。
友達として過ごす中で、あなたのことがだんだんと特別な存在になっていくのがわかりました。
えっと、私、あなたのことが好きです！もしよければ、私と付き合ってくれませんか？""",
		"JP",
	],
	[  # 夏目漱石『吾輩は猫である』
		"""吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。なんでも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで初めて人間というものを見た。しかもあとで聞くと、それは書生という、人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕まえて煮て食うという話である。""",
		"JP",
	],
	[  # 梶井基次郎『桜の樹の下には』
		"""桜の樹の下には屍体が埋まっている！これは信じていいことなんだよ。
何故って、桜の花があんなにも見事に咲くなんて信じられないことじゃないか。俺はあの美しさが信じられないので、このにさんにち不安だった。
しかしいま、やっとわかるときが来た。桜の樹の下には屍体が埋まっている。これは信じていいことだ。""",
		"JP",
	],
	[  # ChatGPTと考えた、感情を表すセリフ
		"""やったー！テストで満点取れた！私とっても嬉しいな！
どうして私の意見を無視するの？許せない！ムカつく！あんたなんか死ねばいいのに。
あはははっ！この漫画めっちゃ笑える、見てよこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しい。""",
		"JP",
	],
	[  # 上の丁寧語バージョン
		"""やりました！テストで満点取れましたよ！私とっても嬉しいです！
どうして私の意見を無視するんですか？許せません！ムカつきます！あんたなんか死んでください。
あはははっ！この漫画めっちゃ笑えます、見てくださいこれ、ふふふ、あはは。
あなたがいなくなって、私は一人になっちゃって、泣いちゃいそうなほど悲しいです。""",
		"JP",
	],
	[  # ChatGPTに考えてもらった音声合成の説明文章
		"""音声合成は、機械学習を活用して、テキストから人の声を再現する技術です。この技術は、言語の構造を解析し、それに基づいて音声を生成します。
この分野の最新の研究成果を使うと、より自然で表現豊かな音声の生成が可能である。深層学習の応用により、感情やアクセントを含む声質の微妙な変化も再現することが出来る。""",
		"JP",
	],
	[
		"Speech synthesis is the artificial production of human speech. A computer system used for this purpose is called a speech synthesizer, and can be implemented in software or hardware products.",
		"EN",
	],
	[
		"语音合成是人工制造人类语音。用于此目的的计算机系统称为语音合成器，可以通过软件或硬件产品实现。",
		"ZH",
	],
]

# ------------------------------------------------------------------------------
# キャッシュ取得
def get_cache(tab, field):
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
	# JSON文字列化（日本語OK）
	#value_str = json.dumps(value, ensure_ascii=False)
	# 文字列の両端のダブルクォートを削除
	#if isinstance(value, str):
	#	value_str = value_str.strip('"')

	conn = sqlite3.connect(conf['db_path'])
	cursor = conn.cursor()
	cursor.execute(f"UPDATE {conf['tbl_gradio']} SET value = ? WHERE tab = ? AND field = ?", (value, tab, field))
	conn.commit()
	conn.close()

# ------------------------------------------------------------------------------
# 言語別メッセージ取得
def get_message(target, id, cd, **kwargs):
	conn = sqlite3.connect(conf['db_path'])
	cursor = conn.cursor()
	cursor.execute("SELECT value FROM language WHERE target = ? AND id = ? AND cd = ?", (target, id, cd))
	template = cursor.fetchone()[0]
	return template.format(**kwargs)

# ------------------------------------------------------------------------------
# 言語別メッセージ群取得
def get_messages(target, cd):
	conn = sqlite3.connect(conf['db_path'])
	df = pd.read_sql_query(f"SELECT id, value FROM language WHERE target = '{target}' AND cd = '{cd}'", conn)
	conn.close()
	return df

# ------------------------------------------------------------------------------
# 文字列カット
def truncate_text(text: str, encoding: str = "utf-8", max_bytes: int = 80) -> str:
	# 改行を削除
	text = text.replace("\r", "").replace("\n", "")

	encoded = text.encode(encoding)

	# bytes以内ならそのまま返す
	if len(encoded) <= max_bytes:
		return text

	# 「…」を付ける分のbyte数
	ellipsis = "…"
	ellipsis_bytes = ellipsis.encode(encoding)
	limit = max_bytes - len(ellipsis_bytes)

	# byte単位で切り詰め（文字途中で切れないように）
	truncated_bytes = encoded[:limit]
	truncated_text = truncated_bytes.decode(encoding, errors="ignore")

	return truncated_text + ellipsis

# ------------------------------------------------------------------------------
# ランダム文字列生成
def generate_random_id(length=8):
	chars = string.ascii_letters + string.digits  # 英大文字・小文字 + 数字
	return ''.join(random.choices(chars, k=length))

# ------------------------------------------------------------------------------
# ユニークなIDを生成
def generate_unique_filename(output_path, model_name, character, audio_type, length=4):
	while True:
		new_id = generate_random_id(length)
		filename = f"{model_name}-{character}-{new_id}.{audio_type}"
		if not os.path.exists(os.path.join(output_path, filename)):
			return filename

# ------------------------------------------------------------------------------
# 言語変更

def change_language(language):

	print(language)

	global conf
	if conf['language_sbv2'] == language: 
		components = [
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update(), 
			gr.update() 
		]

		return components
	else: 
		update_cache("sbv2", "language_sbv2", language)
		conf['language_sbv2'] = language

	components = [
		gr.update(label=get_message('sbv2', 'label_model_list', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_pth_files', conf['language'])), 
		gr.update(value=get_message('sbv2', 'button_update', conf['language'])), 
		gr.update(value=get_message('sbv2', 'button_load', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_text', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_pitch_scale', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_intonation_scale', conf['language'])), 
		gr.update(label=get_message('sbv2', 'checkbox_line_split', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_split_interval', conf['language'])), 
		gr.update(
			label=get_message('sbv2', 'label_tone', conf['language']), 
			info=get_message('sbv2', 'label_tone_info', conf['language'])
		), 
		gr.update(label=get_message('sbv2', 'checkbox_use_tone', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_bert_language', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_speaker', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_advanced_settings', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_sdp_ratio', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_noise_scale', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_noise_scale_w', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_length_scale', conf['language'])), 
		gr.update(label=get_message('sbv2', 'checkbox_use_assist_text', conf['language'])), 
		gr.update(
			label=get_message('sbv2', 'label_assist_text', conf['language']),
			placeholder=get_message('sbv2', 'label_assist_text_placeholder', conf['language']),
			info=get_message('sbv2', 'label_assist_text_info', conf['language'])
		), 
		gr.update(label=get_message('sbv2', 'label_assist_text_weight', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_style_mode', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_style', conf['language'], default_style=DEFAULT_STYLE)), 
		gr.update(label=get_message('sbv2', 'label_style_weight', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_ref_audio_path', conf['language'])), 
		gr.update(value=get_message('sbv2', 'button_tts', conf['language'])), 
		gr.update(label=get_message('sbv2', 'label_message', conf['language'])) 
	]
	return tuple(components)

# ------------------------------------------------------------------------------
# フィールド変更
def update_model_name(model_name):
	global model_holder
	update_cache("sbv2", "model_name", model_name)
	return model_holder.update_model_files_for_gradio(model_name)

def update_model_path(model_path):
	update_cache("sbv2", "model_path", model_path)
	return gr.update(interactive=False, value=get_message('sbv2', 'button_tts', conf['language']))

def update_load_button(model_name, model_path):
	global model_holder
	return model_holder.get_model_for_gradio(model_name, model_path)

def update_text(text):
	update_cache("sbv2", "text", text)

def update_pitch_scale(pitch_scale):
	update_cache("sbv2", "pitch_scale", pitch_scale)

def update_intonation_scale(intonation_scale):
	update_cache("sbv2", "intonation_scale", intonation_scale)

def update_line_split(line_split: bool):
	update_cache("sbv2", "line_split", line_split)
	return gr.Slider(visible=line_split)

def update_split_interval(split_interval):
	update_cache("sbv2", "split_interval", split_interval)

def update_tone(tone):
	update_cache("sbv2", "tone", tone)

def update_use_tone(use_tone: bool):
	update_cache("sbv2", "use_tone", use_tone)
	if use_tone:
		return gr.Checkbox(value=False)
	return gr.Checkbox()

def update_language(language):
	update_cache("sbv2", "language", language)

def update_speaker(speaker):
	update_cache("sbv2", "speaker", speaker)

def update_sdp_ratio(sdp_ratio):
	update_cache("sbv2", "sdp_ratio", sdp_ratio)

def update_noise_scale(noise_scale):
	update_cache("sbv2", "noise_scale", noise_scale)

def update_noise_scale_w(noise_scale_w):
	update_cache("sbv2", "noise_scale_w", noise_scale_w)

def update_length_scale(length_scale):
	update_cache("sbv2", "length_scale", length_scale)

def update_use_assist_text(use_assist_text):
	update_cache("sbv2", "use_assist_text", use_assist_text)

def update_assist_text(assist_text):
	update_cache("sbv2", "assist_text", assist_text)

def update_assist_text_weight(assist_text_weight):
	update_cache("sbv2", "assist_text_weight", assist_text_weight)

def update_style_mode(style_mode):
	update_cache("sbv2", "style_mode", style_mode)

def update_style(style):
	update_cache("sbv2", "style", style)

def update_style_weight(style_weight):
	update_cache("sbv2", "style_weight", style_weight)

def update_ref_audio_path(ref_audio_path):
	update_cache("sbv2", "ref_audio_path", ref_audio_path)

def update_character_dropdown(model_name, character, audio_type):
	update_cache("sbv2", "audio_type", audio_type)

	output_path = "/".join([get_message('sbv2', 'default_path', conf['language']), model_name])

	return gr.update(value=generate_unique_filename(output_path, model_name, character, audio_type))


def update_audio_type(model_name, character, audio_type):
	update_cache("sbv2", "audio_type", audio_type)

	output_path = "/".join([get_message('sbv2', 'default_path', conf['language']), model_name])

	return gr.update(value=generate_unique_filename(output_path, model_name, character, audio_type))



# ------------------------------------------------------------------------------
# ドロップダウン
def get_dropdown_options():
	with sqlite3.connect(conf['db_path']) as conn:
		character_df = pd.read_sql_query(f"SELECT id, 'c' || id || ' ' || name AS view FROM {conf['tbl_character']}", conn)
		category_df  = pd.read_sql_query(f"SELECT id, name FROM {conf['tbl_category']}",  conn)

	character_options = [(row["view"], row["id"]) for _, row in character_df.iterrows()]
	category_options  = [(row["name"], row["id"]) for _, row in category_df.iterrows()]
	return character_options, category_options

# ------------------------------------------------------------------------------
# モデル初期化（強制ロード）
def initialize_model(model_name, model_path): 
	global model_holder

	style, tts_button, speaker = model_holder.get_model_for_gradio(model_name, model_path)
	style.value      = get_cache("sbv2", "style")
	tts_button.value = get_cache("sbv2", "tts_button")
	speaker.value    = get_cache("sbv2", "speaker")
	#style   = gr.update(value=get_cache("sbv2", "style"))
	#speaker = gr.update(value=get_cache("sbv2", "speaker"))

	#print(type(style), type(tts_button), type(speaker))

	return style, tts_button, speaker

# ------------------------------------------------------------------------------
# モデルファイル有効化切替
def make_interactive():
	return gr.update(interactive=True, value="音声合成")

# ------------------------------------------------------------------------------
# モデルファイル無効化切替
def make_non_interactive():
	return gr.update(interactive=False, value="音声合成（モデルをロードしてください）")

# ------------------------------------------------------------------------------
# スタイルの指定方法切替
def gr_util(item):
	#if item == "プリセットから選ぶ":
	if item == 0:
		return (gr.update(visible=True), gr.Audio(visible=False, value=None))
	else:
		return (gr.update(visible=False), gr.update(visible=True))

# ------------------------------------------------------------------------------
# Gradioインターフェース
#def create_inference_app(model_holder: TTSModelHolder) -> gr.Blocks:
def create_inference_app(language_state) -> gr.Blocks:

	global conf

	# 言語
	conf['language_sbv2'] = conf['language']
	update_cache("sbv2", "language_sbv2", conf['language_kkeve'])

	import torch
	from style_bert_vits2.constants import Languages
	from style_bert_vits2.nlp import bert_models

	# BERT日本語解析モデル
	bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
	bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

	global model_holder
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_holder = TTSModelHolder(Path(conf['assets_root']), device)

	def tts_fn(
		model_name, 
		model_path, 
		text, 
		language, 
		reference_audio_path, 
		sdp_ratio, 
		noise_scale, 
		noise_scale_w, 
		length_scale, 
		line_split, 
		split_interval, 
		assist_text, 
		assist_text_weight, 
		use_assist_text, 
		style, 
		style_weight, 
		kata_tone_json_str, 
		use_tone, 
		speaker, 
		pitch_scale, 
		intonation_scale, 
		audio_type, 
	):
		model_holder.get_model(model_name, model_path)
		assert model_holder.current_model is not None

		wrong_tone_message = ""
		kata_tone: Optional[list[tuple[str, int]]] = None
		if use_tone and kata_tone_json_str != "":

			if language != "JP":
				logger.warning("Only Japanese is supported for tone generation.")
				wrong_tone_message = "アクセント指定は現在日本語のみ対応しています。"

			if line_split:
				logger.warning("Tone generation is not supported for line split.")
				wrong_tone_message = (
					"アクセント指定は改行で分けて生成を使わない場合のみ対応しています。"
				)


			try:
				kata_tone = []
				json_data = json.loads(kata_tone_json_str)
				# tupleを使うように変換
				for kana, tone in json_data:
					assert isinstance(kana, str) and tone in (0, 1), f"{kana}, {tone}"
					kata_tone.append((kana, tone))

			except Exception as e:
				logger.warning(f"Error occurred when parsing kana_tone_json: {e}")
				wrong_tone_message = f"アクセント指定が不正です: {e}"
				kata_tone = None

		# toneは実際に音声合成に代入される際のみnot Noneになる
		tone: Optional[list[int]] = None
		if kata_tone is not None:
			phone_tone = kata_tone2phone_tone(kata_tone)
			tone = [t for _, t in phone_tone]

		speaker_id = model_holder.current_model.spk2id[speaker]

		start_time = datetime.datetime.now()

		try:
			sr, audio = model_holder.current_model.infer(
				text=text,
				language=language,
				reference_audio_path=reference_audio_path,
				sdp_ratio=sdp_ratio,
				noise=noise_scale,
				noise_w=noise_scale_w,
				length=length_scale,
				line_split=line_split,
				split_interval=split_interval,
				assist_text=assist_text,
				assist_text_weight=assist_text_weight,
				use_assist_text=use_assist_text,
				style=style,
				style_weight=style_weight,
				given_tone=tone,
				speaker_id=speaker_id,
				pitch_scale=pitch_scale,
				intonation_scale=intonation_scale,
			)
		except InvalidToneError as e:
			logger.error(f"Tone error: {e}")
			return f"Error: アクセント指定が不正です:\n{e}", None, kata_tone_json_str
		except ValueError as e:
			logger.error(f"Value error: {e}")
			return f"Error: {e}", None, kata_tone_json_str

		end_time = datetime.datetime.now()
		duration = (end_time - start_time).total_seconds()

		if tone is None and language == "JP":
			# アクセント指定に使えるようにアクセント情報を返す
			norm_text = normalize_text(text)
			kata_tone = g2kata_tone(norm_text)
			kata_tone_json_str = json.dumps(kata_tone, ensure_ascii=False)
		elif tone is None:
			kata_tone_json_str = ""
		message = f"Success, time: {duration} seconds."
		if wrong_tone_message != "":
			message = wrong_tone_message + "\n" + message

		#output_path = os.path.join(get_message('sbv2', 'default_path', conf['language']), model_name)
		output_path = "/".join([get_message('sbv2', 'default_path', conf['language']), model_name])

		dir_path = os.path.join(conf['assets_root'], model_name)
		config_path = os.path.join(dir_path, "config.json")

		with open(config_path, "r", encoding="utf-8") as f:
			config = json.load(f)
		character = config.get("character")



		#return message, (sr, audio), kata_tone_json_str

		components = [
			message, 
			(sr, audio), 
			kata_tone_json_str, 
			gr.update(visible=True), 
			gr.update(value=character), 
			gr.update(value=truncate_text(text)), 
			gr.update(value=output_path), 
			gr.update(value=generate_unique_filename(output_path, model_name, character, audio_type)), 
		]

		return tuple(components)






	model_names = []
	for model_name in model_holder.model_names: 

		dir_path = os.path.join(conf['assets_root'], model_name)
		config_path = os.path.join(dir_path, "config.json")

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

			model_names.append((f"{model_name}({description})", model_name))

		except Exception as e:
			print(f"エラー発生 ({config_path}): {e}")



	# 音声合成モデルチェック
	#model_names = model_holder.model_names
	if len(model_names) == 0:
		logger.error(
			f"モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
		)
		with gr.Blocks() as app:
			gr.Markdown(
				f"Error: モデルが見つかりませんでした。{model_holder.root_dir}にモデルを置いてください。"
			)
		return app

	df_model_name = get_cache("sbv2", "model_name")
	df_model_path = get_cache("sbv2", "model_path")
	initial_id = 0
	if df_model_name is None or not any(df_model_name == item[1] for item in model_names):
		df_model_name = model_names[initial_id][1]

	initial_pth_files = [
		#str(f) for f in model_holder.model_files_dict[model_names[initial_id][1]]
		str(f) for f in model_holder.model_files_dict[df_model_name]
	]
	if df_model_path is None or not df_model_path in initial_pth_files:
		df_model_path = initial_pth_files[0]

	df_speaker = get_cache("sbv2", "speaker")
	df_style = get_cache("sbv2", "style")


	# --------------------------------------------------------------------------
	with gr.Blocks(theme=GRADIO_THEME) as app:
		with gr.Row():
			with gr.Column():
				with gr.Row():
					with gr.Column(scale=3):

						# モデル一覧
						model_name = gr.Dropdown(
							label=get_message('sbv2', 'label_model_list', conf['language']), 
							choices=model_names, 
							#value=model_names[initial_id], 
							value=df_model_name
						)

						# モデルファイル
						model_path = gr.Dropdown(
							label=get_message('sbv2', 'label_pth_files', conf['language']), 
							choices=initial_pth_files, 
							#value=initial_pth_files[0], 
							value=df_model_path 
						)

					# 更新
					refresh_button = gr.Button(get_message('sbv2', 'button_update', conf['language']), scale=1, visible=True)

					# ロード
					load_button = gr.Button(get_message('sbv2', 'button_load', conf['language']), scale=1, variant="primary")

				# テキスト
				text_input = gr.TextArea(
					label=get_message('sbv2', 'label_text', conf['language']), 
					#value=initial_text
					value=get_cache("sbv2", "text")
				)

				# 音高(1以外では音質劣化)
				pitch_scale = gr.Slider(
					minimum=0.8, 
					maximum=1.5, 
					value=float(get_cache("sbv2", "pitch_scale")), 
					step=0.05, 
					label=get_message('sbv2', 'label_pitch_scale', conf['language']), 
				)

				# 抑揚(1以外では音質劣化)
				intonation_scale = gr.Slider(
					minimum=0,
					maximum=2,
					value=float(get_cache("sbv2", "intonation_scale")), 
					step=0.1,
					label=get_message('sbv2', 'label_intonation_scale', conf['language']),
				)

				# 改行で分けて生成（分けたほうが感情が乗ります）
				line_split = gr.Checkbox(
					label=get_message('sbv2', 'checkbox_line_split', conf['language']),
					#value=DEFAULT_LINE_SPLIT,
					value=bool(int(get_cache("sbv2", "line_split"))), 
				)

				# 改行ごとに挟む無音の長さ（秒）
				split_interval = gr.Slider(
					minimum=0.0,
					maximum=2,
					#value=DEFAULT_SPLIT_INTERVAL,
					value=float(get_cache("sbv2", "split_interval")), 
					visible=bool(int(get_cache("sbv2", "line_split"))), 
					step=0.1,
					label=get_message('sbv2', 'label_split_interval', conf['language']),
				)

				# アクセント調整（数値は 0=低 か1=高 のみ）
				# 改行で分けない場合のみ使えます。万能ではありません。
				tone = gr.Textbox(
					label=get_message('sbv2', 'label_tone', conf['language']), 
					info=get_message('sbv2', 'label_tone_info', conf['language']), 
					value=get_cache("sbv2", "tone"), 
				)

				# アクセント調整を使う
				use_tone = gr.Checkbox(
					label=get_message('sbv2', 'checkbox_use_tone', conf['language']), 
					#value=False
					value=bool(int(get_cache("sbv2", "use_tone"))), 
				)

				# Language
				language = gr.Dropdown(
					choices=languages, 
					#value="JP", 
					value=get_cache("sbv2", "language"), 
					label=get_message('sbv2', 'label_bert_language', conf['language'])
				)

				# 話者
				speaker = gr.Dropdown(
					label=get_message('sbv2', 'label_speaker', conf['language']),
					#value=get_cache("sbv2", "speaker"), 
				)

				# 詳細設定
				accordion_advanced_settings = gr.Accordion(
					label=get_message('sbv2', 'label_advanced_settings', conf['language']), 
					open=False
				)
				with accordion_advanced_settings:

					# SDP Ratio
					sdp_ratio = gr.Slider(
						minimum=0,
						maximum=1,
						value=DEFAULT_SDP_RATIO,
						#value=get_cache("sbv2", "sdp_ratio"), 
						step=0.1,
						label=get_message('sbv2', 'label_sdp_ratio', conf['language']),
					)

					# Noise
					noise_scale = gr.Slider(
						minimum=0.1,
						maximum=2,
						value=DEFAULT_NOISE,
						#value=get_cache("sbv2", "noise_scale"), 
						step=0.1,
						label=get_message('sbv2', 'label_noise_scale', conf['language']),
					)

					# Noise_W
					noise_scale_w = gr.Slider(
						minimum=0.1,
						maximum=2,
						value=DEFAULT_NOISEW,
						#value=get_cache("sbv2", "noise_scale_w"), 
						step=0.1,
						label=get_message('sbv2', 'label_noise_scale_w', conf['language']),
					)

					# Length
					length_scale = gr.Slider(
						minimum=0.1,
						maximum=2,
						value=DEFAULT_LENGTH,
						#value=get_cache("sbv2", "length_scale"), 
						step=0.1,
						label=get_message('sbv2', 'label_length_scale', conf['language']),
					)

					# Assist textを使う
					use_assist_text = gr.Checkbox(
						label=get_message('sbv2', 'checkbox_use_assist_text', conf['language']), 
						value=False
						#value=get_cache("sbv2", "use_assist_text"), 
					)

					# Assist text
					# どうして私の意見を無視するの？許せない、ムカつく！死ねばいいのに。
					# このテキストの読み上げと似た声音・感情になりやすくなります。ただ抑揚やテンポ等が犠牲になる傾向があります。
					assist_text = gr.Textbox(
						label=get_message('sbv2', 'label_assist_text', conf['language']),
						placeholder=get_message('sbv2', 'label_assist_text_placeholder', conf['language']),
						info=get_message('sbv2', 'label_assist_text_info', conf['language']),
						visible=False,
						#value=get_cache("sbv2", "assist_text"), 
					)

					# Assist textの強さ
					assist_text_weight = gr.Slider(
						minimum=0,
						maximum=1,
						value=DEFAULT_ASSIST_TEXT_WEIGHT,
						#value=get_cache("sbv2", "assist_text_weight"), 
						step=0.1,
						label=get_message('sbv2', 'label_assist_text_weight', conf['language']),
						visible=False,
					)


			with gr.Column():

				# スタイルの指定方法
				# プリセットから選ぶ
				# ["プリセットから選ぶ", "音声ファイルを入力"]
				style_mode = gr.Radio(
					[[get_message('sbv2', 'radio_preset', conf['language']), 0], [get_message('sbv2', 'radio_voice_file', conf['language']), 1]],
					label=get_message('sbv2', 'label_style_mode', conf['language']),
					#value=get_message('sbv2', 'radio_preset', conf['language']),
				)

				# スタイル（{DEFAULT_STYLE}が平均スタイル）
				# モデルをロードしてください
				# モデルをロードしてください
				style = gr.Dropdown(
					label=get_message('sbv2', 'label_style', conf['language'], default_style=DEFAULT_STYLE),
					choices=[get_message('sbv2', 'button_load_error', conf['language'])],
					#value=get_message('sbv2', 'button_load_error', conf['language']),
				)

				# スタイルの強さ（声が崩壊したら小さくしてください）
				style_weight = gr.Slider(
					minimum=0,
					maximum=20,
					value=DEFAULT_STYLE_WEIGHT,
					#value=get_cache("sbv2", "style_weight"), 
					step=0.1,
					label=get_message('sbv2', 'label_style_weight', conf['language']),
				)

				# 参照音声
				ref_audio_path = gr.Audio(
					label=get_message('sbv2', 'label_ref_audio_path', conf['language']), 
					type="filepath", 
					#value=get_cache("sbv2", "ref_audio_path"), 
					visible=False
				)

				# 音声合成（モデルをロードしてください）
				tts_button = gr.Button(
					get_message('sbv2', 'button_tts', conf['language']),
					variant="primary",
					interactive=False,
				)

				# 情報
				text_output = gr.Textbox(label=get_message('sbv2', 'label_message', conf['language']))

				# 結果
				audio_output = gr.Audio(label=get_message('sbv2', 'label_voice', conf['language']))

				with gr.Column(visible=False, elem_classes="no-border") as audio_output_to_kkeve:

					# ドロップダウン取得
					character_options, category_options = get_dropdown_options()

					character_dropdown = gr.Dropdown(
						label=get_message('sbv2', 'label_character', conf['language']), 
						choices=character_options, 
						value=get_cache("sbv2", "character_dropdown"), 
						allow_custom_value=True,
						interactive=True, 
					)
					category_dropdown  = gr.Dropdown(
						label=get_message('sbv2', 'label_category', conf['language']), 
						choices=category_options, 
						value=get_cache("sbv2", "category_dropdown"), 
						allow_custom_value=True,
						interactive=True, 
					)
					words_input = gr.Textbox(
						label=get_message('sbv2', 'label_lines', conf['language']), 
						value=get_cache("sbv2", "words_input"), 
						interactive=True 
					)
					path_input  = gr.Textbox(
						label=get_message('sbv2', 'label_path', conf['language']), 
						value=get_cache("sbv2", "path_input"), 
						interactive=True 
					)

					# 音声形式
					audio_type = gr.Radio(
						[[get_message('sbv2', 'radio_ogg', conf['language']), "ogg"], [get_message('sbv2', 'radio_wav', conf['language']), "wav"]],
						label=get_message('sbv2', 'label_audio_type', conf['language']),
						value=get_cache("sbv2", "audio_type"), 
					)

					file_input  = gr.Textbox(label=get_message('sbv2', 'label_filename', conf['language']), value=get_cache("sbv2", "file_input"))

					# 外部音声へ送る
					send_button = gr.Button(get_message('sbv2', 'button_send', conf['language']), scale=1, visible=True)


				with gr.Accordion("テキスト例", open=False):
					gr.Examples(examples, inputs=[text_input, language])

		# ----------------------------------------------------------------------
		# イベントハンドラ

		outputs=[
			model_name, 
			model_path, 
			refresh_button, 
			load_button, 
			text_input, 
			pitch_scale, 
			intonation_scale, 
			line_split, 
			split_interval, 
			tone, 
			use_tone, 
			language, 
			speaker, 
			accordion_advanced_settings, 
			sdp_ratio, 
			noise_scale, 
			noise_scale_w, 
			length_scale, 
			use_assist_text, 
			assist_text, 
			assist_text_weight, 
			style_mode, 
			style, 
			style_weight, 
			ref_audio_path, 
			tts_button, 
			text_output 
		]

		language_state.change(
			fn=change_language,
			inputs=[language_state], 
			outputs=outputs
		)

		tts_button.click(
			tts_fn, 
			inputs=[
				model_name, 
				model_path, 
				text_input, 
				language, 
				ref_audio_path, 
				sdp_ratio, 
				noise_scale, 
				noise_scale_w, 
				length_scale, 
				line_split, 
				split_interval, 
				assist_text, 
				assist_text_weight, 
				use_assist_text, 
				style, 
				style_weight, 
				tone, 
				use_tone, 
				speaker, 
				pitch_scale, 
				intonation_scale, 
				audio_type, 
			],
			outputs=[
				text_output, 
				audio_output, 
				tone, 
				audio_output_to_kkeve, 
				character_dropdown, 
				words_input, 
				path_input, 
				file_input
			], 
		)

		model_name.change(
			#model_holder.update_model_files_for_gradio, 
			fn=update_model_name, 
			inputs=[model_name], 
			outputs=[model_path], 
		)

		model_path.change(
			#make_non_interactive, 
			fn=update_model_path, 
			inputs=[model_path], 
			outputs=[tts_button]
		)

		refresh_button.click(
			model_holder.update_model_names_for_gradio, 
			outputs=[model_name, model_path, tts_button], 
		)

		load_button.click(
			#model_holder.get_model_for_gradio, 
			fn=update_load_button, 
			inputs=[model_name, model_path], 
			outputs=[style, tts_button, speaker], 
		)

		style_mode.change(
			gr_util, 
			inputs=[style_mode], 
			outputs=[style, ref_audio_path], 
		)

		text_input.change(fn=update_text, inputs=text_input)
		pitch_scale.change(fn=update_pitch_scale, inputs=pitch_scale)
		intonation_scale.change(fn=update_intonation_scale, inputs=intonation_scale)

		line_split.change(
			#lambda x: (gr.Slider(visible=x)),
			fn=update_line_split, 
			inputs=[line_split],
			outputs=[split_interval],
		)

		split_interval.change(fn=update_split_interval, inputs=split_interval)
		tone.change(fn=update_tone, inputs=tone)

		use_tone.change(
			# use_tone がON（True）になったらline_splitを強制的にOFF（False）にする
			#lambda x: (gr.Checkbox(value=False) if x else gr.Checkbox()),
			fn=update_use_tone, 
			inputs=[use_tone],
			outputs=[line_split],
		)

		language.change(fn=update_language, inputs=language)
		speaker.change(fn=update_speaker, inputs=speaker)


		sdp_ratio.change(fn=update_sdp_ratio, inputs=sdp_ratio)
		noise_scale.change(fn=update_noise_scale, inputs=noise_scale)
		noise_scale_w.change(fn=update_noise_scale_w, inputs=noise_scale_w)
		length_scale.change(fn=update_length_scale, inputs=length_scale)

		#use_assist_text.change(fn=update_use_assist_text, inputs=use_assist_text)
		use_assist_text.change(
			lambda x: (gr.Textbox(visible=x), gr.Slider(visible=x)),
			inputs=[use_assist_text],
			outputs=[assist_text, assist_text_weight],
		)

		assist_text.change(fn=update_assist_text, inputs=assist_text)
		assist_text_weight.change(fn=update_assist_text_weight, inputs=assist_text_weight)
		style_mode.change(fn=update_style_mode, inputs=style_mode)
		style.change(fn=update_style, inputs=style)
		style_weight.change(fn=update_style_weight, inputs=style_weight)
		ref_audio_path.change(fn=update_ref_audio_path, inputs=ref_audio_path)

		character_dropdown.change(fn=update_character_dropdown, inputs=[model_name, character_dropdown, audio_type], outputs=file_input)

		audio_type.change(fn=update_audio_type, inputs=[model_name, character_dropdown, audio_type], outputs=file_input)


		app.load(
			initialize_model,
			inputs=[model_name, model_path],
			outputs=[style, tts_button, speaker],
		)



	return app

# ============================================================================ #
#                                [ メイン関数 ]                                #
# ============================================================================ #
def main(config):
	global conf
	conf = config

if __name__ == "__main__":

	# コンフィグファイルを読み込む
	with open('config.json', 'r', encoding='utf-8') as f:
		config = json.load(f)

	main(config)
