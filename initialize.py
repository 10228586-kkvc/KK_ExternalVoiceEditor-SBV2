# ┌──────────────────────────────────────
# │  initialize.py (2025.12.25)
# └──────────────────────────────────────
# ==============================================================================
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

# ------------------------------------------------------------------------------
# bertモデルダウンロード
def download_bert_models():
	with open("bert/bert_models.json", encoding="utf-8") as fp:
		models = json.load(fp)
	for k, v in models.items():
		local_path = Path("bert").joinpath(k)
		for file in v["files"]:
			if not Path(local_path).joinpath(file).exists():
				hf_hub_download(v["repo_id"], file, local_dir=local_path)

# ------------------------------------------------------------------------------
# slmモデルダウンロード
def download_slm_model():
	local_path = Path("slm/wavlm-base-plus/")
	file = "pytorch_model.bin"
	if not Path(local_path).joinpath(file).exists():
		hf_hub_download("microsoft/wavlm-base-plus", file, local_dir=local_path)

# ============================================================================ #
#                                [ メイン関数 ]                                #
# ============================================================================ #
def main():

	# bertモデルダウンロード
	download_bert_models()

	# slmモデルダウンロード
	download_slm_model()

if __name__ == "__main__":
	main()
