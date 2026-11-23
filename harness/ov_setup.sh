#!/usr/bin/env bash
# Install OpenVINO GenAI venv + download Qwen3.5-2B-int4-ov IR. Run via harness/locked.sh.
set -euo pipefail
R=/home/user/Infer-opts
python3 -m venv $R/engines/ov-venv
$R/engines/ov-venv/bin/pip install -q "openvino==2026.4.1" "openvino-tokenizers==2026.4.1.0" "openvino-genai==2026.4.1.0"
D=$R/models/ov/Qwen3.5-2B-int4-ov; mkdir -p $D
for f in README.md chat_template.jinja config.json generation_config.json openvino_config.json openvino_detokenizer.bin openvino_detokenizer.xml openvino_language_model.bin openvino_language_model.xml openvino_mtp_model.bin openvino_mtp_model.xml openvino_text_embeddings_model.bin openvino_text_embeddings_model.xml openvino_tokenizer.bin openvino_tokenizer.xml openvino_vision_embeddings_merger_model.bin openvino_vision_embeddings_merger_model.xml openvino_vision_embeddings_model.bin openvino_vision_embeddings_model.xml openvino_vision_embeddings_pos_model.bin openvino_vision_embeddings_pos_model.xml preprocessor_config.json processor_config.json tokenizer.json tokenizer_config.json video_preprocessor_config.json; do
  [ -s $D/$f ] || curl -fsSL -o $D/$f https://huggingface.co/OpenVINO/Qwen3.5-2B-int4-ov/resolve/main/$f
done
du -sh $D; $R/engines/ov-venv/bin/pip list 2>/dev/null | grep -i openvino
