#!/usr/bin/env bash
# Build MNN-LLM (run through harness/locked.sh). Pinned commit b0239771b07b317a258fbc384683bbd540741406.
set -ex
cd /home/user/Infer-opts/engines/MNN
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMNN_BUILD_LLM=ON -DMNN_LOW_MEMORY=ON \
  -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_AVX512=ON -DMNN_BUILD_LLM_OMNI=ON
time cmake --build build --target llm_demo llm_bench -j4
