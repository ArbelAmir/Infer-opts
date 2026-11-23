set -ex
cd /home/user/Infer-opts/engines
if [ ! -d MNN/.git ]; then
  mkdir MNN && cd MNN && git init -q && git remote add origin https://github.com/alibaba/MNN.git
  git fetch --depth 1 origin b0239771b07b317a258fbc384683bbd540741406 && git checkout -q FETCH_HEAD
  cd ..
fi
git -C MNN log -1 --oneline
cd /home/user/Infer-opts/models/mnn/Qwen3.5-2B-MNN
B=https://huggingface.co/taobao-mnn/Qwen3.5-2B-MNN/resolve/main
for f in llm.mnn llm.mnn.json llm.mnn.weight tokenizer.mtok config.json llm_config.json; do
  [ -s $f ] || curl -fsSL -o $f $B/$f
done
ls -la
