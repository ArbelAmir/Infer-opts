// Minimal MNN-LLM driver: runs one chat workload (system.txt + user.txt) greedy, N repeats.
// usage: mnn_workload config.json system.txt user.txt max_new_tokens repeats out_prefix [extra_config_json]
#include <llm/llm.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <memory>
using namespace MNN::Transformer;
static std::string slurp(const char* p) { std::ifstream f(p); std::stringstream s; s << f.rdbuf(); return s.str(); }
int main(int argc, char** argv) {
    if (argc < 7) { std::cerr << "usage\n"; return 1; }
    std::string cfg = argv[1];
    int maxTok = atoi(argv[4]), reps = atoi(argv[5]);
    std::string prefix = argv[6];
    ChatMessages msgs = {{"system", slurp(argv[2])}, {"user", slurp(argv[3])}};
    std::unique_ptr<Llm> llm(Llm::createLLM(cfg));
    llm->set_config("{\"tmp_path\":\"/tmp/mnn_tmp_qwen35\"}");
    if (argc > 7) llm->set_config(argv[7]);
    auto t0 = std::chrono::steady_clock::now();
    if (!llm->load()) { std::cerr << "load failed\n"; return 2; }
    double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    llm->set_config("{\"sampler_type\":\"greedy\",\"async\":false,\"jinja\":{\"context\":{\"enable_thinking\":false}}}");
    auto ctx = llm->getContext();
    for (int r = 0; r <= reps; r++) {  // r==0 is warm-up
        llm->reset();
        std::ostringstream os;
        auto a = std::chrono::steady_clock::now();
        llm->response(msgs, &os, nullptr, maxTok);
        double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        double pf = ctx->prefill_us / 1e6, dc = ctx->decode_us / 1e6;
        printf("RESULT run=%d warmup=%d load_s=%.3f prompt_n=%d gen_n=%d prefill_s=%.4f decode_s=%.4f pp_tps=%.3f gen_tps=%.3f total_s=%.3f\n",
               r, r == 0, load_s, ctx->prompt_len, ctx->gen_seq_len, pf, dc, ctx->prompt_len / pf,
               dc > 0 ? (ctx->gen_seq_len - 1) / dc : 0.0, total_s);
        std::ofstream(prefix + ".run" + std::to_string(r) + ".txt") << os.str();
        fflush(stdout);
    }
    return 0;
}
