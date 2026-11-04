#include <llama-cpp.h>

#include <CLI/CLI.hpp>
#include <common.hpp>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "engine.hpp"

struct LlamaConfig {
    std::string model;
    bool specdec;
    BenchSampling sampling;
    ReasoningEfforts reasoning_efforts;
    std::optional<std::string> draft_model;
    std::optional<size_t> speculative_depth;
};

extern "C" int benchmarks_llamacpp_main(
    int argc,
    char* argv[]
) {
    std::string model_file;

    auto opts = CLI::App("benchmark_llamacpp");
    opts.add_option("model.json", model_file, "Engine model configuration")->required();
    CLI11_PARSE(opts, argc, argv);

    const auto output = std::unique_ptr<std::FILE, decltype(&std::fclose)>(redirect_stdout_to_stderr(), std::fclose);

    // setup logs
    ggml_log_callback log_callback = [](enum ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    };
    llama_log_set(log_callback, nullptr);

    // Process requests until stdin closes, keeping the backend alive between requests.
    int ret = 0;
    llama_backend_init();
    try {
        const auto config = read_model_config<LlamaConfig>(model_file);
        check_specdec(config.specdec, config.speculative_depth.value_or(0) > 0);
        const auto engine = LlamaEngine(
            config.model,
            config.sampling,
            config.draft_model.value_or(""),
            config.speculative_depth.value_or(0)
        );
        run_loop(engine, output.get());
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        ret = 1;
    }
    llama_backend_free();

    return ret;
}

int main(
    int argc,
    char* argv[]
) {
    return benchmarks_llamacpp_main(argc, argv);
}
