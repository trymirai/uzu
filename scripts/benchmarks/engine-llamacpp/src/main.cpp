#include <llama-cpp.h>

#include <CLI/CLI.hpp>
#include <common.hpp>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "engine.hpp"

extern "C" int benchmarks_llamacpp_main(
    int argc,
    char* argv[]
) {
    std::string model;
    std::string draft_model;
    std::string kv_cache_type = "f16";

    auto opts = CLI::App("benchmark_llamacpp");
    opts.add_option("-m,--model", model, "Model local path or HuggingFace id")->required();
    opts.add_option("-d,--draft-model", draft_model, "DFlash draft model local path or HuggingFace id");
    opts.add_option("--kv-cache-type", kv_cache_type, "K and V cache element type: a ggml type name, such as q8_0");
    CLI11_PARSE(opts, argc, argv);
    ggml_type kv_type = GGML_TYPE_COUNT;
    for (int type = 0; type < GGML_TYPE_COUNT; type++) {
        if (kv_cache_type == ggml_type_name(static_cast<ggml_type>(type))) {
            kv_type = static_cast<ggml_type>(type);
        }
    }
    if (kv_type == GGML_TYPE_COUNT) {
        std::cerr << "Unknown --kv-cache-type " << kv_cache_type << std::endl;
        return 1;
    }

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
        const auto engine = LlamaEngine(model, draft_model, kv_type);
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
