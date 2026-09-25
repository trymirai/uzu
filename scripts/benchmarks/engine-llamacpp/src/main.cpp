#include <llama-cpp.h>

#include <CLI/CLI.hpp>
#include <common.hpp>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "engine.hpp"

int main(
    int argc,
    char* argv[]
) {
    std::string model;

    auto opts = CLI::App("benchmark_llamacpp");
    opts.add_option("-m,--model", model, "Model local path or HuggingFace id")->required();
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
        const LlamaEngine engine(model);
        run_loop(engine, output.get());
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        ret = 1;
    }
    llama_backend_free();

    return ret;
}
