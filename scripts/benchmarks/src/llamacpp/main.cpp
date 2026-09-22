#include <llama-cpp.h>

#include <iostream>

#include "common.hpp"
#include "content.hpp"
#include "runner.hpp"

int main(int argc, char* argv[]) {
    const auto request = RunRequest{
        "unsloth/Qwen3.6-27B-GGUF:Q4_K_S",
        // "unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_S",
        std::string{"Tell me about London"},
        256,
    };

    ggml_log_callback log_callback = [](enum ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    };
    llama_log_set(log_callback, nullptr);

    int ret = 0;
    llama_backend_init();
    RunResponse response;
    try {
        response = run(request);
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        ret = 1;
    }
    llama_backend_free();

    if (ret == 0) {
        // std::cout << response.text << std::endl;
        std::cout << "time_to_first_token: " << response.time_to_first_token << '\n'
                  << "prompt_tps: " << response.prompt_tps << '\n'
                  << "generation_tps: " << response.generation_tps << '\n'
                  << "tokens_per_fp: " << response.tokens_per_fp << '\n'
                  << "duration: " << response.duration << '\n'
                  << "phys_footprint: " << response.memory_counters.phys_footprint << '\n'
                  << "resident_size: " << response.memory_counters.resident_size << '\n'
                  << "graphics_total: " << response.memory_counters.graphics_total << '\n';
    }

    return ret;
}
