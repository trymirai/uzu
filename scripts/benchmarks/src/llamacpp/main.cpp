#include <llama-cpp.h>

#include <CLI/CLI.hpp>
#include <glaze/glaze.hpp>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "content.hpp"
#include "runner.hpp"

struct InputConfig {
    std::optional<std::string> prompt_text;
    std::optional<std::vector<ChatMessage>> prompt_chat;
    std::optional<size_t> max_tokens;
    std::optional<size_t> speculative_depth;
    std::optional<SamplingConfig> sampling;
};

struct OutputMetrics {
    std::string text;
    double time_to_first_token;
    double prompt_tps;
    double decode_tps;
    double tokens_per_forward_pass;
    double duration;
    uint64_t memory_phys_footprint;
    uint64_t memory_resident_peak;
    uint64_t memory_graphics_total;
};

RunRequest get_request_from_json(
    const std::string& model,
    const std::string& file_path
) {
    InputConfig config;
    std::string json_text;
    if (const auto error = glz::read_file_json(config, file_path, json_text)) {
        throw std::runtime_error("Failed to read " + file_path + ": " + glz::format_error(error, json_text));
    }

    Content input;
    if (config.prompt_text.has_value()) {
        input = config.prompt_text.value();
    } else if (config.prompt_chat.has_value()) {
        input = config.prompt_chat.value();
    } else {
        throw std::runtime_error("prompt_text and prompt_chat are abscent");
    }

    return RunRequest{
        .model = model,
        .input = input,
        .max_tokens = config.max_tokens,
        .speculative_depth = config.speculative_depth,
        .sampling = config.sampling,
    };
}

void validate_request(const RunRequest& request) {
    if (request.model.empty()) {
        throw std::runtime_error("Model is empty");
    }

    if (std::holds_alternative<std::string>(request.input)) {
        const auto text = std::get<std::string>(request.input);
        if (text.empty()) {
            throw std::runtime_error("Input prompt is empty");
        }
    } else if (std::holds_alternative<std::vector<ChatMessage>>(request.input)) {
        const auto messages = std::get<std::vector<ChatMessage>>(request.input);
        if (messages.empty()) {
            throw std::runtime_error("Input chat is empty");
        }
    }
}

OutputMetrics get_output_metrics(const RunResponse& response) {
    return OutputMetrics{
        .text = response.text,
        .time_to_first_token = response.time_to_first_token,
        .prompt_tps = response.prompt_tps,
        .decode_tps = response.generation_tps,
        .tokens_per_forward_pass = response.tokens_per_fp,
        .duration = response.duration,
        .memory_phys_footprint = response.memory_counters.phys_footprint,
        .memory_resident_peak = response.memory_counters.resident_size_peak,
        .memory_graphics_total = response.memory_counters.graphics_total,
    };
}

int main(
    int argc,
    char* argv[]
) {
    std::string model;
    std::string input_json_path;
    std::optional<std::string> output_json_path;

    auto opts = CLI::App("benchmark_llamacpp");
    opts.add_option("-m,--model", model, "Model local path or HuggingFace id")->required();
    opts.add_option("-i,--input", input_json_path, "Path to json file with input configuration")->required();
    opts.add_option("-o,--output", output_json_path, "Path to json file with results");
    CLI11_PARSE(opts, argc, argv);

    // prepare request
    auto request = RunRequest{};
    try {
        request = get_request_from_json(model, input_json_path);
        validate_request(request);
    } catch (const std::exception& exc) {
        std::cerr << exc.what() << std::endl;
        return 1;
    }

    // setup logs
    ggml_log_callback log_callback = [](enum ggml_log_level level, const char* text, void*) {
        if (level == GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    };
    llama_log_set(log_callback, nullptr);

    // execute request
    llama_backend_init();
    int ret = 0;
    RunResponse response;
    try {
        response = run(request);
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        ret = 1;
    }
    llama_backend_free();

    // handle response
    if (ret == 0) {
        auto output = get_output_metrics(response);
        constexpr auto glz_opts = glz::opts{.prettify = true};
        if (output_json_path.has_value()) {
            std::string json_text;
            const auto error = glz::write_file_json<glz_opts>(output, output_json_path.value(), json_text);
            if (error) {
                std::cerr << "Failed to write " << output_json_path.value() << ": " << glz::format_error(error) << '\n';
                return 1;
            }
        } else {
            std::string json_text;
            const auto error = glz::write<glz_opts>(output, json_text);
            if (error) {
                std::cerr << "Failed to serialize output: " << glz::format_error(error) << '\n';
                return 1;
            }
            std::cout << json_text << '\n';
        }
    }

    return ret;
}
