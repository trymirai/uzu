#include <llama-cpp.h>

#include <CLI/CLI.hpp>
#include <glaze/glaze.hpp>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "runner.hpp"

BenchRequest get_request_from_json(const std::string& file_path) {
    BenchRequest request;
    std::string json_text;
    if (const auto error = glz::read_file_json(request, file_path, json_text)) {
        throw std::runtime_error("Failed to read " + file_path + ": " + glz::format_error(error, json_text));
    }
    return request;
}

void validate_request(const BenchRequest& request) {
    if (request.prompt_text.has_value()) {
        if (request.prompt_text.value().empty()) {
            throw std::runtime_error("Input prompt is empty");
        }
    } else if (request.prompt_chat.has_value()) {
        if (request.prompt_chat.value().empty()) {
            throw std::runtime_error("Input chat is empty");
        }
    } else {
        throw std::runtime_error("prompt_text and prompt_chat are absent");
    }
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
    auto request = BenchRequest{};
    try {
        request = get_request_from_json(input_json_path);
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
    std::vector<BenchResponse> responses;
    try {
        responses = run(model, request);
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        ret = 1;
    }
    llama_backend_free();

    // handle response
    if (ret == 0) {
        constexpr auto glz_opts = glz::opts{.prettify = true};
        if (output_json_path.has_value()) {
            std::string json_text;
            const auto error = glz::write_file_json<glz_opts>(responses, output_json_path.value(), json_text);
            if (error) {
                std::cerr << "Failed to write " << output_json_path.value() << ": " << glz::format_error(error) << '\n';
                return 1;
            }
        } else {
            std::string json_text;
            const auto error = glz::write<glz_opts>(responses, json_text);
            if (error) {
                std::cerr << "Failed to serialize output: " << glz::format_error(error) << '\n';
                return 1;
            }
            std::cout << json_text << '\n';
        }
    }

    return ret;
}
