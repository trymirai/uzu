#include "common.hpp"

#include <unistd.h>

#include <glaze/glaze.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

std::FILE* redirect_stdout_to_stderr() {
    std::cout.flush();
    std::fflush(stdout);

    // Keep the original stdout for responses; send C and C++ library output to stderr.
    const int output_fd = dup(STDOUT_FILENO);
    if (output_fd < 0) {
        throw std::runtime_error("Failed to duplicate stdout");
    }

    std::FILE* output = fdopen(output_fd, "w");
    if (!output) {
        close(output_fd);
        throw std::runtime_error("Failed to open response output");
    }

    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
        std::fclose(output);
        throw std::runtime_error("Failed to redirect engine output to stderr");
    }
    return output;
}

static BenchRequest get_request_from_json(const std::string& json_text) {
    BenchRequest request;
    if (const auto error = glz::read<glz::opts_validate{}>(request, json_text)) {
        throw std::runtime_error(glz::format_error(error, json_text));
    }
    return request;
}

static void validate_request(const BenchRequest& request) {
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

void run_loop(
    const InferenceEngine& engine,
    std::FILE* output
) {
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r\n\v\f") == std::string::npos) {
            continue;
        }

        std::string response_json;
        try {
            const auto request = get_request_from_json(line);
            validate_request(request);

            const auto responses = engine.execute(request);
            if (const auto error = glz::write_json(responses, response_json)) {
                throw std::runtime_error("Failed to serialize output: " + glz::format_error(error));
            }
        } catch (const std::exception& error) {
            std::cerr << "Failed to process request: " << error.what() << std::endl;
            continue;
        }

        if (std::fwrite(response_json.data(), 1, response_json.size(), output) != response_json.size() ||
            std::fputc('\n', output) == EOF || std::fflush(output) != 0) {
            throw std::runtime_error("Failed to write response to stdout");
        }
    }

    if (std::cin.bad()) {
        throw std::runtime_error("Failed to read request from stdin");
    }
}
