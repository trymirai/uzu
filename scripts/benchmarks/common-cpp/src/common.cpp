#include "common.hpp"

#include <unistd.h>

#include <cinttypes>
#include <fstream>
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

memory_counters_t collect_memory_counters() {
    memory_counters_t memory_counters{};
#ifdef __APPLE__
    const kern_return_t result = get_memory_counters(&memory_counters, false);
    if (result != KERN_SUCCESS) {
        throw std::runtime_error(
            std::string{"Failed to collect memory counters: "} + memory_counters_error_string(result)
        );
    }
#else
    // Private anonymous memory including swap as the footprint, and the resident set, as the Python adapters report.
    std::ifstream status("/proc/self/status");
    uint64_t anonymous = 0, swap = 0, resident = 0;
    for (std::string line; std::getline(status, line);) {
        std::sscanf(line.c_str(), "RssAnon: %" SCNu64, &anonymous);
        std::sscanf(line.c_str(), "VmSwap: %" SCNu64, &swap);
        std::sscanf(line.c_str(), "VmRSS: %" SCNu64, &resident);
    }
    if (resident == 0) {
        throw std::runtime_error("Failed to read memory counters from /proc/self/status");
    }
    memory_counters.phys_footprint = (anonymous + swap) * 1024;
    memory_counters.resident_size = resident * 1024;
#endif
    return memory_counters;
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
    std::cin.clear();
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r\n\v\f") == std::string::npos) {
            continue;
        }

        std::string response_json;
        try {
            const auto request = get_request_from_json(line);
            validate_request(request);
            if (request.reasoning_effort.has_value() && !engine.supports_reasoning_effort()) {
                throw std::runtime_error("This engine does not support reasoning_effort");
            }

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
