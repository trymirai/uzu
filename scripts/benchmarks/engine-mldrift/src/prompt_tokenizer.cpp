#include "prompt_tokenizer.hpp"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <glaze/glaze.hpp>
#include <memory>
#include <stdexcept>
#include <string>

extern char** environ;

PromptTokenizer::PromptTokenizer(
    const std::filesystem::path& project,
    const std::filesystem::path& checkpoint
) {
    int request_pipe[2];
    int response_pipe[2];
    if (pipe(request_pipe) != 0) {
        throw std::runtime_error("Failed to create the prompt tokenizer request pipe");
    }
    if (pipe(response_pipe) != 0) {
        close(request_pipe[0]);
        close(request_pipe[1]);
        throw std::runtime_error("Failed to create the prompt tokenizer response pipe");
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, request_pipe[0], STDIN_FILENO);
    posix_spawn_file_actions_adddup2(&actions, response_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_addinherit_np(&actions, STDERR_FILENO);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);

    std::string arguments[] = {"uv", "run", "--project", project, "--quiet", "mldrift-prompt-tokens", checkpoint};
    char* argv[] = {
        arguments[0].data(),
        arguments[1].data(),
        arguments[2].data(),
        arguments[3].data(),
        arguments[4].data(),
        arguments[5].data(),
        arguments[6].data(),
        nullptr
    };
    const int result = posix_spawnp(&this->pid, argv[0], &actions, &attributes, argv, environ);
    posix_spawnattr_destroy(&attributes);
    posix_spawn_file_actions_destroy(&actions);
    close(request_pipe[0]);
    close(response_pipe[1]);
    if (result != 0) {
        close(request_pipe[1]);
        close(response_pipe[0]);
        throw std::runtime_error("Failed to start the prompt tokenizer: " + std::string{std::strerror(result)});
    }

    this->input = fdopen(request_pipe[1], "w");
    this->output = fdopen(response_pipe[0], "r");
    if (!this->input || !this->output) {
        throw std::runtime_error("Failed to open the prompt tokenizer pipes");
    }
}

PromptTokenizer::~PromptTokenizer() {
    std::fclose(this->input);
    std::fclose(this->output);
    waitpid(this->pid, nullptr, 0);
}

std::vector<int> PromptTokenizer::tokenize(const BenchRequest& request) const {
    const auto request_json = glz::write_json(request);
    if (!request_json) {
        throw std::runtime_error("Failed to serialize request: " + glz::format_error(request_json.error()));
    }
    if (std::fputs(request_json.value().c_str(), this->input) == EOF || std::fputc('\n', this->input) == EOF ||
        std::fflush(this->input) != 0) {
        throw std::runtime_error("Failed to send the prompt to the tokenizer");
    }

    char* line = nullptr;
    size_t capacity = 0;
    const ssize_t length = getline(&line, &capacity, this->output);
    const auto line_guard = std::unique_ptr<char, decltype(&std::free)>(line, std::free);
    if (length <= 0) {
        throw std::runtime_error("Prompt tokenizer exited");
    }

    std::vector<int> tokens;
    if (const auto error = glz::read_json(tokens, std::string_view{line, static_cast<size_t>(length)})) {
        throw std::runtime_error("Invalid prompt tokenizer response: " + glz::format_error(error));
    }
    if (tokens.empty()) {
        throw std::runtime_error("Failed to tokenize the prompt");
    }
    return tokens;
}
