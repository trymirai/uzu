#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common.hpp"
#include "engine.hpp"

ABSL_FLAG(
    std::string,
    checkpoint,
    "",
    "Hugging Face checkpoint directory with config.json and the tokenizer"
);
ABSL_FLAG(
    std::string,
    weights,
    "",
    "Directory with weights extracted by extract_weights_hf.py"
);
ABSL_FLAG(
    std::string,
    prompt_project,
    "",
    "uv project that provides mldrift-prompt-tokens"
);

int main(
    int argc,
    char* argv[]
) {
    absl::ParseCommandLine(argc, argv);

    const auto output = std::unique_ptr<std::FILE, decltype(&std::fclose)>(redirect_stdout_to_stderr(), std::fclose);

    try {
        const auto engine = MlDriftEngine(
            absl::GetFlag(FLAGS_checkpoint),
            absl::GetFlag(FLAGS_weights),
            absl::GetFlag(FLAGS_prompt_project)
        );
        run_loop(engine, output.get());
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
