#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "common.hpp"
#include "engine.hpp"

struct MlDriftConfig {
    std::string model;
    bool specdec;
    std::optional<std::string> revision;
    BenchSampling sampling;
    ReasoningEfforts reasoning_efforts;
    std::optional<std::string> precision;
};

ABSL_FLAG(
    std::string,
    config,
    "",
    "Engine model configuration JSON"
);

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

extern "C" int benchmarks_mldrift_main(
    int argc,
    char* argv[]
) {
    absl::ParseCommandLine(argc, argv);

    const auto output = std::unique_ptr<std::FILE, decltype(&std::fclose)>(redirect_stdout_to_stderr(), std::fclose);

    try {
        const auto config = read_model_config<MlDriftConfig>(absl::GetFlag(FLAGS_config));
        check_specdec(config.specdec, false);
        if (!std::holds_alternative<GreedySampling>(config.sampling)) {
            throw std::invalid_argument("ML Drift supports greedy decoding only");
        }
        const auto engine = MlDriftEngine(absl::GetFlag(FLAGS_checkpoint), absl::GetFlag(FLAGS_weights));
        run_loop(engine, output.get());
    } catch (const std::exception& error) {
        std::cerr << "Failed: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

int main(
    int argc,
    char* argv[]
) {
    return benchmarks_mldrift_main(argc, argv);
}
