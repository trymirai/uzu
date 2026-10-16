#include "util.hpp"

#include <arg.h>
#include <common.h>
#include <ggml-cpp.h>

#include <stdexcept>
#include <string_view>
#include <vector>

std::filesystem::path get_model_path(
    const std::string& model,
    bool dflash
) {
    if (std::filesystem::is_regular_file(model)) {
        return model;
    }

    common_params params;
    auto& model_params = dflash ? params.speculative.draft.mparams : params.model;
    model_params.hf_repo = model;
    if (dflash) {
        // DFlash sidecars are excluded from the upstream primary-model search.
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    }

    auto handler = common_models_handler_init(params, LLAMA_EXAMPLE_COMMON);
    common_models_handler_apply(handler, params);
    if (model_params.path.empty()) {
        throw std::runtime_error("No model found in " + model);
    }

    return model_params.path;
}
