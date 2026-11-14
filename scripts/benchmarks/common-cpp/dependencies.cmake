include(FetchContent)

function(benchmark_dependency name)
    file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../sources.json" sources)
    string(JSON owner GET "${sources}" "${name}" owner)
    string(JSON repo GET "${sources}" "${name}" repo)
    string(JSON revision GET "${sources}" "${name}" rev)
    FetchContent_Declare(${name}
        GIT_REPOSITORY "https://github.com/${owner}/${repo}.git"
        GIT_TAG "${revision}"
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(${name})
endfunction()
