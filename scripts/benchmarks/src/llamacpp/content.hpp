#ifndef __llama_cpp_benchmarks_content_hpp__
#define __llama_cpp_benchmarks_content_hpp__

#include <string>
#include <variant>
#include <vector>

struct ChatMessage {
    std::string message;
    std::string role;
};

using Content = std::variant<std::string, std::vector<ChatMessage>>;

#endif  // __llama_cpp_benchmarks_content_hpp__