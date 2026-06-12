#include "utils.h"

#include <chrono>
#include <stdexcept>

namespace bag {

long long calc_execution_time_in_us(const std::function<void(void)>& func) {
    const auto start = std::chrono::steady_clock::now();
    func();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

std::unordered_map<std::string, std::string> parse_cli_args(int argc, char** argv, int start_index) {
    std::unordered_map<std::string, std::string> args;
    for (int i = start_index; i < argc; ++i) {
        std::string key = argv[i];
        if (!key.starts_with("--")) {
            throw std::runtime_error("expected --key style argument, got: " + key);
        }
        if (i + 1 >= argc) {
            throw std::runtime_error("missing value for argument: " + key);
        }
        args.emplace(key.substr(2), argv[++i]);
    }
    return args;
}

}  // namespace bag
