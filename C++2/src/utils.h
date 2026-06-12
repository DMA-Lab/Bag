#pragma once

#include <functional>
#include <string>
#include <unordered_map>

namespace bag {

long long calc_execution_time_in_us(const std::function<void(void)>& func);
std::unordered_map<std::string, std::string> parse_cli_args(int argc, char** argv, int start_index);

}  // namespace bag
