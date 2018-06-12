#pragma once

#include <string>
#include <vector>

struct ArgsParser final {
    ArgsParser(int &argc, char **argv);
    std::string get(const std::string &option) const;
    bool has(const std::string &option) const;

private:
    std::vector<std::string> tokens;
};
