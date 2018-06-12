#include "argsparser.h"

#include <algorithm>

ArgsParser::ArgsParser(int &argc, char **argv) {
	for (int i = 1; i < argc; ++i)
		tokens.emplace_back(argv[i]);
}

std::string ArgsParser::get(const std::string &option) const {
	std::vector<std::string>::const_iterator itr;
	itr = std::find(tokens.begin(), tokens.end(), option);
	if (itr != tokens.end() && ++itr != tokens.end()) {
		return *itr;
	}

	return std::string();
}

bool ArgsParser::has(const std::string &option) const {
	return std::find(tokens.begin(), tokens.end(), option) != tokens.end();
}
