#include <iostream>
#include <fstream>
#include <string>
#include "Config.h"

static std::string trim(std::string s, char c)
{
    int start = s.find_first_not_of(c);
    int end = s.find_last_not_of(c);

    return s.substr(start, end-start);
}

static bool getFields(std::string s, std::string &left, std::string &right)
{
    int eq = s.find("=");

    if(eq == -1) {
        return false;
    }

    left = trim(s.substr(0, eq), ' ');
    right = trim(s.substr(eq), ' ');
}

ConfigFile ConfigFile::parse(const std::string &path)
{
    std::ifstream fs(path.c_str());

    std::string line;

    ConfigFile config;

    while(std::getline(fs, line)) {

        std::string left;
        std::string right;

        if(!getFields(line, left, right)) {
            continue;
        }

        config._fields.insert(std::pair<std::string, ConfigField>(left, ConfigField(left, right)));
    }

    return config;
}

ConfigField ConfigFile::get(const std::string &fieldName)
{
    return _fields[fieldName];
}

ConfigField ConfigFile::get(const std::string &fieldName, const std::string &defaultValue)
{
    if(_fields.find(fieldName) != _fields.end()) {
        return _fields[fieldName];
    }

    return ConfigField(fieldName, defaultValue);
}