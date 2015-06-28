#ifndef _CONFIG_H
#define _CONFIG_H

#include<string>
#include<map>
#include"BigInteger.h"

typedef std::pair<std::string, std::string> ConfigPair;
typedef std::map<std::string, std::string> ConfigMap;

ConfigMap readConfigFile(std::string filename, const char **fields);

class Config {

private:
    ConfigMap _configMap;

public:
    Config(std::string path);


    int readInt(std::string name);
    std::string readString(std::string name);
    BigInteger readBigInt(std::string name);
};

#endif
