#ifndef _CONFIGFILE_H
#define _CONFIGFILE_H

#include <map>
#include <stdlib.h>

class ConfigField {

private:
    std::string _key; 
    std::string _value;

public:

    ConfigField()
    {

    }
    
    ConfigField(std::string key, std::string value)
    {
        _key = key;
        _value = value;
    }

    int asInt()
    {
        return atoi(_value.c_str());
    }

    std::string asString()
    {
        return _value;
    }
};

class ConfigFile {

private:

    std::map<std::string, ConfigField> _fields;


public:

    static ConfigFile parse(const std::string &path);

    bool contains (const std::string &key);
   
    ConfigField get(const std::string &key);

    ConfigField get(const std::string &key, const std::string &defaultValue);
};

#endif