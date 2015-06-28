#ifndef _LOGGER_H
#define _LOGGER_H

#include<string>

class Logger {
public:
    static void logInfo(std::string s);
    static void logError(std::string s);
    static void logInfo(const char *format, ...);
    static void logError(const char *format, ...);
};

#endif
