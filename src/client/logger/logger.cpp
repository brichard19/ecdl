#include"logger.h"
#include<string>
#include<iostream>
#include<stdarg.h>
#include<stdio.h>

void Logger::logInfo(std::string s)
{
    fprintf(stdout, "%s", s.c_str());
}

void Logger::logError(std::string s)
{
    fprintf(stdout, "%s", s.c_str());
}

void Logger::logInfo(const char *format, ...)
{
    va_list args;

    va_start(args, format);
    //fprintf(stdout, "[INFO] ");
    vfprintf(stdout,format, args);
    fprintf(stdout, "\n");
    va_end(args);
}

void Logger::logError(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

