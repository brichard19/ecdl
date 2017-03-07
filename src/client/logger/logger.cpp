#include"logger.h"
#include <string>
#include <iostream>
#include <stdarg.h>
#include <stdio.h>
#include <ctime>

static std::string getDateString()
{
  time_t rawtime;
  struct tm * timeinfo;
  char buffer[128];

  time (&rawtime);
  timeinfo = localtime(&rawtime);

  strftime(buffer,128,"%d-%m-%Y %I:%M:%S",timeinfo);
  
  return std::string(buffer);

}

void Logger::logInfo(std::string s)
{
    std::string dateTime = getDateString();

    fprintf(stdout, "[%s] ", dateTime.c_str());
    fprintf(stdout, "%s\n", s.c_str());
}

void Logger::logInfo(const char *format, ...)
{
    va_list args;

    std::string dateTime = getDateString();

    fprintf(stdout, "[%s] ", dateTime.c_str());
    va_start(args, format);
    vfprintf(stdout,format, args);
    fprintf(stdout, "\n");
    va_end(args);
}

void Logger::logError(std::string s)
{
    std::string dateTime = getDateString();

    fprintf(stdout, "[%s] ", dateTime.c_str());
    fprintf(stdout, "%s\n", s.c_str());
}


void Logger::logError(const char *format, ...)
{
    std::string dateTime = getDateString();

    fprintf(stderr, "[%s] ", dateTime.c_str());

    va_list args;
    va_start(args, format);
    fprintf(stderr, "[ERROR] ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

