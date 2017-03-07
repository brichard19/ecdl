#include <fstream>
#include "client.h"
#include "json/json.h"

static std::string readFile(std::string fileName)
{
    std::ifstream file(fileName.c_str());
    std::string s;
    std::string buf;

    while(std::getline(file, buf)) {
        s += buf;
    }

    return s;
}

ClientConfig loadConfig(std::string fileName)
{
    Json::Value root;
    Json::Reader reader;
    ClientConfig configObj;

    std::string content = readFile(fileName);

    if(!reader.parse(content, root)) {
        std::string err = "Error parsing config file: " + reader.getFormattedErrorMessages();
        throw err;
    }

    configObj.serverHost = root.get("server_host", "").asString();
    configObj.serverPort = root.get("server_port", -1).asInt();
    configObj.pointCacheSize = root.get("point_cache_size", 4).asInt();

#ifdef _CUDA
    configObj.threads = root.get("cuda_threads", 32).asInt();
    configObj.blocks = root.get("cuda_blocks", 1).asInt();
    configObj.pointsPerThread = root.get("cuda_points_per_thread", 1).asInt();
    configObj.device = root.get("cuda_device", -1).asInt();
    configObj.pointCacheSize = root.get("point_cache_size", 4).asInt();
#else
    configObj.threads = root.get("cpu_threads", -1).asInt();
    configObj.pointsPerThread = root.get("cpu_points_per_thread", -1).asInt();
#endif

    return configObj;
}
