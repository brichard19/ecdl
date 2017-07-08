#include <fstream>
#include "client.h"
#include "json/json.h"
#include "Config.h"

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
    ClientConfig configObj;
    ConfigFile config = ConfigFile::parse(fileName);

    configObj.serverHost = config.get("server_host", "").asString();
    configObj.serverPort = config.get("server_port", "-1").asInt();
    configObj.pointCacheSize = config.get("point_cache_size").asInt();

#ifdef _CUDA
    configObj.threads = config.get("cuda_threads", "32").asInt();
    configObj.blocks = config.get("cuda_blocks", "1").asInt();
    configObj.pointsPerThread = config.get("cuda_points_per_thread").asInt();
    configObj.device = config.get("cuda_device").asInt();
    configObj.pointCacheSize = config.get("point_cache_size", "4").asInt();
#else
    configObj.threads = config.get("cpu_threads", "-1").asInt();
    configObj.pointsPerThread = config.get("cpu_points_per_thread", "-1").asInt();
#endif

    return configObj;
}
