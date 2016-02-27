#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <curl/curl.h>
#include "json/json.h"
#include "ServerConnection.h"

static std::string to_string(int x)
{
    char str[10] = {0};

    sprintf(str, "%d", x);

    return std::string(str);
}

static std::string encodePoints(std::vector<DistinguishedPoint> points)
{
    Json::Value root;
    unsigned int count = points.size();
    for(unsigned int i = 0; i < count; i++) {
        root[i]["a"] = points[i].a.toString(10);
        root[i]["b"] = points[i].b.toString(10);
        root[i]["x"] = points[i].x.toString(10);
        root[i]["y"] = points[i].y.toString(10);
        root[i]["length"] = points[i].length;
    }

    Json::StyledWriter writer;

    return writer.write(root);
}

static int decodeStatusMsg(std::string encoded)
{
    Json::Value root;
    Json::Reader reader;

    if(!reader.parse(encoded, root)) {
        std::cout << "JSON parsing error: " << reader.getFormattedErrorMessages();
        return -1; 
    }

    std::string statusString = root.get("status", "").asString();

    if(statusString == "running") {
        return SERVER_STATUS_RUNNING;
    }

    return SERVER_STATUS_STOPPED;
}

static BigInteger readBigInt(Json::Value &root, std::string field)
{
    std::string s = root.get(field, "").asString();
    if(s == "") {
        throw std::string("Parsing error: " + field + " is not an integer");
    }
    return BigInteger(s);
}

static ParamsMsg decodeParametersMsg(std::string encoded)
{
    ParamsMsg paramsMsg;

    Json::Value root;
    Json::Reader reader;

    if(!reader.parse(encoded, root)) {
        std::string err = "JSON parsing error: " + reader.getFormattedErrorMessages();
        throw err;
    }

    // Decode problem parameters
    Json::Value params = root["params"];

    paramsMsg.a = readBigInt(params, "a");
    paramsMsg.b = readBigInt(params, "b");
    paramsMsg.p = readBigInt(params, "p");
    paramsMsg.n = readBigInt(params, "n");
    paramsMsg.gx = readBigInt(params, "gx");
    paramsMsg.gy = readBigInt(params, "gy");
    paramsMsg.qx = readBigInt(params, "qx");
    paramsMsg.qy = readBigInt(params, "qy");
    paramsMsg.dBits = params.get("bits", -1).asInt();

    // Decode R points
    Json::Value points = root["points"];
    for(int i = 0; i < 32; i++) {
        Json::Value point = points[i];
        paramsMsg.rx[i] = readBigInt(point, "x");
        paramsMsg.ry[i] = readBigInt(point, "y");
    }

    return paramsMsg;
}


static size_t curlCallback(void *data, size_t size, size_t count, void *destPtr)
{
    if(data == NULL || count == 0) {
        return 0;
    }

    std::string *dest = (std::string *)destPtr;
    std::string dataString((char *)data, size * count);

    (*dest) += dataString;

    return size * count;
}


ServerConnection::ServerConnection(std::string host, unsigned short port)
{
    if(port > 65535) {
        throw std::string("Invalid port number");
    }

    _host = host;
    _port = port;

    // Convert port number to string
    char buf[8] = {0};
    sprintf(buf, "%d", port);

    _url = host + ":" + std::string(buf);
}


ParamsMsg ServerConnection::getParameters(std::string id)
{
    CURL *curl;

    std::string url = _url + "/params/" + id;
    std::string result;
    long httpCode = 0; 
    curl = curl_easy_init();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

    CURLcode res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        std::string errorMsg(curl_easy_strerror(res));
        printf("curl error: %s\n", errorMsg.c_str());
        curl_easy_cleanup(curl);
        throw errorMsg;
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    if(httpCode != 200) {
        curl_easy_cleanup(curl);
        throw std::string("HTTP error " + to_string(httpCode));
    }

    curl_easy_cleanup(curl);

    return decodeParametersMsg(result);
}


void ServerConnection::submitPoints(std::string id, std::vector<DistinguishedPoint> &points)
{
    std::string encodedPoints = encodePoints(points);

    CURL *curl;

    std::string url = _url + "/submit/" + id;
    std::string result;
    long httpCode = 0;

    curl = curl_easy_init();

    struct curl_slist *headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, NULL);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, encodedPoints.c_str());

    CURLcode res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        std::string errorMsg(curl_easy_strerror(res));        
        curl_easy_cleanup(curl);
        throw errorMsg;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    if(httpCode != 200) {
        curl_easy_cleanup(curl);
        throw std::string("HTTP error " + to_string(httpCode));
    }
    curl_easy_cleanup(curl);
}

int ServerConnection::getStatus(std::string id)
{
    CURL *curl = NULL;
    CURLcode res;
    long httpCode = 0;

    std::string url = _url + "/status/" + id;
    std::string result = "";

    curl = curl_easy_init();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, NULL);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);
    res = curl_easy_perform(curl);

    if(res != CURLE_OK) {
        std::string errorMsg(curl_easy_strerror(res));
        curl_easy_cleanup(curl);
        throw errorMsg;
    }

    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);

    if(httpCode == 404) {
        curl_easy_cleanup(curl);
        throw std::string("id does not exist");
    } else if(httpCode != 200) {
        std::string errorMsg("HTTP error " + to_string(httpCode));
        curl_easy_cleanup(curl);
        throw errorMsg;
    }

    curl_easy_cleanup(curl);
    
    return decodeStatusMsg(result);
}
