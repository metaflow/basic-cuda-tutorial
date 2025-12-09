#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>

using json = nlohmann::json;

struct Config {
    long long vector_size = 500000;
    bool validate = true;
    bool profile_gpu = false;
    bool profile_cpu = false;
    int threads_per_block = 256;
    int timeout_s = 10;
    json experiment = nullptr;  // Arbitrary JSON passed through

    // Load configuration from JSON file
    static Config loadFromFile(const char* filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            fprintf(stderr, "Error: Cannot open config file '%s'\n", filepath);
            exit(EXIT_FAILURE);
        }

        json j;
        try {
            file >> j;
        } catch (const json::parse_error& e) {
            fprintf(stderr, "Error: Failed to parse JSON file '%s': %s\n", filepath, e.what());
            exit(EXIT_FAILURE);
        }

        Config cfg;

        // Require all fields to be present
        if (!j.contains("vector_size") || !j.contains("validate") ||
            !j.contains("profile_gpu") || !j.contains("profile_cpu") ||
            !j.contains("threads_per_block") || !j.contains("timeout_s") ||
            !j.contains("experiment")) {
            fprintf(stderr, "Error: Config file must contain all fields: vector_size, validate, profile_gpu, profile_cpu, threads_per_block, timeout_s, experiment\n");
            exit(EXIT_FAILURE);
        }

        cfg.vector_size = j["vector_size"];
        cfg.validate = j["validate"];
        cfg.profile_gpu = j["profile_gpu"];
        cfg.profile_cpu = j["profile_cpu"];
        cfg.threads_per_block = j["threads_per_block"];
        cfg.timeout_s = j["timeout_s"];
        cfg.experiment = j["experiment"];

        return cfg;
    }

    // Validate configuration values
    bool isValid() const {
        if (vector_size <= 0) {
            fprintf(stderr, "Error: vector_size must be positive\n");
            return false;
        }
        if (threads_per_block <= 0 || threads_per_block > 1024) {
            fprintf(stderr, "Error: threads_per_block must be between 1 and 1024\n");
            return false;
        }
        if (timeout_s <= 0) {
            fprintf(stderr, "Error: timeout_s must be positive\n");
            return false;
        }
        return true;
    }

    // Convert config to JSON for output
    json toJson() const {
        return json{
            {"vector_size", vector_size},
            {"validate", validate},
            {"profile_gpu", profile_gpu},
            {"profile_cpu", profile_cpu},
            {"threads_per_block", threads_per_block},
            {"timeout_s", timeout_s},
            {"experiment", experiment}
        };
    }
};
