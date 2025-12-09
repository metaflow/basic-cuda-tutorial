#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>

using json = nlohmann::json;

struct Config {
    int vector_size = 500000;
    bool validate = true;
    bool profile = false;
    int threads_per_block = 256;

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
            !j.contains("profile") || !j.contains("threads_per_block")) {
            fprintf(stderr, "Error: Config file must contain all fields: vector_size, validate, profile, threads_per_block\n");
            exit(EXIT_FAILURE);
        }

        cfg.vector_size = j["vector_size"];
        cfg.validate = j["validate"];
        cfg.profile = j["profile"];
        cfg.threads_per_block = j["threads_per_block"];

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
        return true;
    }

    // Convert config to JSON for output
    json toJson() const {
        return json{
            {"vector_size", vector_size},
            {"validate", validate},
            {"profile", profile},
            {"threads_per_block", threads_per_block}
        };
    }
};
