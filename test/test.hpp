#ifndef TEST_H
#define TEST_H

#define TEST_VERBOSE false

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <tuple>

#include <assert.h>

#include "../include/json/json.hpp"

int expect(bool assertion, std::string message) {
    if (!assertion) {
        std::cout << "\033[1;31m" << message << "\033[0m" << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect(int expectation, int reality, std::string message) {
    if (expectation != reality) {
        std::cout << "\033[1;31m" << message << " Expectation: " << expectation << " Reality: " << reality << "\033[0m" << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect(float expectation, float reality, std::string message) {
    float const epsilon = std::numeric_limits<float>::epsilon();    
    if (abs(expectation - reality) > epsilon) {
        std::cout << "\033[1;31m" << message << " Expectation: " << expectation << " Reality: " << reality << "\033[0m" << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect(std::string expectation, std::string reality, std::string message) {
    if (expectation != reality) {
        std::cout << "\033[1;31m" << message << " Expectation: " << expectation << " Reality: " << reality << "\033[0m" << std::endl;
        return 1;
    } else {
        return 0;
    }
}

template <typename T>
int expect(T const & expectation, T const & reality, std::string message) {
    if (expectation != reality) {
        std::cout << "\033[1;31m" << message << "\033[0m" << std::endl;
        return 1;
    } else {
        return 0;
    }
}

void pass(std::string message) {
    std::cout << "\033[1;32m" << message << "\033[0m" << std::endl;
}

void fail(std::string message) {
    std::cout << "\033[1;31m" << message << "\033[0m" << std::endl;
}

int run_tests(std::string unit_name, int (*tests)(void)) {
    int failures = tests();
    if (failures == 0) {
        std::cout << "\033[1;32m" << unit_name << " Tests Passed" << "\033[0m" << std::endl;
    } else {
        std::cout << "\033[1;31m" << failures << " " << unit_name << " Tests Failed" << "\033[0m" << std::endl;
    }
    return failures;
}


#endif
