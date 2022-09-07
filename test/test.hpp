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
#include <iomanip>
//#include <alloca.h>

#include <assert.h>

#include "../include/json/json.hpp"

template<typename T>
std::string error_message(T expectation, T reality, std::string message, std::string context = "") {
    std::stringstream error_message;
    error_message << "\033[1;31m";
    if (context != "") {
        error_message << context << "\n    ";
    }
    error_message << message << " Expectation: " << expectation << " Reality: " << reality;
    error_message << "\033[0m";
    return error_message.str();
}

std::string error_message(std::string message, std::string context = "") {
    std::stringstream error_message;
    error_message << "\033[1;31m";
    if (context != "") {
        error_message << context << " :: ";
    }
    error_message << message;
    error_message << "\033[0m";
    return error_message.str();
}

int expect(bool assertion, std::string message, std::string context = "") {
    if (!assertion) {
        std::cout << error_message(message, context) << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect(int expectation, int reality, std::string message, std::string context = "") {
    if (expectation != reality) {
        std::cout << error_message(expectation, reality, message, context) << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect(float expectation, float reality, std::string message, std::string context = "") {
    float const epsilon = std::numeric_limits<float>::epsilon();    
    if (std::abs(expectation - reality) >= epsilon) {
        std::cout << std::setprecision(15) << error_message(expectation, reality, message, context) << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect_fuzzy(float expectation, float reality, std::string message, std::string context = "", unsigned int fuzziness = 2) {
    float const epsilon = fuzziness * std::numeric_limits<float>::epsilon();    
    if (std::abs(expectation - reality) >= epsilon) {
        std::cout << std::setprecision(15) << error_message(expectation, reality, message, context) << std::endl;
        return 1;
    } else {
        return 0;
    }
}

int expect(std::string expectation, std::string reality, std::string message, std::string context = "") {
    if (expectation != reality) {
        std::cout << error_message(expectation, reality, message, context) << std::endl;
        return 1;
    } else {
        return 0;
    }
}

template <typename T>
int expect(T const & expectation, T const & reality, std::string message, std::string context = "") {
    if (expectation != reality) {
        std::cout << error_message(expectation, reality, message, context) << std::endl;
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
