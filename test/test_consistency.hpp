#include "../src/gosdt.hpp"

int test_consistency(void) {
    int failures = 0;

    {
        std::string context = "Test Consistency test/fixtures/binary_sepal";
        std::ifstream data("test/fixtures/binary_sepal.csv");
        std::ifstream expectation("test/fixtures/binary_sepal.json");
        std::stringstream buffer;
        buffer << expectation.rdbuf();

        GOSDT model;
        std::string result;
        model.fit(data, result);

        failures += expect(buffer.str(), result, "Consistency Test test/fixtures/binary_sepal", context);
    }

    {
        std::string context = "Test Consistency test/fixtures/dataset";
        std::ifstream data("test/fixtures/dataset.csv");
        std::ifstream expectation("test/fixtures/dataset.json");
        std::stringstream buffer;
        buffer << expectation.rdbuf();

        GOSDT model;
        std::string result;
        model.fit(data, result);

        failures += expect(buffer.str(), result, "Consistency Test test/fixtures/dataset", context);
    }

    {
        std::string context = "Test Consistency test/fixtures/sequences";
        std::ifstream data("test/fixtures/sequences.csv");
        std::ifstream expectation("test/fixtures/sequences.json");
        std::stringstream buffer;
        buffer << expectation.rdbuf();

        GOSDT model;
        std::string result;
        model.fit(data, result);

        failures += expect(buffer.str(), result, "Consistency Test test/fixtures/sequences", context);
    }

    {
        std::string context = "Test Consistency test/fixtures/tree";
        std::ifstream data("test/fixtures/tree.csv");
        std::ifstream expectation("test/fixtures/tree.json");
        std::stringstream buffer;
        buffer << expectation.rdbuf();

        GOSDT model;
        std::string result;
        model.fit(data, result);

        failures += expect(buffer.str(), result, "Consistency Test test/fixtures/tree", context);
    }

    return failures;
}