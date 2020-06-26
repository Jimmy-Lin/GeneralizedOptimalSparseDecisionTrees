#include "test.hpp"
#include "test_bitmask.hpp"
#include "test_index.hpp"
#include "test_queue.hpp"
#include "test_consistency.hpp"

int main() {
    int failures = 0;
    std::map< std::string, int (*)(void) > units;
    units["Bitmask"] = test_bitmask;
    units["Index"] = test_index;
    units["Queue"] = test_queue;
    units["Consistency"] = test_consistency;

    for (std::map< std::string, int (*)(void) >::iterator iterator = units.begin(); iterator != units.end(); ++iterator ) {
        try {
            failures += run_tests(iterator -> first, iterator -> second);
        } catch (char const * exception) {
            std::cout << "\033[1;31m" << "Uncaught Exception in "  << iterator -> first  << " Tests" << "\033[0m" << std::endl;
            std::cout << "\033[1;31m" << "Uncaught Exception: "  << exception << "\033[0m" << std::endl;
            failures += 1;
        }
    }

    if (failures == 0) {
        std::cout << "\033[1;32m" << "All Tests Passed" << "\033[0m" << std::endl;
        return 0;
    } else {
        std::cout << "\033[1;31m" << failures << " Tests Failed" << "\033[0m" << std::endl;
        return 1;
    }
}