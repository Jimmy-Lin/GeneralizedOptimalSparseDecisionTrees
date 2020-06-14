#include "../src/key.hpp"

int test_key(void) {
    int failures = 0;

    // // Test Key::Key(indicator);
    // // Test Key::Key(indicator, feature_index);
    bitset indicator_value(4);
    indicator_value[0] = 0;
    indicator_value[1] = 1;
    indicator_value[2] = 0;
    indicator_value[3] = 1;
    Bitmask indicator = Bitmask(indicator_value);

    Key key(indicator);

    // Test Key::indicator()
    failures += expect(true, indicator == key.indicator(), "Test Key::indicator reference Failed.");

    // Test Key::==(other)
    failures += expect(true, key == key, "Test Key::== Operator.");

    return failures;

}