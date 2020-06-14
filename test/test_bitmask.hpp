#include "../src/bitmask.hpp"

int test_bitmask(void) {
    int failures = 0;
    int size = 4;

    bitset alternate(size);
    alternate[0] = 0;
    alternate[1] = 1;
    alternate[2] = 0;
    alternate[3] = 1;

    bitset half(size);
    half[0] = 0;
    half[1] = 0;
    half[2] = 1;
    half[3] = 1;

    Bitmask x = Bitmask::ones(size);
    Bitmask y = Bitmask::zeros(size);
    Bitmask z = Bitmask(alternate);
    Bitmask w = Bitmask(half);

    Bitmask a = Bitmask::ones(size);
    Bitmask b = Bitmask::zeros(size);
    Bitmask c = Bitmask(alternate);
    Bitmask d = Bitmask(half);

    failures += expect(0, w[0], "Bitmask::[] Value Equality Test Failed.");
    failures += expect(0, w[1], "Bitmask::[] Value Equality Test Failed.");
    failures += expect(1, w[2], "Bitmask::[] Value Equality Test Failed.");
    failures += expect(1, w[3], "Bitmask::[] Value Equality Test Failed.");

    failures += expect("1100", w.to_string(), "Bitmask::to_string Value Equality Test Failed.");
    failures += expect(2, w.count(), "Bitmask::count Test Failed.");
    failures += expect(4, w.size(), "Bitmask::size Test Failed.");

    failures += expect(true, w <= w, "Bitmask::<= Relational Operator Test Failed.");
    failures += expect(true, w >= w, "Bitmask::>= Relational Operator Test Failed.");
    failures += expect(false, w < w, "Bitmask::<= Relational Operator Test Failed.");
    failures += expect(false, w > w, "Bitmask::>= Relational Operator Test Failed.");

    failures += expect(true, x == a, "Bitmask::== Value Equality Test Failed.");
    failures += expect(true, x == x, "Bitmask::== Referential Equality Failed.");
    failures += expect(true, y == ~x, "Bitmask::~ Negation Value Test Failed");
    failures += expect(true, x == ~~x, "Bitmask::~ Negation Value Test Failed");
    failures += expect(true, y == (x & y), "Bitmask::& Intersection Value Test Failed");
    failures += expect(true, x == (x | y), "Bitmask::| Union Value Test Failed");
    failures += expect(true, y == (z ^ z), "Bitmask::^ XOR Value Test Failed");
    failures += expect(true, (x & ~y) == (x - y), "Bitmask::- Difference Value Test Failed");
    failures += expect(2, w.count(), "Bitmask::count Count Test Failed");

    // failures += expect(4, Bitmask::population(), "Bitmask::population Test Failed.");

    Bitmask intersection_bitmask = z & w; // 0001
    Bitmask union_bitmask = z | w; // 0111
    Bitmask xor_bitmask = z ^ w; // 0110
    Bitmask difference_bitmask = z - w; // 0100
    Bitmask negation_bitmask = ~z; // 1010

    // failures += expect(9, Bitmask::population(), "Bitmask::population Test Failed.");

    return failures;

}