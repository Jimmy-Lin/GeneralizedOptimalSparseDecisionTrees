#include "../src/bitmask.hpp"

int test_bitmask(void) {
    int failures = 0;
    unsigned int bitblock_size = 8 * sizeof(bitblock);

    {
        std::string context = "Test Bitmask Layout with single under-full block";
        // Plan the layout of an array of bits
        unsigned int size = bitblock_size  - 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);
        failures += expect((unsigned int) 1, num_blocks, "Bitmask::block_layout A provided incorrect block requirement.", context);
        failures += expect(bitblock_size  - 10, offset, "Bitmask::block_layout A provided incorrect block offset.", context);
    }
    
    {
        std::string context = "Test Bitmask Layout with single full block";
        // Plan the layout of an array of bits
        unsigned int size = bitblock_size ;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);
        failures += expect((unsigned int) 1, num_blocks, "Bitmask::block_layout B provided incorrect block requirement.", context);
        failures += expect((unsigned int) 0, offset, "Bitmask::block_layout B provided incorrect block offset.", context);
    }

    {
        std::string context = "Test Bitmask Layout with single full block and one under-full block";
        // Plan the layout of an array of bits
        unsigned int size = bitblock_size + 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);
        failures += expect((unsigned int) 2, num_blocks, "Bitmask::block_layout C provided incorrect block requirement.", context);
        failures += expect((unsigned int) 10, offset, "Bitmask::block_layout C provided incorrect block offset.", context);
    }

    {
        std::string context = "Test Bitmask construction of zeros and ones with single under-full block";
        unsigned int size = bitblock_size  - 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        // Initialize the bits to zero
        Bitmask::zeros(blocks, size);
        
        // Observe the bits
        std::string bitstring;
        bitstring = Bitmask::to_string(blocks, size);
        failures += expect(size, (int) bitstring.size(), "Bitmask::to_string provided incorrect string length.", context);
        failures += expect(std::string(size, '0'), bitstring, "Bitmask::to_string produced incorrect string representation.", context);

        // Initialize the bits to zero
        Bitmask::ones(blocks, size);
        
        // Observe the bits
        bitstring = Bitmask::to_string(blocks, size);
        failures += expect(size, (int) bitstring.size(), "Bitmask::to_string provided incorrect string length.", context);
        failures += expect(std::string(size, '1'), bitstring, "Bitmask::to_string produced incorrect string representation.", context);
    }

    {
        std::string context = "Test Bitmask construction of zeros and ones with single full block";
        unsigned int size = bitblock_size;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        // Initialize the bits to zero
        Bitmask::zeros(blocks, size);
        
        // Observe the bits
        std::string bitstring;
        bitstring = Bitmask::to_string(blocks, size);
        failures += expect(size, (int) bitstring.size(), "Bitmask::to_string provided incorrect string length.", context);
        failures += expect(std::string(size, '0'), bitstring, "Bitmask::to_string produced incorrect string representation.", context);

        // Initialize the bits to zero
        Bitmask::ones(blocks, size);
        
        // Observe the bits
        bitstring = Bitmask::to_string(blocks, size);
        failures += expect(size, (int) bitstring.size(), "Bitmask::to_string provided incorrect string length.", context);
        failures += expect(std::string(size, '1'), bitstring, "Bitmask::to_string produced incorrect string representation.", context);
    }

    {
        std::string context = "Test Bitmask construction of zeros and ones with single full block and one underfull block";

        unsigned int size = bitblock_size + 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        // Initialize the bits to zero
        Bitmask::zeros(blocks, size);
        
        // Observe the bits
        std::string bitstring;
        bitstring = Bitmask::to_string(blocks, size);
        failures += expect(size, (int) bitstring.size(), "Bitmask::to_string provided incorrect string length.", context);
        failures += expect(std::string(size, '0'), bitstring, "Bitmask::to_string produced incorrect string representation.", context);

        // Initialize the bits to zero
        Bitmask::ones(blocks, size);
        
        // Observe the bits
        bitstring = Bitmask::to_string(blocks, size);
        failures += expect(size, (int) bitstring.size(), "Bitmask::to_string provided incorrect string length.", context);
        failures += expect(std::string(size, '1'), bitstring, "Bitmask::to_string produced incorrect string representation.", context);
    }

    {
        std::string context = "Test Bitmask count with mixed bits";
        unsigned int size = 32;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        for (unsigned int i = 0; i < size; ++i) {
            Bitmask::set(blocks, size, i, i % 2);
        }
        
        // Observe the bits
        failures += expect(16, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);
    }

    { //
        std::string context = "Test Bitmask bit count with single under-full block";
        unsigned int size = bitblock_size  - 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        // Initialize the bits to zero
        Bitmask::zeros(blocks, size);
        failures += expect((unsigned int) 0, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);

        // Initialize the bits to ones
        Bitmask::ones(blocks, size);
        failures += expect(size, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);
    }


    {
        std::string context = "Test Bitmask bit count with single full block";
        unsigned int size = bitblock_size;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        // Initialize the bits to zero
        Bitmask::zeros(blocks, size);
        failures += expect((unsigned int) 0, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);

        // Initialize the bits to zero
        Bitmask::ones(blocks, size);
        failures += expect(size, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);
    }

    {
        std::string context = "Test Bitmask bit count with single full block and one underfull block";
        unsigned int size = bitblock_size  + 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        // Initialize the bits to zero
        Bitmask::zeros(blocks, size);
        failures += expect((unsigned int) 0, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);

        // Initialize the bits to ones
        Bitmask::ones(blocks, size);
        failures += expect(size, Bitmask::count(blocks, size), "Bitmask::count provided incorrect bit count.", context);
    }

    {
        std::string context = "Test Bitmask word countwith mixed bits";
        unsigned int size = 64;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        for (unsigned int i = 0; i < size; ++i) {
            Bitmask::set(blocks, size, i, i % 2);
        }
        
        // Observe the bits
        failures += expect(32, Bitmask::words(blocks, size), "Bitmask::word provided incorrect word count.", context);
    }

    {
        std::string context = "Test Bitmask word count with mixed bits";
        unsigned int size = 64;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        for (unsigned int i = 0; i < size; ++i) {
            Bitmask::set(blocks, size, i, i % 2);
        }
        
        // Observe the bits
        failures += expect(32, Bitmask::words(blocks, size), "Bitmask::word provided incorrect word count.", context);
    }

    {
        std::string context = "Test Bitmask word count with non-trivial blocks";
        unsigned int size = 10;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);

        Bitmask::zeros(blocks, size);

        Bitmask::set(blocks, size, 0, 1);
        Bitmask::set(blocks, size, 1, 1);
        Bitmask::set(blocks, size, 2, 1);

        Bitmask::set(blocks, size, 5, 1);
        Bitmask::set(blocks, size, 6, 1);
        Bitmask::set(blocks, size, 7, 1);

        Bitmask::set(blocks, size, 9, 1);
        
        // Observe the bits
        failures += expect(3, Bitmask::words(blocks, size), "Bitmask::word provided incorrect word count.", context);
    }

    {
        std::string context = "Test Bitmask multi-block words estimation";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, & num_blocks, & offset);

        // Allocate the stack space
        bitblock * blocks = (bitblock *) alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(blocks, size);

        // Block-0 Words
        Bitmask::set(blocks, size, 0, 1);
        Bitmask::set(blocks, size, 1, 1);
        Bitmask::set(blocks, size, 2, 1);

        Bitmask::set(blocks, size, 15, 1);
        Bitmask::set(blocks, size, 16, 1);
        Bitmask::set(blocks, size, 17, 1);

        Bitmask::set(blocks, size, 20, 1);
        Bitmask::set(blocks, size, 21, 1);
        Bitmask::set(blocks, size, 22, 1);

        Bitmask::set(blocks, size, 34, 1);
        Bitmask::set(blocks, size, 35, 1);
        Bitmask::set(blocks, size, 36, 1);

        Bitmask::set(blocks, size, 44, 1);
        Bitmask::set(blocks, size, 45, 1);
        Bitmask::set(blocks, size, 46, 1);

        // Oeverlap Word (Both Block 0 and 1)

        Bitmask::set(blocks, size, 62, 1);
        Bitmask::set(blocks, size, 63, 1);
        Bitmask::set(blocks, size, 64, 1);
        Bitmask::set(blocks, size, 65, 1);

        // Block-1 Words

        Bitmask::set(blocks, size, 75, 1);
        Bitmask::set(blocks, size, 76, 1);
        Bitmask::set(blocks, size, 77, 1);

        Bitmask::set(blocks, size, 80, 1);
        Bitmask::set(blocks, size, 81, 1);
        Bitmask::set(blocks, size, 82, 1);

        Bitmask::set(blocks, size, 94, 1);
        Bitmask::set(blocks, size, 95, 1);
        Bitmask::set(blocks, size, 96, 1);

        Bitmask::set(blocks, size, 104, 1);
        Bitmask::set(blocks, size, 105, 1);
        Bitmask::set(blocks, size, 106, 1);
        
        // Observe the bits
        failures += expect(10, Bitmask::words(blocks, size), "Bitmask::word provided incorrect word count.", context);
    }

    {
        std::string context = "Test Bitmask bitwise logical AND";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, &num_blocks, &offset);

        // Allocate the stack space
        bitblock * a = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(a, size);
        bitblock * b = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(b, size);
        
        bitblock * d = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(d, size);
        bitblock * e = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(d, size);

        for (unsigned int i = 0; i < 30; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(d, size, i, 0);
            Bitmask::set(e, size, i, 0);
        }
        for (unsigned int i = 30; i < 60; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(d, size, i, 0);
            Bitmask::set(e, size, i, 1);
        }
        for (unsigned int i = 60; i < 90; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(d, size, i, 0);
            Bitmask::set(e, size, i, 0);
        }
        for (unsigned int i = 90; i < size; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(d, size, i, 1);
            Bitmask::set(e, size, i, 0);
        }

       // Sanity Test
        failures += expect("00000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111111111111111111111111111111", Bitmask::to_string(a, size), "Sanity test for a", context);
        failures += expect("00000000000000000000000000000011111111111111111111111111111100000000000000000000000000000011111111111111111111111111111111111111", Bitmask::to_string(b, size), "Sanity test for b", context);
        failures += expect("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111", Bitmask::to_string(d, size), "Sanity test for d", context);
        failures += expect("00000000000000000000000000000011111111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000", Bitmask::to_string(e, size), "Sanity test for e", context);

        bitblock * c = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::copy(b, c, size);
        failures += expect(Bitmask::equals(b, c, size), "Bitmask::equals is not consistent with Bitmask::copy", context);

        // Sanity Test
        failures += expect("00000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111111111111111111111111111111", Bitmask::to_string(a, size), "Sanity test for a", context);
        failures += expect("00000000000000000000000000000011111111111111111111111111111100000000000000000000000000000011111111111111111111111111111111111111", Bitmask::to_string(b, size), "Sanity test for b", context);
        failures += expect("00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111", Bitmask::to_string(d, size), "Sanity test for d", context);
        failures += expect("00000000000000000000000000000011111111111111111111111111111100000000000000000000000000000000000000000000000000000000000000000000", Bitmask::to_string(e, size), "Sanity test for e", context);

        Bitmask::bit_and(a, b, size);
        for (unsigned int i = 0; i < size; ++i) {
            std::stringstream message;
            message << "Bitmask::bit_and(a, b, size, false) => b == d provided wrong bit at index " << i;
            failures += expect(Bitmask::get(d, size, i), Bitmask::get(b, size, i), message.str(), context);
        }

        Bitmask::bit_and(a, c, size, true);
        for (unsigned int i = 0; i < size; ++i) {
            std::stringstream message;
            message << "Bitmask::bit_and(a, c, size, true) => c == e provided wrong bit at index " << i;
            failures += expect(Bitmask::get(e, size, i), Bitmask::get(c, size, i), message.str(), context);
        }
    }

    {
        std::string context = "Test Bitmask bitwise logical OR";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, &num_blocks, &offset);

        // Allocate the stack space
        bitblock * a = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(a, size);
        bitblock * b = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(b, size);
        
        bitblock * d = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(d, size);
        bitblock * e = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(d, size);

        for (unsigned int i = 0; i < 30; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(d, size, i, 0);
            Bitmask::set(e, size, i, 1);
        }
        for (unsigned int i = 30; i < 60; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(d, size, i, 1);
            Bitmask::set(e, size, i, 1);
        }
        for (unsigned int i = 60; i < 90; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(d, size, i, 1);
            Bitmask::set(e, size, i, 0);
        }
        for (unsigned int i = 90; i < size; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(d, size, i, 1);
            Bitmask::set(e, size, i, 1);
        }

        bitblock * c = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        // Compute c = b
        Bitmask::copy(b, c, size);
        failures += expect(Bitmask::equals(b, c, size), "Bitmask::equals is not consistent with Bitmask::copy", context);

        // Compute b = a | b
        Bitmask::bit_or(a, b, size);
        // Compare b = d
        for (unsigned int i = 0; i < size; ++i) {
            std::stringstream message;
            message << "Bitmask::bit_or(a, b, size, false) => b == d provided wrong bit at index " << i;
            failures += expect(Bitmask::get(d, size, i), Bitmask::get(b, size, i), message.str(), context);
        }

        // Compute c = ~a | c
        Bitmask::bit_or(a, c, size, true);
        // Compare c = e
        for (unsigned int i = 0; i < size; ++i) {
            std::stringstream message;
            message << "Bitmask::bit_or(a, c, size, true) => c == e provided wrong bit at index " << i;
            failures += expect(Bitmask::get(e, size, i), Bitmask::get(c, size, i), message.str(), context);
        }
    }

{
        std::string context = "Test Bitmask bitwise logical XOR";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, &num_blocks, &offset);

        // Allocate the stack space
        bitblock * a = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(a, size);
        bitblock * b = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(b, size);
        
        bitblock * d = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(d, size);
        bitblock * e = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(d, size);

        for (unsigned int i = 0; i < 30; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(d, size, i, 0);
            Bitmask::set(e, size, i, 1);
        }
        for (unsigned int i = 30; i < 60; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(d, size, i, 1);
            Bitmask::set(e, size, i, 0);
        }
        for (unsigned int i = 60; i < 90; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(d, size, i, 1);
            Bitmask::set(e, size, i, 0);
        }
        for (unsigned int i = 90; i < size; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(d, size, i, 0);
            Bitmask::set(e, size, i, 1);
        }

        bitblock * c = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        // Compute c = b
        Bitmask::copy(b, c, size);
        failures += expect(Bitmask::equals(b, c, size), "Bitmask::equals is not consistent with Bitmask::copy", context);

        // Compute b = a ^ b
        Bitmask::bit_xor(a, b, size);
        // Compare b = d
        for (unsigned int i = 0; i < size; ++i) {
            std::stringstream message;
            message << "Bitmask::bit_xor(a, b, size, false) => b == d provided wrong bit at index " << i;
            failures += expect(Bitmask::get(d, size, i), Bitmask::get(b, size, i), message.str(), context);
        }

        // Compute c = ~a ^ c
        Bitmask::bit_xor(a, c, size, true);
        // Compare c = e
        for (unsigned int i = 0; i < size; ++i) {
            std::stringstream message;
            message << "Bitmask::bit_xor(a, c, size, true) => c == e provided wrong bit at index " << i;
            failures += expect(Bitmask::get(e, size, i), Bitmask::get(c, size, i), message.str(), context);
        }
    }

    {
        std::string context = "Test Bitmask Equality and Hashing";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, &num_blocks, &offset);

        // Allocate the stack space
        bitblock * a = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(a, size);
        bitblock * b = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(b, size);
        bitblock * c = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(c, size);

        for (unsigned int i = 0; i < 30; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(c, size, i, 0);
        }
        for (unsigned int i = 30; i < 60; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(c, size, i, 1);
        }
        for (unsigned int i = 60; i < 90; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 0);
            Bitmask::set(c, size, i, 1);
        }
        for (unsigned int i = 90; i < size; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
            Bitmask::set(c, size, i, 0);
        }
        failures += expect(Bitmask::equals(a, b, size), "Bitmask::equals(a, b) == true", context);
        failures += expect(Bitmask::equals(a, c, size) == false, "Bitmask::equals(a, c) == false", context);
        failures += expect(Bitmask::hash(a, size), Bitmask::hash(b, size), "Bitmask::hash(a) == Bitmask::hash(b)", context);
        failures += expect(Bitmask::hash(a, size) != Bitmask::hash(c, size), "Bitmask::hash(a) != Bitmask::hash(c)", context);
    }

    {
        std::string context = "Test Bitmask Ordinal Comparison";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, &num_blocks, &offset);

        // Allocate the stack space
        bitblock * a = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(a, size);
        bitblock * b = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(b, size);
       

        for (unsigned int i = 0; i < 30; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 0);
        }
        for (unsigned int i = 30; i < 60; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
        }
        for (unsigned int i = 60; i < 90; ++i) {
            Bitmask::set(a, size, i, 0);
            Bitmask::set(b, size, i, 1);
        }
        for (unsigned int i = 90; i < size; ++i) {
            Bitmask::set(a, size, i, 1);
            Bitmask::set(b, size, i, 1);
        }
        bitblock * c = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::copy(a, c, size);
        
        failures += expect(false, Bitmask::less_than(a, a, size), "Bitmask::less_than(a, b)", context);
        failures += expect(false, Bitmask::greater_than(a, a, size), "Bitmask::less_than(a, b)", context);
        failures += expect(false, Bitmask::less_than(a, c, size), "Bitmask::less_than(a, c)", context);
        failures += expect(false, Bitmask::greater_than(a, c, size), "Bitmask::less_than(a, c)", context);
        failures += expect(true, Bitmask::less_than(a, b, size), "Bitmask::less_than(a, b)", context);
        failures += expect(false, Bitmask::greater_than(a, b, size), "Bitmask::less_than(a, b)", context);
        failures += expect(false, Bitmask::less_than(b, a, size), "Bitmask::less_than(b, a)", context);
        failures += expect(true, Bitmask::greater_than(b, a, size), "Bitmask::less_than(b, a)", context);
    }

    {
        std::string context = "Test Bitmask Ordinal Comparison";
        unsigned int size = 128;
        unsigned int num_blocks, offset;
        Bitmask::block_layout(size, &num_blocks, &offset);

        // Allocate the stack space
        bitblock * a = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(a, size);
        bitblock * b = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
        Bitmask::zeros(b, size);
       

        for (unsigned int i = 0; i < 30; ++i) {
            Bitmask::set(a, size, i, 0);
        }
        for (unsigned int i = 30; i < 60; ++i) {
            Bitmask::set(a, size, i, 1);
        }
        for (unsigned int i = 60; i < 90; ++i) {
            Bitmask::set(a, size, i, 0);
        }
        for (unsigned int i = 90; i < size; ++i) {
            Bitmask::set(a, size, i, 1);
        }
        
        Bitmask mask(a, size);
        Bitmask other_mask(a, size);
        mask.copy_to(b);
        
        failures += expect(true, Bitmask::equals(mask.data(), a, size), "Instance comparison with import source", context);
        failures += expect(true, Bitmask::equals(mask.data(), b, size), "Instance comparison with export destination", context);
        failures += expect(true, Bitmask::equals(a, b, size), "Comparison between import and export", context);
        failures += expect(true, mask == other_mask, "Bitmask::operator==", context);
    }

    // FIREWOLF: boost has been removed
//    {
//        std::string context = "Test consistency across different constructors";
//        unsigned int size = 128;
//        unsigned int num_blocks, offset;
//        Bitmask::block_layout(size, &num_blocks, &offset);
//
//        // Allocate the stack space
//        bitblock * a_source = (bitblock *)alloca(sizeof(bitblock) * num_blocks);
//        Bitmask::zeros(a_source, size);
//        Bitmask a(a_source, size);
//
//        Bitmask b(size, false);
//
//        dynamic_bitset c_source(size);
//        for (unsigned int i = 0; i < size; ++i) { c_source[i] = 0; }
//        Bitmask c(c_source);
//
//        Bitmask d(c);
//
//        failures += expect(true, a == a, "Bitmask::operator==(a,a)", context);
//        failures += expect(true, a == b, "Bitmask::operator==(a,b)", context);
//        failures += expect(true, a == c, "Bitmask::operator==(a,c)", context);
//        failures += expect(true, a == d, "Bitmask::operator==(a,d)", context);
//        failures += expect(true, b == a, "Bitmask::operator==(b,a)", context);
//        failures += expect(true, b == b, "Bitmask::operator==(b,b)", context);
//        failures += expect(true, b == c, "Bitmask::operator==(b,c)", context);
//        failures += expect(true, b == d, "Bitmask::operator==(b,d)", context);
//        failures += expect(true, c == a, "Bitmask::operator==(c,a)", context);
//        failures += expect(true, c == b, "Bitmask::operator==(c,b)", context);
//        failures += expect(true, c == c, "Bitmask::operator==(c,c)", context);
//        failures += expect(true, c == d, "Bitmask::operator==(c,d)", context);
//        failures += expect(true, d == a, "Bitmask::operator==(d,a)", context);
//        failures += expect(true, d == b, "Bitmask::operator==(d,b)", context);
//        failures += expect(true, d == c, "Bitmask::operator==(d,c)", context);
//        failures += expect(true, d == d, "Bitmask::operator==(d,d)", context);
//    }

    {
        std::string context = "Test Stack Buffer Bitmasks";

        unsigned int size = 8 * sizeof(bitblock);
        bitblock * buffer = (bitblock *) alloca(sizeof(bitblock));
        Bitmask mask(size, true, buffer);

        failures += expect(Bitmask::equals(mask.data(), buffer, size) ,"Source-Instance Equality Before Direct Manipulation" , context);
        Bitmask::zeros(buffer, size);
        failures += expect(Bitmask::equals(mask.data(), buffer, size) ,"Source-Instance Equality After Direct Manipulation" , context);
        Bitmask other_mask(size, true, buffer);
        failures += expect(mask == other_mask,"Referentially equal instances" , context);
    }

    {
        std::string context = "Test Bitmask Ordering";
        unsigned int size = 12;

        Bitmask a(size); // 000111110010
        a.set(3); a.set(4); a.set(5); a.set(6); a.set(7); a.set(10);
        failures += expect("000111110010", a.to_string(), "Initalization Sanity Test Failed", context);

        Bitmask b(size); // 001111110010
        b.set(2); b.set(3); b.set(4); b.set(5); b.set(6); b.set(7); b.set(10);
        failures += expect("001111110010", b.to_string(), "Initalization Sanity Test Failed", context);
        
        Bitmask c(size); // 000011110001
        c.set(4); c.set(5); c.set(6); c.set(7); c.set(11);
        failures += expect("000011110001", c.to_string(), "Initalization Sanity Test Failed", context);

        Bitmask d(size); // 000111110001
        d.set(3); d.set(4); d.set(5); d.set(6); d.set(7); d.set(11);
        failures += expect("000111110001", d.to_string(), "Initalization Sanity Test Failed", context);

        Bitmask e(size); // 001111100001
        e.set(2); e.set(3); e.set(4); e.set(5); e.set(6); e.set(11);
        failures += expect("001111100001", e.to_string(), "Initalization Sanity Test Failed", context);

        Bitmask f(size); // 000001111000
        f.set(5); f.set(6); f.set(7); f.set(8);
        failures += expect("000001111000", f.to_string(), "Initalization Sanity Test Failed", context);

        // Expected Ordering: f < a < b < e < c < d

        failures += expect(false, f < f, "Bitmask::operator<(f, f)", context);
        failures += expect(true, f < a, "Bitmask::operator<(f, a)", context);
        failures += expect(true, f < b, "Bitmask::operator<(f, b)", context);
        failures += expect(true, f < e, "Bitmask::operator<(f, e)", context);
        failures += expect(true, f < c, "Bitmask::operator<(f, c)", context);
        failures += expect(true, f < d, "Bitmask::operator<(f, d)", context);

        failures += expect(false, a < f, "Bitmask::operator<(a, f)", context);
        failures += expect(false, a < a, "Bitmask::operator<(a, a)", context);
        failures += expect(true, a < b, "Bitmask::operator<(a, b)", context);
        failures += expect(true, a < e, "Bitmask::operator<(a, e)", context);
        failures += expect(true, a < c, "Bitmask::operator<(a, c)", context);
        failures += expect(true, a < d, "Bitmask::operator<(a, d)", context);

        failures += expect(false, b < f, "Bitmask::operator<(b, f)", context);
        failures += expect(false, b < a, "Bitmask::operator<(b, a)", context);
        failures += expect(false, b < b, "Bitmask::operator<(b, b)", context);
        failures += expect(true, b < e, "Bitmask::operator<(b, e)", context);
        failures += expect(true, b < c, "Bitmask::operator<(b, c)", context);
        failures += expect(true, b < d, "Bitmask::operator<(b, d)", context);

        failures += expect(false, e < f, "Bitmask::operator<(e, f)", context);
        failures += expect(false, e < a, "Bitmask::operator<(e, a)", context);
        failures += expect(false, e < b, "Bitmask::operator<(e, b)", context);
        failures += expect(false, e < e, "Bitmask::operator<(e, e)", context);
        failures += expect(true, e < c, "Bitmask::operator<(e, c)", context);
        failures += expect(true, e < d, "Bitmask::operator<(e, d)", context);

        failures += expect(false, c < f, "Bitmask::operator<(c, f)", context);
        failures += expect(false, c < a, "Bitmask::operator<(c, a)", context);
        failures += expect(false, c < b, "Bitmask::operator<(c, b)", context);
        failures += expect(false, c < e, "Bitmask::operator<(c, e)", context);
        failures += expect(false, c < c, "Bitmask::operator<(c, c)", context);
        failures += expect(true, c < d, "Bitmask::operator<(c, d)", context);

        failures += expect(false, d < f, "Bitmask::operator<(d, f)", context);
        failures += expect(false, d < a, "Bitmask::operator<(d, a)", context);
        failures += expect(false, d < b, "Bitmask::operator<(d, b)", context);
        failures += expect(false, d < e, "Bitmask::operator<(d, e)", context);
        failures += expect(false, d < c, "Bitmask::operator<(d, c)", context);
        failures += expect(false, d < d, "Bitmask::operator<(d, d)", context);
    }
    return failures;
}