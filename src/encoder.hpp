#ifndef ENCODER_H
#define ENCODER_H

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <math.h>
#include <vector>
#include <set>
#include <unordered_set>
#include <boost/dynamic_bitset.hpp>
#include <tbb/scalable_allocator.h>

#include <csv/csv.h>

#include "bitmask.hpp"

typedef boost::dynamic_bitset< unsigned long, tbb::scalable_allocator< unsigned long > > bitset;

class Encoder {
public:
    Encoder(void);
    Encoder(std::istream & input, unsigned int precision, bool verbose = true);
    std::vector< std::vector< std::string > > const & read_rows(void) const;
    std::vector< bitset > const & read_binary_rows(void) const;
    // Return the number of source features and target features
    std::tuple< unsigned int, unsigned int > split(void) const;

    // Returns: the pre-encode feature index and the offset to the binary feature generated
    // Input: j is the index of the binary feature
    std::tuple< unsigned int, unsigned int > decode(unsigned int j) const;
    // Select the j-th header, where j is the pre-encode feature index
    std::string header(unsigned int j) const;
    std::string header(void) const;
    // Return: vector of < Type, Relation, Reference > which defines the encoding rule
    // Input: j is the pre-encode feature index
    // Input: offset is the index of the particular binary feature generated from the j-th feature
    std::vector< std::string > decoder(unsigned int j, unsigned offset) const;
    std::vector< std::string > decoder(void) const;
    // Return: the k-th unique label used in the target feature 
    std::string label(unsigned int k) const;
private:
    bool verbose;
    unsigned int precision;

    // Original data
    std::vector< std::string > headers;
    std::vector< std::vector< std::string > > rows;

    // Dimensions of the dataset
    unsigned int number_of_features = 0;
    unsigned int number_of_samples = 0;
    unsigned int number_of_binary_features = 0;

    // Summaries used to describe each feature
    std::vector< std::set< std::string > > values;
    std::vector< unsigned int > cardinality;
    std::vector< bool > optional;
    std::vector< std::string > type;

    // Codex with Entries of Format: < Type, Relation, Reference >
    // Used for encoding and decoding
    std::vector< std::vector< std::vector< std::string > > > codex;

    // Encoded data
    std::vector< bitset > binary_rows;

    bool test_integral(std::string const & string) const;
    bool test_rational(std::string const & string) const;
    int limit_precision(int number) const;
    float limit_precision(float number) const;

    // Used to convert csv into element strings
    std::vector< std::vector< std::string > > tokenize(std::istream & input);
    // Used to infer the type of each feature
    void parse(std::vector< std::vector< std::string > > const & rows);
    // Build the encoding rules used to convert to binary
    void build(void);
    // Used to apply the encoding rules
    std::vector< bitset > encode(std::vector< std::vector< std::string > > const & rows) const;
};

#endif
