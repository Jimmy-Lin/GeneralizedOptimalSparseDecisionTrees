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
#include <csv/csv.h>

#include "bitmask.hpp"
#include "configuration.hpp"
#include "integrity_violation.hpp"

// Translation Object used to convert arbitrary data sets into binary data sets
// This class also holds all information needed for undoing the translation
// Encoding Semantics:
// Header:
//  - Headers are required
// Column Type Encoding:
//  - Rational: Decimal values are encoded using greater-than-or-equal conditions that are midpoints between observed non-null values
//  - Integral: Integer values are encoded using greater-than-or-equal conditions that are observed non-null values
//  - Enumerable: Integer values with less than or equal to 5 unique values are encoded using equal-to conditions for each unique non-null value
//  - Binary: Non-null columns with 2 unique values are encoded using a single equal-to condition of the smaller value
//  - Redundant: Columns with 1 or less unique values are not encoded
//  - Categorical: Remaining columns are encoded using equal-to conditions for each unqiue non-null value
//  - Target Column is always encoded using equal-to conditions for each unique non-null value
//  Type Inference:
//  - Rational: Must conform to the following expression ^[+-]?\d+\.\d*$
//  - Integral: Must conform to the following expression ^[+-]?\d+\$
//  - Null: Matches an empty string or one of many common representative strings (see definition for details)
//  - Categorical: Any value that fails all other inference conditions
class Encoder {
public:
    Encoder(void);
    // @require input csv must satisfy the following preconditions:
    //  - file starts with a header row
    //  - elements are not quote-wrapped
    //  - elements are not padded
    //  - elements contain no commas
    // @param input: Input stream of bytes containing contents of a CSV
    Encoder(std::istream & input);

    // @param string: a string that may contain a number
    // @return whether string qualifies as a string representation of an integer (Using pattern: ^[+-]?\d+$)
    static bool test_integral(std::string const & string);

    // @param string: a string that may contain a number
    // @return whether string qualifies as a floating point (Using pattern: ^[+-]?\d+\.\d*$)
    static bool test_rational(std::string const & string);

    // @return matrix of string elements of the csv
    // std::vector< std::vector< std::string > > const & read_rows(void) const;
    // @return matrix of bit elements of the binary csv
    std::vector< Bitmask > const & read_binary_rows(void) const;
    // @return the number of features read as input
    unsigned int features(void) const;
    // @return the number of targets read as input
    unsigned int targets(void) const;
    // @return the number of features after encoding
    unsigned int binary_features(void) const;
    // @return the number of targets after encoding
    unsigned int binary_targets(void) const;
    // @return the number of samples in the dataset
    unsigned int samples(void) const;

    // Returns: the pre-encode feature index and the offset to the binary feature generated
    // @param encoded_column_index: the index of the binary feature
    // @param decoded_column_index: the index of the original feature
    // @param encoding_offset: the offset referring of the binary feature from the orignal feature's value set
    // @modifies encoded_column_index: intialized with the index of the original feature
    // @modifies encoding_offset: initialized with the offset referring of the binary feature from the orignal feature's value set
    void decode(unsigned int encoded_column_index, unsigned int * decoded_column_index) const;

    // @param decoded_column_index: column index of the original data set
    // @param header: string to contain the name of the corresponding column
    // @modifies header: initialized with the name of the column in the original data set
    void header(unsigned int decoded_column_index, std::string & header) const;

    // @param header: string to contain the name of the target column
    // @modifies header: initialized with the name of the target column in the original data set
    void header(std::string & header) const;

    // @param decoded_column_index: the column index of the original dataset
    // @param reference: the reference values used to specify a particular encoding of the feature at decoded_column_index
    // @param encoded_column_index: the index of the binary feature generated from splitting the feature at decoded_column_index at the reference value
    // @modifies encoded_column_index: initialized to the index of the binary feature generated from splitting the feature at decoded_column_index at the reference value
    void find_encoding(unsigned int decoded_column_index, std::string const & reference, unsigned int * encoded_column_index) const;

    // @param encoded_column_index: the column index for the binary data set
    // @param type: string to contain the type of the specified column
    // @param relation: string to contain the relation used for the specified column
    // @param reference: string to contain the reference value which specifies the binary relation into a predicate
    // @modifies type: initialized to contain the type of the specified column
    // @modifies relation: initialized to contain the relation used for the specified column
    // @modifies reference: initialized to contain the reference value which specifies the binary relation into a predicate
    void encoding(unsigned int encoded_column_index, std::string & type, std::string & relation, std::string & reference) const;

    // @param value_index: the index to the particular value encoded in the target column
    // @param value: the value_index-th target value
    // @modifies value: initializes value with the value_index-th target value

    void target_value(unsigned int value_index, std::string & value) const;
    // @param type: string representing the type inferred for the target column
    // @modifies type: initializes with a string representing the type inferred for the target column
    void target_type(std::string & type) const;

    // @param index: index of the sample being queried
    // @returns the importance given to the queried sample
    // float weight(unsigned int index) const;

    // The boundary indices for binary features that belong to the same ordinal feature.
    std::vector< std::pair< unsigned int, unsigned int > > boundaries;

private:
    // Original data
    std::vector< std::string > headers;

    // Dimensions of the dataset
    unsigned int number_of_rows = 0;
    unsigned int number_of_columns = 0;
    unsigned int number_of_binary_columns = 0;
    unsigned int number_of_binary_targets = 0;

    // The importance given to each sample
    std::vector< float > weights;

    // Summaries used to describe each column
    std::vector< std::set< std::string > > values;
    std::vector< unsigned int > cardinalities;
    std::vector< bool > optionalities;
    std::vector< std::string > inferred_types;

    // Codex with Entries of Format: < Source Index, Type, Relation, Reference >
    std::vector< std::pair< unsigned int, std::vector< std::string > > > codex;

    // Binary representation of rows
    std::vector< Bitmask > binary_rows;

    // @param number: input number to reduce precision
    // @return an input equivalent to number rounded to k significant figures
    // @note digits of value 5 are rounded away from 0
    int limit_precision(int number) const;
    float limit_precision(float number) const;

    // @param input: an input stream containing a CSV
    // @modifies tokens: stores the tokens of input in tokens in row-major order
    // @modifies this -> header: stores the csv header 
    // @modifies this -> number_of_rows: stores the number of rows (excluding header)
    void tokenize(std::istream & input, std::vector< std::vector< std::string > > & tokens);

    // @param rows: csv of tokens
    // @modifies this -> values: stores the sets of unqiue non-null values per column
    // @modifies this -> cardinality: stores the number of unqiue non-null values per column
    // @modifies this -> optional: stores the presence of null values per column
    // @modifies this -> type: stores the inferred type per column
    void parse(std::vector< std::vector< std::string > > const & rows);

    // @param values: sets of unqiue non-null values per column
    // @modifies values: reduced sets of unqiue non-null values after reducing to k significant figures
    void limit_precision(std::vector< std::set< std::string > > & values) const;

    // @modifies this -> codex: initializes a vector of vectors (see definition for codex structure)
    void build(void);

    // @param rows: csv of tokens
    // @param binary_rows: vector of bitmasks representing a binary data set in row-major order
    // @modifies binary_rows: initialized with the binary representation of rows
    void encode(std::vector< std::vector< std::string > > const & rows, std::vector< Bitmask > & binary_rows) const;

    // @modifes: reorders the columns by increasing 1-step information gain
    // @note: Currently unsused
    void reindex(std::vector< std::vector< std::string > > const & rows);
};

#endif
