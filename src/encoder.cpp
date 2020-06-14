#include "encoder.hpp"

Encoder::Encoder(void) {}

Encoder::Encoder(std::istream & input, unsigned int precision, bool verbose) : precision(precision), verbose(verbose) {
    auto rows = tokenize(input); // Produce string tokens from the input
    parse(rows); // Parse the tokens to infer the type of each feature
    build(); // Build a set of encoding rules for each feature
    this -> binary_rows = encode(rows); // Encode the tokenized data using the encoding rules
}

bool Encoder::test_rational(std::string const & string) const {
    std::string::const_iterator it = string.begin();
    bool decimalPoint = false;
    int min_size = 0;
    if (string.size() > 0 && (string[0] == '-' || string[0] == '+')) {
        it++;
        min_size++;
    }
    while (it != string.end()) {
        if (*it == '.') {
            if (!decimalPoint) {
                decimalPoint = true;
            } else {
                break;
            }
        } else if (!std::isdigit(*it)) {
            break;
        }
        ++it;
    }
    return string.size() > min_size && it == string.end();
}

bool Encoder::test_integral(std::string const & string) const {
    std::string::const_iterator it = string.begin();
    int min_size = 0;
    if (string.size() > 0 && (string[0] == '-' || string[0] == '+')) {
        it++;
        min_size++;
    }
    while (it != string.end()) {
        if (!std::isdigit(*it)) { break; }
        ++it;
    }
    return string.size() > min_size && it == string.end();
}

int Encoder::limit_precision(int number) const {
    if (number == 0) { return number; }
    return (int)limit_precision((float)number);
};

float Encoder::limit_precision(float number) const {
    if (number == 0.0) { return number; }
    float sign = number >= 0.0 ? 1.0 : -1.0;
    float magnitude = abs(number);
    unsigned int precision = this -> precision;
    float max = pow(10.0, precision); 
    float min = pow(10.0, precision - 1);
    int k = 0;
    while (magnitude >= max) {
        magnitude /= 10.0;
        k += 1;
    }
    while (magnitude < min) {
        magnitude *= 10.0;
        k -= 1;
    }
    magnitude = round(magnitude);
    while (k > 0) {
        magnitude *= 10.0;
        k -= 1;
    }
    while (k < 0) {
        magnitude /= 10.0;
        k += 1;
    }
    return sign * magnitude;
};

// Separates the header from the data and tokenizes each into vectors of strings
// Input: LineReader to some input source
// Output: 2D vector of strings representing the data
// Modifies: The headers field is initialized using tokens from first line of the input
//   Sets the number of features and number of samples according to the columns and rows counted
std::vector< std::vector< std::string > > Encoder::tokenize(std::istream & input_stream) {
    io::LineReader input("", input_stream);
    std::vector< std::vector< std::string > > rows;
    unsigned int line_index = 0;
	while (char * line = input.next_line()) {
        std::stringstream stream(line);
        std::string token;
        std::vector< std::string > tokens;
        while (stream.good()) {
            getline(stream, token, ',');
            tokens.push_back(token);
        }
        if (tokens.size() <= 1) { continue; }
		if (line_index == 0) {
            this -> headers = tokens;
            this -> number_of_features = tokens.size();
		} else {
			rows.push_back(tokens);
            this -> number_of_samples += 1;
		}
        ++line_index;
	}
    return rows;
}

// Analyzes the content of each column and infer the type, cardinality, optionality, and observable values
// Input: 2D vector of strings representing the tokenized data
// Modifies: initializes the type, cardinality, optionality, values vector with one entry for each feature
void Encoder::parse(std::vector< std::vector< std::string > > const & rows) {
    const unsigned int n = this -> number_of_samples;
    const unsigned int m = this -> number_of_features;

    std::vector< std::set< std::string > > values(m);
    std::vector< unsigned int > cardinality(m, 0);
    std::vector< bool > optional(m, false);
    std::vector< std::string > type(m, "Undefined");

    // Infer the type of each feature as either Numerical or Categorical
    // Infer whether each feature is optional (or possibly Null)
    // Determine the set of uniquely observed values
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            std::string const & element = rows[i][j];
            if (element == "" || element == "NULL" || element == "null" || element == "Null" || element == "NA" || element == "na" || element == "NaN") {
                optional[j] = true;
                continue;
            }
            if ((type[j] == "Undefined" || type[j] == "Integral") && test_integral(element)) {
                // Element belongs to the Integral Type
                // This is the strictest base type
                type[j] = "Integral";
            } else if ((type[j] == "Undefined" || type[j] == "Integral" || type[j] == "Rational") && test_rational(element)) {
                // Element belongs to the Rational Type
                // This is the second stricted base type
                type[j] = "Rational";
            } else {
                // Element belongs to the Categorical Type
                // This is the least strict base type
                type[j] = "Categorical";
            }
            values[j].insert(element);
        }
    }

    // Special processing of numerical data, the user can choose to limit the numerical precision
    if (this -> precision > 0) {
        std::vector< std::set< std::string > > precise_values(values);
        values.clear();
        values.resize(m);
        for (unsigned int j = 0; j < m; ++j) {
            if (type[j] == "Integral") {
                for (auto iterator = precise_values[j].begin(); iterator != precise_values[j].end(); ++iterator) {
                    values[j].insert(std::to_string(limit_precision((int)atoi((* iterator).c_str()))));
                }
            } else if (type[j] == "Rational") {
                for (auto iterator = precise_values[j].begin(); iterator != precise_values[j].end(); ++iterator) {
                    values[j].insert(std::to_string(limit_precision((float)atof((* iterator).c_str()))));
                }
            } else {
                for (auto iterator = precise_values[j].begin(); iterator != precise_values[j].end(); ++iterator) {
                    values[j].insert(* iterator);
                }
            }
        }
    }

    // Distinguish special types based on cardinality
    for (unsigned int j = 0; j < m; ++j) {
        unsigned int card = values[j].size();
        cardinality[j] = card;
        if (card == 1) {
            // Redundant features don't need to be encoded
            // This is a strict subtype of Categorical
            type[j] = "Redundant"; 
        } else if (card == 2 && type[j] == "Integral" &&  !optional[j]) {
            // Binary features do not require header translation or default value encoding
            // This is a strict subtype of Categorical
            type[j] = "Binary"; 
        } else if (card <= 5 && type[j] == "Integral") {
            // Low-cardinality categorical types are often encoded as integers
            // For such features, we assume it is categorical to avoid using threshold features
            // since it would be inappropriate.
            type[j] = "Categorical";
        }
    }

    // Override Type Inference for Target Column
    // This ensures use of equality encoding instead of threshold encoding
    type[m - 1] = "Categorical";
    
    this -> values = values;
    this -> cardinality = cardinality;
    this -> optional = optional;
    this -> type = type;
}

// Initialize the codex with a set of encoding rules to convert between the original feature space to a binary feature space
// The codex takes on the following structure is a vector of rule lists, one for each original feature:
// codex =  < rule_list >
// Each rule list is a vector of encoding rules, one for each binary faeture extracted from the value set
// rule_list = < encoding_rule >
// Each encoding rule is a vector of strings, providing the data type, the relational operator, and a reference value:
// encoding_rule = < type, relation, reference >
//   Type is a string representing the inferred type of this feature, which is used for casting
//   Relation is a string representing a relational operator
//   Reference is a string representing a reference value that relates to the observed feature
// Note that all values are in string representation since we don't know the type at compile time
// This means type has to be inferred from the input string and casted whenever necessary
// Modifies: The resulting codex is stored as a member variable

// TODO: Describe Codex Indexing Structure
void Encoder::build(void) {
    const unsigned int n = this -> number_of_samples;
    const unsigned int m = this -> number_of_features;
    std::vector< std::set< std::string > > const & values = this -> values;
    std::vector< unsigned int > const & cardinality = this -> cardinality;
    std::vector< bool > const & optional = this -> optional;
    std::vector< std::string > const & type = this -> type;

    std::vector< std::vector< std::vector< std::string > > > codex(m);

    // Encoding Entry Schema < Type, Relation, Reference >
    for (unsigned int j = 0; j < m; ++j) {
        std::set< std::string > const & value_set = values[j];
        if (type[j] == "Redundant") {
            continue; // Empty Encoding List
        } else if (type[j] == "Binary") {
            auto it = value_set.begin();
            if (j < m - 1) { ++it; }
            for (; it != value_set.end(); ++it) {
                std::string const & value = * it;
                std::vector< std::string > rule { type[j], "==", value };
                codex[j].push_back(rule);
            }
        } else if (type[j] == "Categorical") {
            for (auto it = value_set.begin(); it != value_set.end(); ++it) {
                std::string const & value = * it;
                std::vector< std::string > rule { type[j], "==", value };
                codex[j].push_back(rule);
            }
        } else if (type[j] == "Integral") {
            for (auto it = value_set.begin(); it != value_set.end(); ++it) {
                int value = atoi((* it).c_str());
                std::vector< std::string > rule { type[j], ">=", std::to_string(value) };
                codex[j].push_back(rule);
            }
        } else if (type[j] == "Rational") {
            std::set< float > parsed_value_set;
            for (auto it = value_set.begin(); it != value_set.end(); ++it) {
                float value = atof((* it).c_str());
                parsed_value_set.insert(value);
            }
            float base_value = * parsed_value_set.begin();
            auto it = parsed_value_set.begin();
            for (++it; it != parsed_value_set.end(); ++it) {
                float value = * it;
                float threshold = 0.5 * (value + base_value);
                base_value = value;
                std::vector< std::string > rule { type[j], ">=", std::to_string(threshold) };
                codex[j].push_back(rule);
            }
        }
    }
    this -> codex = codex;

    unsigned int number_of_binary_features = 0;
    for (unsigned int j = 0; j < m; ++j) {
        number_of_binary_features += codex[j].size();
    }
    this -> number_of_binary_features = number_of_binary_features;

    // Display the result of type inference and codex building
    if (this -> verbose) {
        for (unsigned int j = 0; j < m; ++j) {
            std::cout << "Feature Index: " << j << ", Feature Name: " << this -> headers[j] << std::endl;
            std::cout << "  Inferred Type: " << type[j];
            std::cout << ", Empirical Cardinality: " << cardinality[j];
            std::cout << ", Optionality: " << optional[j];
            std::cout << ", Number of Encoding Rules: " << codex[j].size() << std::endl;
        }
        std::cout << "Original Dataset Dimension: " << n << " x " << m << std::endl;
        std::cout << "Binary Dataset Dimension: " << n << " x " << number_of_binary_features << std::endl;
    }
}

// Converts a 2D vector of string tokens into a vector of bitstrings
// The vector of bitstrings is a binary encoding of the 2D vector
// It is expected that the number of features will increase due to one-hot encoding.
// Input: 2D vector of string tokens representing the original data set
// Output: vector of bitstrings representing the encoded binary data set
std::vector< bitset > Encoder::encode(std::vector< std::vector< std::string > > const & rows) const {
    const unsigned int m = this -> number_of_features;
    const unsigned int n = this -> number_of_samples;
    const unsigned int o = this -> number_of_binary_features;
    std::vector< std::vector< std::vector< std::string > > > const & codex = this -> codex;
    std::vector< bitset > binary_rows;
    for (unsigned int i = 0; i < n; ++i) {
        bitset binary_row(o);
        unsigned int offset = 0;
        for (unsigned int j = 0; j < m; ++j) {
            std::vector< std::vector< std::string > > const & rules = codex[j];
            for (unsigned int k = 0; k < rules.size(); ++k) {
                std::string type = rules[k][0];
                std::string relation = rules[k][1];
                std::string reference = rules[k][2];
                if (type == "Integral") {
                    binary_row[offset + k] = (int) (atoi(rows[i][j].c_str()) >= atoi(reference.c_str()));
                } else if (type == "Rational") {
                    binary_row[offset + k] = (int) (atof(rows[i][j].c_str()) >= atof(reference.c_str()));
                } else {
                    binary_row[offset + k] = (int) (rows[i][j] == reference);
                }
            }
            offset += rules.size();
        }
        binary_rows.push_back(binary_row);
    }
    return binary_rows;
}

std::tuple< unsigned int, unsigned int > Encoder::decode(unsigned int j) const {
    const unsigned int m = this -> number_of_features;
    std::vector< std::vector< std::vector< std::string > > > const & codex = this -> codex;
    unsigned int offset = 0;
    for (unsigned int k = 0; k < m; ++k) {
        std::vector< std::vector< std::string > > const & rules = codex[k];
        if (offset + rules.size() <= j) { 
            offset += rules.size();
            continue;
        } else {
            return std::tuple< unsigned int, unsigned int >(k, j - offset);
        }
    }
    throw "No matching decoder found";
}
std::string Encoder::header(void) const {
    return this -> headers[this -> headers.size()-1];
}

std::string Encoder::header(unsigned int j) const {
    return this -> headers[j];
}

std::vector< std::string > Encoder::decoder(void) const {
    return codex[codex.size()-1][0];
}

std::vector< std::string > Encoder::decoder(unsigned int j, unsigned int offset) const {
    std::vector< std::vector< std::vector< std::string > > > const & codex = this -> codex;
    return codex[j][offset];
}

std::string Encoder::label(unsigned int k) const {
    std::vector< std::vector< std::vector< std::string > > > const & codex = this -> codex;
    return codex[codex.size()-1][k][2];
}


std::vector< std::vector< std::string > > const & Encoder::read_rows(void) const {
    return this -> rows;
}
std::vector< bitset > const & Encoder::read_binary_rows(void) const {
    return this -> binary_rows;
}

std::tuple< unsigned int, unsigned int > Encoder::split(void) const {
    const unsigned int m = this -> number_of_features;
    const unsigned int k = this -> number_of_binary_features;
    std::vector< std::vector< std::vector< std::string > > > const & codex = this -> codex;
    unsigned int number_of_target_features = codex[m-1].size();
    return std::tuple< unsigned int, unsigned int >(k - number_of_target_features, number_of_target_features);
}
