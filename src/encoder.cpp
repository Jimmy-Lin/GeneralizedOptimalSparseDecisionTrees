#include "encoder.hpp"

Encoder::Encoder(void) {}

// Construct an encoder which converts a non-binary data set into a binary-encoded dataset
Encoder::Encoder(std::istream & input) {
    std::vector< std::vector< std::string > > tokens;
    tokenize(input, tokens); // Produce string tokens from the input
    parse(tokens); // Parse the tokens to infer the type of each feature
    build(); // Build a set of encoding rules for each featuree

    // reindex has been disabled since it interferes with the binary search of the new ordinal feature bound
    // reindex(tokens); // Determine an efficient ordering of encoding rules

    encode(tokens, this -> binary_rows); // Encode the tokenized data using the encoding rules
}

// Tests whether a string should be considered an integer
// Pattern Implemented: Must conform to the following expression ^[+-]?\d+\$
bool Encoder::test_rational(std::string const & string) {
    if (string.size() == 0) { return false; }
    std::string::const_iterator it = string.begin();
    bool decimal = false;
    unsigned int min_size = 0;
    if (string.size() > 0 && (string[0] == '-' || string[0] == '+')) {
        it++;
        min_size++;
    }
    while (it != string.end()) {
        if (* it == '.') {
            if (!decimal) {
                decimal = true;
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

// Tests whether a string should be considered a floating point
// Pattern Implemented: Must conform to the following expression ^[+-]?\d+\.\d*$
bool Encoder::test_integral(std::string const & string) {
    if (string.size() == 0) { return false; }
    std::string::const_iterator it = string.begin();
    unsigned int min_size = 0;
    if (string.size() > 0 && (string[0] == '-' || string[0] == '+')) {
        it++;
        min_size++;
    }
    while (it != string.end()) {
        if (!std::isdigit(* it)) { break; }
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
    unsigned int precision = Configuration::precision_limit;
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

void Encoder::tokenize(std::istream & input_stream, std::vector< std::vector< std::string > > & rows) {
    io::LineReader input("", input_stream);
    unsigned int line_index = 0;
	while (char * line = input.next_line()) {
        std::stringstream stream(line);
        std::string token;
        std::vector< std::string > row;
        while (stream.good()) {
            getline(stream, token, ',');
            row.push_back(token);
        }
        if (row.empty()) { continue; }
		if (line_index == 0) {
            this -> headers = row;
            this -> number_of_columns = row.size();
		} else {
			rows.push_back(row);
            this -> number_of_rows += 1;
		}
        ++line_index;
	}
}

void Encoder::parse(std::vector< std::vector< std::string > > const & rows) {
    const unsigned int n = this -> number_of_rows;
    const unsigned int m = this -> number_of_columns;

    std::vector< std::set< std::string > > & values = this -> values;
    values.resize(m);
    std::vector< unsigned int > & cardinalities = this -> cardinalities;
    cardinalities.resize(m, 0);
    std::vector< bool > & optionalities = this -> optionalities;
    optionalities.resize(m, false);
    std::vector< std::string > & inferred_types = this -> inferred_types;
    inferred_types.resize(m, "Undefined");

    std::map< std::string, unsigned int > target_distribution;

    // Content-based type inference
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            std::string const & element = rows[i][j];
            std::string & inferred_type = inferred_types[j];
            if (element == "" || element == "NULL" || element == "null" || element == "Null" || element == "NA" || element == "na" || element == "NaN") {
                optionalities[j] = true;
                continue;
            }
            if ((inferred_type == "Undefined" || inferred_type == "Integral") && Encoder::test_integral(element)) {
                inferred_type = "Integral";
            } else if ((inferred_type == "Undefined" || inferred_type == "Integral" || inferred_type == "Rational") && Encoder::test_rational(element)) {
                inferred_type = "Rational";
            } else {
                inferred_type = "Categorical";
            }
            values[j].insert(element);

            if (j == m - 1) {
                target_distribution[element] += 1;
            }
        }
    }

    // Special processing of numerical data, the user can choose to limit the numerical precision
    if (Configuration::precision_limit > 0) { limit_precision(values); }

    // Further type specification based on column statistics
    for (unsigned int j = 0; j < m; ++j) {
        std::string & inferred_type = inferred_types[j];
        unsigned int card = values[j].size();
        cardinalities[j] = card;
        if (card <= 1) {
            // Redundant features don't need to be encoded
            inferred_type = "Redundant"; 
        } else if (card == 2 && !optionalities[j]) {
            // Binary features do not require header translation or default value encoding
            inferred_type = "Binary"; 
        }
    }

    // Override Type Inference for Target Column
    // This ensures use of equality encoding instead of threshold encoding
    inferred_types[m - 1] = "Categorical";
}

void Encoder::limit_precision(std::vector< std::set< std::string > > & values) const {
    const unsigned int m = this -> number_of_columns;
    std::vector< std::string > const & inferred_types = this -> inferred_types;
    std::vector< std::set< std::string > > precise_values(values);
    values.clear();
    values.resize(m);
    for (unsigned int j = 0; j < m; ++j) {
        std::string const & inferred_type = inferred_types[j];
        if (inferred_type == "Integral") {
            for (auto iterator = precise_values[j].begin(); iterator != precise_values[j].end(); ++iterator) {
                values[j].insert(std::to_string(limit_precision((int)atoi((* iterator).c_str()))));
            }
        } else if (inferred_type == "Rational") {
            for (auto iterator = precise_values[j].begin(); iterator != precise_values[j].end(); ++iterator) {
                values[j].insert(std::to_string(limit_precision((float)atof((* iterator).c_str()))));
            }
        } else {
            for (auto iterator = precise_values[j].begin(); iterator != precise_values[j].end(); ++iterator) {
                values[j].insert(* iterator);
            }
        }
    }
    return;
}


// Initialize the codex with a set of encoding rules to convert between the original feature space to a binary feature space
// The codex takes on the following structure is a vector of rule lists, one for each original feature:
//   codex =  < rule_list >
// Each rule list is a vector of encoding rules, one for each binary faeture extracted from the value set
//   rule_list = < source index, encoding_rule >
// Each source index is the index of the feature in the original feature space before conversion to binary feature space
// Each encoding rule is a vector of strings, providing the data type, the relational operator, and a reference value:
//   encoding_rule = < type, relation, reference >
//     type: is a string representing the inferred type of this feature, which is used for casting
//     relation: is a string representing a relational operator
//     reference: is a string representing a reference value that relates to the observed feature
// Note that all values are in string representation since we don't know the type at compile time

void Encoder::build(void) {
    const unsigned int n = this -> number_of_rows;
    const unsigned int m = this -> number_of_columns;
    std::vector< std::set< std::string > > const & values = this -> values;
    std::vector< unsigned int > const & cardinalities = this -> cardinalities;
    std::vector< bool > const & optionalities = this -> optionalities;
    std::vector< std::string > const & inferred_types = this -> inferred_types;

    std::vector< std::pair< unsigned int, std::vector< std::string > > > codex;

    // Encoding Entry Schema < Type, Relation, Reference >
    for (unsigned int j = 0; j < m; ++j) {
        std::string const & inferred_type = inferred_types[j];
        std::set< std::string > const & value_set = values[j];
        unsigned int initial_size = codex.size();
        if (inferred_type == "Redundant") {
            continue; // Empty Encoding List
        } else if (inferred_type == "Binary") {
            auto it = value_set.begin();
            if (j < m - 1) { ++it; }
            for (; it != value_set.end(); ++it) {
                std::string const & value = * it;
                std::vector< std::string > rule { inferred_type, "==", value };
                codex.push_back(std::make_pair(j, rule));
            }
        } else if (inferred_type == "Categorical" || inferred_type == "Enumerable") {
            for (auto it = value_set.begin(); it != value_set.end(); ++it) {
                std::string const & value = * it;
                std::vector< std::string > rule { inferred_type, "==", value };
                codex.push_back(std::make_pair(j, rule));
            }
        } else if (inferred_type == "Integral") {
            unsigned int start, finish;
            start = codex.size();
            for (auto it = value_set.begin(); it != value_set.end(); ++it) {
                int value = atoi((* it).c_str());
                std::vector< std::string > rule { inferred_type, ">=", std::to_string(value) };
                codex.push_back(std::make_pair(j, rule));
            }
            finish = codex.size();
            this -> boundaries.emplace_back(start, finish);
        } else if (inferred_type == "Rational") {
            unsigned int start, finish;
            start = codex.size();
            // Create an ordered set to sort the thresholds
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
                std::vector< std::string > rule { inferred_type, ">=", std::to_string(threshold) };
                codex.push_back(std::make_pair(j, rule));
            }
            finish = codex.size();
            this -> boundaries.emplace_back(start, finish);
        }
        unsigned int final_size = codex.size();
        if (j == m - 1) {
            this -> number_of_binary_targets = final_size - initial_size;
        }
    }
    this -> codex = codex;

    unsigned int number_of_binary_columns = codex.size();
    this -> number_of_binary_columns = number_of_binary_columns;

    // Display the result of type inference and codex building
    if (Configuration::verbose) {
        for (unsigned int j = 0; j < m; ++j) {
            std::cout << "Feature Index: " << j << ", Feature Name: " << this -> headers[j] << std::endl;
            std::cout << "  Inferred Type: " << inferred_types[j];
            std::cout << ", Empirical Cardinality: " << cardinalities[j];
            std::cout << ", Optionality: " << optionalities[j] << std::endl;
        }
        std::cout << "Original Dataset Dimension: " << n << " x " << m << std::endl;
        std::cout << "Binary Dataset Dimension: " << n << " x " << number_of_binary_columns << std::endl;
    }
}

void Encoder::reindex(std::vector< std::vector< std::string > > const & rows) {
    const unsigned int m = this -> number_of_columns;
    const unsigned int n = this -> number_of_rows;
    const unsigned int o = this -> number_of_binary_columns;
    const unsigned int p = this -> number_of_binary_targets;
    std::vector< std::pair< unsigned int, std::vector< std::string > > > const & codex = this -> codex;
    std::vector< std::pair< unsigned int, std::vector< std::string > > > ordered_codex;

    std::set< std::pair< float, unsigned int > > entropies;
    // Compute entropies conditioned on a single split on each binary feature
    for (unsigned int k = 0; k < (o - p); ++k) {
        std::pair< unsigned int, std::vector< std::string > > rule = codex.at(k);
        std::string inferred_type = rule.second[0];
        std::string reference = rule.second[2];

        std::vector< float > negative(p);
        std::vector< float > positive(p);
        float negative_total = 0.0;
        float positive_total = 0.0;

        for (unsigned int i = 0; i < n; ++i) {
            bool value;
            if (inferred_type == "Integral") {
                value = (atoi(rows[i][rule.first].c_str()) >= atoi(reference.c_str()));
            } else if (inferred_type == "Rational") {
                value = (atof(rows[i][rule.first].c_str()) >= atof(reference.c_str()));
            } else {
                value = (rows[i][rule.first] == reference);
            }

            if (value) {
                positive_total += 1.0 / n;
            } else  {
                negative_total += 1.0 / n;
            }

            std::string label = rows[i][m - 1];
            for (unsigned int l = 0; l < p; l++) {
                std::pair< unsigned int, std::vector< std::string > > target = codex.at(o - p + l);
                std::string target_inferred_type = target.second[0];
                std::string target_reference = target.second[2];
                bool target_value;

                if (inferred_type == "Integral") {
                    target_value = (atoi(rows[i][target.first].c_str()) >= atoi(target_reference.c_str()));
                } else if (inferred_type == "Rational") {
                    target_value = (atof(rows[i][target.first].c_str()) >= atof(target_reference.c_str()));
                } else {
                    target_value = (rows[i][target.first] == reference);
                }

                if (value) {
                    positive[l] += (int)(target_value);
                } else {
                    negative[l] += (int)(target_value);
                }
            }            
        }

        float negative_entropy = 0.0;
        float positive_entropy = 0.0;
        for (unsigned int l = 0; l < p;  ++l) {
            if (positive[l] > 0.0) {
                positive_entropy += positive_total * positive[l] * (log(positive[l]) - log(positive_total));
            }
            if (negative[l] > 0.0) {
                negative_entropy += negative_total * negative[l] * (log(negative[l]) - log(negative_total));
            }
        }
        float entropy = negative_entropy + positive_entropy;
        entropies.insert(std::make_pair(entropy, k));
    }

    // Order a new codex based on entropy change
    for (auto iterator = entropies.begin(); iterator != entropies.end(); ++iterator) {
        std::pair< float, unsigned int > entry = * iterator;
        ordered_codex.push_back(codex[entry.second]);
    }
    for (unsigned int l = 0; l < p; l++) {
        ordered_codex.push_back(codex[o - p + l]);
    }
    // Replace existing codex with new codex
    this -> codex = ordered_codex;
    return;
}

void Encoder::encode(std::vector< std::vector< std::string > > const & rows, std::vector< Bitmask > & binary_rows) const {
    const unsigned int n = this -> number_of_rows;
    const unsigned int o = this -> number_of_binary_columns;
    std::vector< std::pair< unsigned int, std::vector< std::string > > > const & codex = this -> codex;
    for (unsigned int i = 0; i < n; ++i) {
        Bitmask binary_row(o);
        for (unsigned int k = 0; k < o; ++k) {
            std::pair< unsigned int, std::vector< std::string > > rule = codex.at(k);
            unsigned int j = rule.first;
            std::string inferred_type = rule.second[0];
            std::string relation = rule.second[1];
            std::string reference = rule.second[2];
            bool value;
            if (inferred_type == "Integral") {
                value = (atoi(rows[i][j].c_str()) >= atoi(reference.c_str()));
            } else if (inferred_type == "Rational") {
                value = (atof(rows[i][j].c_str()) >= atof(reference.c_str()));
            } else {
                value = (rows[i][j] == reference);
            }
            binary_row.set(k, value);
        }
        binary_rows.emplace_back(binary_row);
    }
    return;
}

void Encoder::decode(unsigned int encoded_column_index, unsigned int * decoded_column_index) const {
    std::vector< std::pair< unsigned int, std::vector< std::string > > > const & codex = this -> codex;
    std::pair< unsigned int, std::vector< std::string > > rule = codex.at(encoded_column_index);
    * decoded_column_index = rule.first;
    return;
}

void Encoder::header(std::string & name) const {
    name = this -> headers[this -> number_of_columns - 1];
}

void Encoder::header(unsigned int decoded_column_index, std::string & name) const {
    name = this -> headers[decoded_column_index];
}

// float Encoder::weight(unsigned int index) const {
//     if (index >= this -> weights.size()) {
//         return 1.0 / this -> number_of_rows;
//     } else {
//         return this -> weights[index];
//     }
// }

void Encoder::find_encoding(unsigned int decoded_column_index, std::string const & reference, unsigned int * encoded_column_index) const {
    std::vector< std::pair< unsigned int, std::vector< std::string > > > const & codex = this -> codex;
    
    unsigned int n = codex.size();

    float distance = std::numeric_limits<float>::max();
    unsigned int index = -1;

    for (unsigned int i = 0; i < n; ++i) {
        std::pair< unsigned int, std::vector< std::string > > const & rule = codex.at(i);
        if (rule.first != decoded_column_index) { continue; }
        std::string const & inferred_type = rule.second.at(0);
        float local_distance;
        if (inferred_type == "Integral") {
            local_distance = std::abs(atoi(reference.c_str()) - atoi(rule.second.at(2).c_str()));
        } else if (inferred_type == "Rational") {
            local_distance = std::abs(atof(reference.c_str()) - atof(rule.second.at(2).c_str()));
        } else {
            local_distance = (reference == rule.second.at(2)) ? 0.0 : 1.0;
        }
        if (local_distance < distance) {
            distance = local_distance;
            index = i;
        }
    }
    * encoded_column_index = index;
    return;
}

void Encoder::encoding(unsigned int encoded_column_index, std::string & type, std::string & relation, std::string & reference) const {
    unsigned int decoded_column_index;
    decode(encoded_column_index, & decoded_column_index);
    std::vector< std::string > rule = this -> codex[encoded_column_index].second;
    type = rule[0];
    relation = rule[1];
    reference = rule[2];
}

void Encoder::target_value(unsigned int value_index, std::string & value) const {
    value = this -> codex[this -> number_of_binary_columns - this -> number_of_binary_targets + value_index].second[2];
}

void Encoder::target_type(std::string & value) const {
    value = this -> codex[this -> number_of_binary_columns - this -> number_of_binary_targets].second[0];
}

std::vector< Bitmask > const & Encoder::read_binary_rows(void) const { return this -> binary_rows; }

unsigned int Encoder::samples(void) const { return this -> number_of_rows; }

unsigned int Encoder::features(void) const { return this -> number_of_columns - 1; }

unsigned int Encoder::targets(void) const { return 1; }

unsigned int Encoder::binary_features(void) const {
    return this -> number_of_binary_columns - this -> number_of_binary_targets;
}

unsigned int Encoder::binary_targets(void) const {
    return this -> number_of_binary_targets;
}