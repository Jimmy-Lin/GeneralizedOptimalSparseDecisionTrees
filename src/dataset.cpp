#include "dataset.hpp"
#include "state.hpp"

Dataset::Dataset(void) {}
Dataset::~Dataset(void) {}

Dataset::Dataset(std::istream & data_source) { load(data_source); }

// Loads the binary-encoded data set into precomputed form:
// Step 1: Build bitmasks for each column and row of the dataset, allowing fast parallel operations
// Step 2: Build the cost matrix. Either from values in an input file or a specified mode.
// Step 3: Compute columnar aggregations of the cost matrix to speed up some calculations from K^2 to K
// Step 4: Data set shape is stored
//   The overall shape of the data set is stored for indexing later
void Dataset::load(std::istream & data_source) {
    // Step 1: Construct all rows, features, and targets in binary form
    construct_bitmasks(data_source);

    // Step 2: Initialize the cost matrix
    construct_cost_matrix();

    // Step 3: Build the majority and minority costs based on the cost matrix
    aggregate_cost_matrix();

    // Step 4: Build the majority bitmask indicating whether a point is in the majority group
    construct_majority();
    
    if (Configuration::verbose) {
        std::cout << "Dataset Dimensions: " << height() << " x " << width() << " x " << depth() << std::endl;
    }
    return;
}

void Dataset::clear(void) {
    this -> features.clear();
    this -> targets.clear();
    this -> rows.clear();
    this -> feature_rows.clear();
    this -> target_rows.clear();
    this -> costs.clear();
    this -> match_costs.clear();
    this -> mismatch_costs.clear();
    this -> max_costs.clear();
    this -> min_costs.clear();
    this -> diff_costs.clear();
    this -> majority = Bitmask();
}

void Dataset::construct_bitmasks(std::istream & data_source) {
    this -> encoder = Encoder(data_source);
    std::vector< Bitmask > rows = this -> encoder.read_binary_rows();
    unsigned int number_of_samples = this -> encoder.samples(); // Number of samples in the dataset
    unsigned int number_of_rows = 0; // Number of samples after compressions
    unsigned int number_of_binary_features = this -> encoder.binary_features(); // Number of source features
    unsigned int number_of_binary_targets = this -> encoder.binary_targets(); // Number of target features
    this -> _size = number_of_samples;

    this -> rows = this -> encoder.read_binary_rows();

    this -> features.resize(number_of_binary_features, number_of_samples);
    this -> feature_rows.resize(number_of_samples, number_of_binary_features);
    this -> targets.resize(number_of_binary_targets, number_of_samples);
    this -> target_rows.resize(number_of_samples, number_of_binary_targets);

    for (unsigned int i = 0; i < number_of_samples; ++i) {
        for (unsigned int j = 0; j < number_of_binary_features; ++j) {
            this -> features[j].set(i, bool(rows[i][j]));
            this -> feature_rows[i].set(j, bool(rows[i][j]));
        }
        for (unsigned int j = 0; j < number_of_binary_targets; ++j) {
            this -> targets[j].set(i, bool(rows[i][number_of_binary_features + j]));
            this -> target_rows[i].set(j, bool(rows[i][number_of_binary_features + j]));
        }
    }
    this -> shape = std::tuple< int, int, int >(this -> rows.size(), this -> features.size(), this -> targets.size());
};

void Dataset::construct_cost_matrix(void) {
    this -> costs.resize(depth(), std::vector< float >(depth(), 0.0));
    if (Configuration::costs != "") { // Customized cost matrix
        std::ifstream input_stream(Configuration::costs);
        parse_cost_matrix(input_stream);
    } else if (Configuration::balance) { // Class-balancing cost matrix
        for (unsigned int i = 0; i < depth(); ++i) {
            for (unsigned int j = 0; j < depth(); ++j) {
                if (i == j) { this -> costs[i][j] = 0.0; continue; }
                this -> costs[i][j] = 1.0 / (float)(depth() * this -> targets[j].count());
            }
        }
    } else { // Default cost matrix
        for (unsigned int i = 0; i < depth(); ++i) {
            for (unsigned int j = 0; j < depth(); ++j) {
                if (i == j) { this -> costs[i][j] = 0.0; continue; }
                this -> costs[i][j] = 1.0 / (float)(height());
            }
        }
    }
}

void Dataset::parse_cost_matrix(std::istream & input_stream) {
    // Parse given cost matrix
    io::LineReader input("", input_stream);
    unsigned int line_index = 0;
    std::unordered_map< std::string, unsigned int > reference_to_decoded;
    std::vector< std::vector< float > > table;
    while (char * line = input.next_line()) {
        std::stringstream stream(line);
        std::string token;
        std::vector< std::string > row;
        std::vector< float > parsed_row;
        while (stream.good()) {
            getline(stream, token, ',');
            row.emplace_back(token);
        }
        if (row.empty()) { continue; }
        if (line_index == 0) {
            for (unsigned int j = 0; j < row.size(); ++j) { reference_to_decoded[row[j]] = j; }
        } else {
            for (unsigned int j = 0; j < row.size(); ++j) { parsed_row.emplace_back(atof(row[j].c_str())); }
            table.emplace_back(parsed_row);
        }
        ++line_index;
    }

    std::vector< std::string > encoded_to_reference;
    for (unsigned int j = 0; j < depth(); ++j) {
        std::string type, relation, reference;
        encoder.encoding(width() + j, type, relation, reference);
        encoded_to_reference.emplace_back(reference);
    }

    if (table.size() == 1) {
        for (unsigned int i = 0; i < depth(); ++i) {
            for (unsigned int j = 0; j < depth(); ++j) {
                if (i == j) { this -> costs[i][j] = 0.0; continue; }
                if (reference_to_decoded.find(encoded_to_reference[j]) == reference_to_decoded.end()) {
                    std::cout << "No cost specified for class = " << encoded_to_reference[j] << std::endl;
                    exit(1);
                }
                unsigned int _i = 0;
                unsigned int _j = reference_to_decoded[encoded_to_reference[j]];
                this -> costs[i][j] = table[_i][_j];
            }
        }
    } else {
        for (unsigned int i = 0; i < depth(); ++i) {
            for (unsigned int j = 0; j < depth(); ++j) {
                if (reference_to_decoded.find(encoded_to_reference[i]) == reference_to_decoded.end() || reference_to_decoded.find(encoded_to_reference[j]) == reference_to_decoded.end()) {
                    std::cout << "No cost specified for prediction = " << encoded_to_reference[i] << ", class = " << encoded_to_reference[j] << std::endl;
                    exit(1);
                }
                unsigned int _i = reference_to_decoded[encoded_to_reference[i]];
                unsigned int _j = reference_to_decoded[encoded_to_reference[j]];
                this -> costs[i][j] = table[_i][_j];
            }
        }
    }
};

void Dataset::aggregate_cost_matrix(void) {
    this -> match_costs.resize(depth(), 0.0);
    this -> mismatch_costs.resize(depth(), std::numeric_limits<float>::max());
    this -> max_costs.resize(depth(), -std::numeric_limits<float>::max());
    this -> min_costs.resize(depth(), std::numeric_limits<float>::max());
    this -> diff_costs.resize(depth(), std::numeric_limits<float>::max());
    for (unsigned int j = 0; j < depth(); ++j) {
        for (unsigned int i = 0; i < depth(); ++i) {
            this -> max_costs[j] = std::max(this -> max_costs[j], this -> costs[i][j]);
            this -> min_costs[j] = std::min(this -> min_costs[j], this -> costs[i][j]);
            if (i == j) { this -> match_costs[j] = this -> costs[i][j]; continue; }
            this -> mismatch_costs[j] = std::min(this -> mismatch_costs[j], this -> costs[i][j]);
        }
    }
    for (unsigned int j = 0; j < depth(); ++j) {
        this -> diff_costs[j] = this -> max_costs[j] - this -> min_costs[j] ;
    }
}

void Dataset::construct_majority(void) {
    std::vector< Bitmask > keys(height(), width());
    for (unsigned int i = 0; i < height(); ++i) {
        for (unsigned int j = 0; j < width(); ++j) {
            keys[i].set(j, bool(this -> rows[i][j]));
        }
    }

    // Step 1: Construct a map from the binary features to their distributions
    std::unordered_map< Bitmask, std::vector< float > > distributions;
    for (unsigned int i = 0; i < height(); ++i) {
        Bitmask const & key = keys.at(i);
        // Initialize the map and resize the value (of type vector) to the number of unique labels
        // This way the vector can hold the label distribution for this feature combination
        if (distributions[key].size() < depth()) { distributions[key].resize(depth(), 0.0); }
        for (unsigned int j = 0; j < depth(); ++j) {
            distributions[key][j] += (float)rows[i][width() + j];
        }
    }

    // Step 2: Construct a map from the binary features to cost minimizers
    std::unordered_map< Bitmask, unsigned int  > minimizers;
    for (auto it = distributions.begin(); it != distributions.end(); ++it) {
        Bitmask const & key = it -> first;
        std::vector< float > const & distribution = it -> second;
        float minimum = std::numeric_limits<float>::max();
        unsigned int minimizer = 0;
        for (unsigned int i = 0; i < depth(); ++i) {
            float cost = 0.0;
            for (unsigned int j = 0; j < depth(); ++j) {
                cost += this -> costs[i][j] * distribution.at(j); // Cost of predicting i when the class is j
            }
            if (cost < minimum) {
                minimum = cost;
                minimizer = i;
            }
        }
        minimizers.emplace(key, minimizer);
    }

    // Step 3: Set the bits associated with each minimizer
    this -> majority.initialize(height());
    for (unsigned int i = 0; i < height(); ++i) {
        Bitmask const & key = keys.at(i);
        unsigned int minimizer = minimizers[key];
        unsigned int label = this -> rows[i].scan(width(), true) - width();
        this -> majority.set(i, minimizer == label); // Set this bit true if the label matches this minimizer
    }
}

float Dataset::distance(Bitmask const & set, unsigned int i, unsigned int j, unsigned int id) const {
    Bitmask & buffer = State::locals[id].columns[0];
    float positive_distance = 0.0, negative_distance = 0.0;
    for (unsigned int k = 0; k < depth(); ++k) {
        buffer = this -> features[i];
        this -> features[j].bit_xor(buffer, false);
        set.bit_and(buffer);
        // this -> majority.bit_and(buffer, false);
        this -> targets[k].bit_and(buffer);
        positive_distance += this -> diff_costs[k] * buffer.count();

        buffer = this -> features[i];
        this -> features[j].bit_xor(buffer, true);
        set.bit_and(buffer);
        // this -> majority.bit_and(buffer, false);
        this -> targets[k].bit_and(buffer);
        negative_distance += this -> diff_costs[k] * buffer.count();
    }
    return std::min(positive_distance, negative_distance);
}

// @param feature_index: selects the feature on which to split
// @param positive: determines whether to provide the subset that tests positive on the feature or tests negative on the feature
// @param set: pointer to bit blocks which indicate the original set before splitting
// @modifies set: set will be modified to indicate the positive or negative subset after splitting
// @notes the set in question is an array of the type bitblock. this allows us to specify the set using a stack-allocated array
void Dataset::subset(unsigned int feature_index, bool positive, Bitmask & set) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(set, !positive);
    if (Configuration::depth_budget != 0){ set.set_depth_budget(set.get_depth_budget()-1);} //subproblems have one less depth_budget than their parent
}

void Dataset::subset(unsigned int feature_index, Bitmask & negative, Bitmask & positive) const {
    // Performs bit-wise and between feature and set with possible bit-flip if performing negative test
    this -> features[feature_index].bit_and(negative, true);
    this -> features[feature_index].bit_and(positive, false);
    if (Configuration::depth_budget != 0){
        negative.set_depth_budget(negative.get_depth_budget()-1);
        positive.set_depth_budget(positive.get_depth_budget()-1);
    } //subproblems have one less depth_budget than their parent
}

void Dataset::summary(Bitmask const & capture_set, float & info, float & potential, float & min_loss, float & guaranteed_min_loss, float & max_loss, unsigned int & target_index, unsigned int id) const {
    Bitmask & buffer = State::locals[id].columns[0];
    unsigned int * distribution; // The frequencies of each class
    distribution = (unsigned int *) alloca(sizeof(unsigned int) * depth());
    for (int j = depth(); --j >= 0;) {
        buffer = capture_set; // Set representing the captured points
        this -> targets.at(j).bit_and(buffer); // Captured points with label j
        distribution[j] = buffer.count(); // Calculate frequency
    }

    float min_cost = std::numeric_limits<float>::max();
    unsigned int cost_minimizer = 0;

    for (int i = depth(); --i >= 0;) { // Prediction index
        float cost = 0.0; // accumulator for the cost of making this prediction
        for (int j = depth(); --j >= 0;) { // Class index
            cost += this -> costs.at(i).at(j) * distribution[j]; // cost of prediction-class combination
        }
        if (cost < min_cost) { // track the prediction that minimizes cost
            min_cost = cost;
            cost_minimizer = i;
        }
    }
    float max_cost_reduction = 0.0;
    float support = (float)(capture_set.count()) / (float)(height());
    float information = 0.0;

    //calculate equivalent point loss for this capture set
    float equivalent_point_loss = 0.0;
    for (int j = depth(); --j >= 0;) { // Class index
        // maximum cost difference across predictions
        max_cost_reduction += this -> diff_costs[j] * distribution[j];

        buffer = capture_set; // Set representing the captured points
        this -> majority.bit_and(buffer, false); // Captured majority points
        this -> targets.at(j).bit_and(buffer); // Captured majority points with label j
        equivalent_point_loss += this -> match_costs[j] * buffer.count(); // Calculate frequency

        buffer = capture_set; // Set representing the captured points
        this -> majority.bit_and(buffer, true); // Captured minority points
        this -> targets.at(j).bit_and(buffer); // Captured minority points with label j
        equivalent_point_loss += this -> mismatch_costs[j] * buffer.count(); // Calculate frequency

        float prob = distribution[j];
        if (prob > 0) { information += support * prob * (log(prob) - log(support)); }
    }

    // use equivalent points as a guaranteed lowerbound, regardless of whether we are using a refence model to guess lower bounds
    // (although most implications of a guessed lower bound are acceptable, we still want the guaranteed lower bound for scoping,
    //  since we do not wish to narrow one subproblem's scope based on an overestimate for the lower bound of another subproblem)
    guaranteed_min_loss = equivalent_point_loss;
    
    // because we are using floating point calculations, we might have our guaranteed_min_loss > max_loss in cases where they should be the same
    // To avoid contradictions and maintain the invariant that guaranteed_min_loss <= max_loss, we correct for that here. 
    // (note that min_cost is the same as max_loss)
    if (guaranteed_min_loss > min_cost){
        guaranteed_min_loss = min_cost;
    }


    if (Configuration::reference_LB){
    //calculate reference model's error on this capture set, use as estimate for min_loss (possible overestimate)
        float reference_model_loss = 0.0;
        for (int j = depth(); --j >= 0;) {
            // maximum cost difference across predictions
            max_cost_reduction += this -> diff_costs[j] * distribution[j];

            buffer = capture_set; // Set representing the captured points
            this -> targets.at(j).bit_and(buffer, false); // Captured points with label j
            Reference::labels[j].bit_and(buffer); // Captured points with label j classified correctly by reference model
            reference_model_loss += this -> match_costs[j] * buffer.count(); // Calculate cost from correct classifications on j

            buffer = capture_set; // Set representing the captured points
            this -> targets.at(j).bit_and(buffer, false); // Captured points with label j
            Reference::labels[j].bit_and(buffer, true); // Captured points with label j classified incorrectly by reference model
            reference_model_loss += this -> mismatch_costs[j] * buffer.count(); // Calculate frequency  
        }
        min_loss = reference_model_loss; 
    } else {
        // when not using a reference model, we do not want min_loss to be an overestimate
        // so we set min_loss to match guaranteed_min_loss
        min_loss = guaranteed_min_loss;
    }

    potential = max_cost_reduction;
    max_loss = min_cost;
    info = information;
    target_index = cost_minimizer;
}

// Assume that data is already of the right size
void Dataset::tile(Bitmask const & capture_set, Bitmask const & feature_set, Tile & tile, std::vector< int > & order, unsigned int id) const {
    tile.content() = capture_set;
    tile.width(0);
    return;
}


unsigned int Dataset::height(void) const {
    return std::get<0>(this -> shape);
}

unsigned int Dataset::width(void) const {
    return std::get<1>(this -> shape);
}

unsigned int Dataset::depth(void) const {
    return std::get<2>(this -> shape);
}

unsigned int Dataset::size(void) const {
    return this -> _size;
}

bool Dataset::index_comparator(const std::pair< unsigned int, unsigned int > & left, const std::pair< unsigned int, unsigned int > & right) {
    return left.second < right.second;
}
