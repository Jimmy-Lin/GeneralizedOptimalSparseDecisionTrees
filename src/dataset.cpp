#include "dataset.hpp"

Dataset::Dataset(void) {}

Dataset::Dataset(std::istream & data_source, unsigned int precision, float regularization, bool verbose) : verbose(verbose) {
    this -> encoder = Encoder(data_source, precision, verbose);
    load(this -> encoder.read_binary_rows(), this -> encoder.split(), regularization);
}

Dataset::~Dataset(void) {}

void Dataset::load(std::vector< bitset > const & rows, std::tuple< unsigned int, unsigned int > const & split, float const regularization) {
    this -> _regularization = regularization; 
    unsigned int k = std::get<1>(split); // Number of target features
    unsigned int size = rows.size(); // Number of actual samples
    unsigned int width = std::get<0>(split); // Number of source features
    unsigned int compressed_size = 0; // Number of samples after compressions
    unsigned int depth = 0; // Number of target values
    this -> _size = size;

    // for (unsigned int i = 0; i < size; ++i) { // Sanity  Check on Binary Conversion
    //     std::cout << Bitmask(rows[i]).to_string(true) << std::endl;
    // }

    // Create a sorted map of unique binary labels
    std::map < bitset, int > labels;
    for (unsigned int i = 0; i < size; ++i) {
        bitset label_bitset(k); // Select the columns associated with the labels
        for (unsigned int j = 0; j < k; ++j) { label_bitset[j] = rows[i][width + j]; }
        labels[label_bitset] = 0; // Initialize entry in the map, the value will later be set to the index of the sorted set
    }
    depth = labels.size();

    // Convert to a reverse index from binary label to position in the ordered map
    {
        unsigned int label_index = 0;
        for (auto iterator = labels.begin(); iterator != labels.end(); ++iterator) {
            iterator -> second = label_index;
            label_index++;
        }
    }

    // Create an ordered map of unique binary features to their conditional distribution of labels
    // label distribution buckets are ordered by their position in the previously constructed reverse index
    std::map< bitset, std::vector< float > > feature_map;
    for (unsigned int i = 0; i < size; ++i) {
        bitset feature_bitset(width); // Select the columns associated with the features
        for (unsigned int j = 0; j < width; ++j) { feature_bitset[j] = rows[i][j]; }
        // Initialize the map and resize the value (of type vector) to the number of unique labels
        // This way the vector can hold the label distribution for this feature combination
        if (feature_map[feature_bitset].size() < depth) { feature_map[feature_bitset].resize(depth); }
        bitset label_bitset(k); // Select the label associated with this instance of features (and convert from JSON array to binary bitset)
        for (unsigned int j = 0; j < k; ++j) { label_bitset[j] = rows[i][width + j]; }
        // Increase the frequency of this label in the distribution associated with this feature conbination
        // Here is where we could change the weight to be based on class size instead of overall dataset size
        feature_map[feature_bitset][labels[label_bitset]] += (1.0 / size);
    }
    compressed_size = feature_map.size();

    {   // Construct segment trees (and other methods) to speed up summations
        std::vector< std::vector< float > > label_indices(depth); // Compute a segment tree for each label
        std::vector< float > loss_index; // Compute a segment tree for the non-majority to quickly compute lowerbounds

        std::vector< float > total_support;
        // Convert the feature_map map into a vector of tuples
        for (auto iterator = feature_map.begin(); iterator != feature_map.end(); ++iterator) {
            Bitmask const & row = Bitmask(iterator -> first);
            std::vector< float > const & distribution = iterator -> second;
            this -> rows.emplace_back(row, distribution);


            float local_maximum = * (distribution.begin());
            float local_sum = 0;
            for (unsigned int j = 0; j < depth; ++j) {
                float element = distribution[j];
                label_indices[j].push_back(element);
                local_maximum = std::max(element, local_maximum);
                local_sum += element;
            }
            float const non_majority = local_sum - local_maximum;
            total_support.emplace_back(local_sum);
            loss_index.emplace_back(non_majority);
        }
        for (unsigned int j = 0; j < depth; ++j) { this -> label_indices.emplace_back(label_indices[j]); }
//        for (unsigned int j = 0; j < depth; ++j) { this -> total_support.emplace_back(total_support[j]); }
        this -> loss_index = Index(loss_index);
        this -> total_support = Index(total_support);
    }

    {   // Create column vectors for quick set partitioning using bitwise operations
        std::vector< bitset > columns(width, bitset(compressed_size));
        for (unsigned int i = 0; i < compressed_size; ++i) {
            for (unsigned int j = 0; j < width; ++j) {
                columns[j][i] = std::get<0>(this -> rows[i])[j];
            }
        } // Converts bitset to bitmask
        for (unsigned int j = 0; j < width; ++j) { this -> columns.emplace_back(columns[j]); }
    }

    {
        this -> distance_map = std::map< std::pair< int, int >, Index >();
        for (unsigned int i = 0; i < (this -> columns).size(); i++) {
            for (unsigned int j = i+1; j < (this -> columns).size(); j++) {
                Bitmask cross = (this -> columns[i])^(this -> columns[j]);
                Index filtered = (this -> total_support).zeroOut(cross);
                std::pair<int, int> key = std::make_pair(i, j);
                this -> distance_map.insert(std::pair< std::pair< int, int>, Index >(key, filtered));
            }
        }
    }

    {   // Convert the labels map into a vector of tuples 
        for (auto iterator = labels.begin(); iterator != labels.end(); ++iterator) {
            this -> labels.emplace_back(Bitmask(iterator -> first), iterator -> second);
        }
    }

    this -> shape = std::tuple< int, int, int >(this -> rows.size(), this -> columns.size(), this -> labels.size());

    if (this -> verbose) {
        std::cout << "Original Dataset Size: " << size << ", Compressed Dataset Size: " << compressed_size << std::endl;
    }
    return;
}

float Dataset::distance(Bitmask const & indicator, int feature_a, int feature_b) {
    // retrieve a/b xor from total support then apply indicator, return
    std::pair<int, int> key;
    if (feature_a < feature_b) {
        key = std::make_pair(feature_a, feature_b);
    } else {
        key = std::make_pair(feature_b, feature_a);
    }
    Index stored = (this -> distance_map)[key];
    return stored.sum(indicator);
}

void Dataset::initialize_similarity_index(float similarity_threshold) {
    unsigned int m = width();

    for (unsigned int j = 0; j < m; ++j) {
        auto partition_j = partition(j);
        Bitmask const & negative_j = * partition_j.begin();
        Bitmask const & positive_j = * (++partition_j.begin());
        for (unsigned int k = 0; k < m; ++k) {
            if (j == k) { continue; }
            auto partition_k = partition(k);
            Bitmask const & negative_k = * partition_k.begin();
            Bitmask const & positive_k = * (++partition_k.begin());
            // Compute the two possible difference sets
            Bitmask const & alpha_mask = (negative_j & positive_k) | (positive_j & negative_k);
            Bitmask const & beta_mask = (negative_j & negative_k) | (positive_j & positive_k);
            // Computes the smaller of the total weight of each mask
            // This forms the distance "omega" in the simiar support bound, with awareness of weights modified by preprocessing
            float distance = std::min(
                support(alpha_mask),
                support(beta_mask)
            );
            // Encodes the maximal distance between j and k
            if (distance < similarity_threshold) {
                this -> similarity_index[j][k] = distance;
            }
        }
    }
}

// Return Prediction, Loss
std::tuple< std::string, float > Dataset::leaf(void) const {
    float max = sum(0);
    unsigned int label_index = 0;
    for (unsigned int j = 1; j < depth();  ++j) {
        float prob = sum(j);
        if (prob > max) {
            max = prob;
            label_index = j;
        }
    }
    return std::tuple< std::string, float >(encoder.label(label_index), support() - max);
}

std::tuple< std::string, float > Dataset::leaf(Bitmask const & indicator) const {
    float max = sum(indicator, 0);
    unsigned int label_index = 0;
    for (unsigned int j = 1; j < depth();  ++j) {
        float prob = sum(indicator, j);
        if (prob > max) {
            max = prob;
            label_index = j;
        }
    }
    return std::tuple< std::string, float >(encoder.label(label_index), support(indicator) - max);
}

float Dataset::loss_lowerbound(void) const {
    return this -> loss_index.sum();
}
float Dataset::loss_lowerbound(Bitmask const & indicator) const {
    if (indicator.to_string() == "111111111111111111111111111111111111111110000000000000000000000000000000000000000000000000000000000000000000000000000000000") {

    }
    return this->loss_index.sum(indicator);
}

float Dataset::support(void) const {
    float total = 0;
    for (unsigned int j = 0; j < depth();  ++j) { total += sum(j); }
    return total;
}
float Dataset::support(Bitmask const & indicator) const {
    float total = 0;
    for (unsigned int j = 0; j < depth();  ++j) { total += sum(indicator, j); }
    return total;
}

float Dataset::sum(unsigned int k) const {
    return this -> label_indices[k].sum();
}
float Dataset::sum(Bitmask const & indicator, unsigned int k) const {
    return this -> label_indices[k].sum(indicator);
}

similarity_index_type const & Dataset::similar_features(unsigned int j) const {
    return this -> similarity_index.at(j);
}

partition_type Dataset::partition(int const feature_index) {
    return _partition(Bitmask::ones(height()), feature_index);
}

partition_type Dataset::partition(Bitmask const & indicator, int const feature_index) {
    return _partition(indicator, feature_index);
}

partition_type Dataset::_partition(Bitmask const & indicator, const int feature_index) {
    if (indicator.size() != height()) { throw "Indicator size does not match dataset height."; }
    Bitmask feature = this -> columns[feature_index];
    partition_type partitions;
    partitions.emplace_back(indicator & ~feature);
    partitions.emplace_back(indicator & feature);
    return partitions;
    // std::tuple< Bitmask, int > key(indicator, feature_index);
    // if (this -> partition_cache.count(key) == 0) {
    //     Bitmask feature = this -> columns[feature_index];
    //     partition_type partitions;
    //     partitions.emplace_back(indicator & ~feature);
    //     partitions.emplace_back(indicator & feature);
    //     this -> partition_cache.insert(std::make_pair(key, partitions)); // Insert new cache entry
    //     return partitions;
    // } else {
    //     partition_table::const_accessor partition_accessor;
    //     if (this -> partition_cache.find(partition_accessor, key) == false) { throw "Failed Access to Task (Read)"; }
    //     return partition_accessor -> second;
    // }
}

float const Dataset::entropy(Bitmask const & indicator) const {
    float total = support(indicator);
    float entr = 0.0;
    for (unsigned int j = 0; j < depth();  ++j) {
        float prob = sum(indicator, j);
        entr -= total * (prob * log(prob) - log(total) * prob);
    }
    return entr;
}

int const Dataset::height(void) const {
    return std::get<0>(this -> shape);
}

int const Dataset::width(void) const {
    return std::get<1>(this -> shape);
}

int const Dataset::depth(void) const {
    return std::get<2>(this -> shape);
}

int const Dataset::size(void) const {
    return this -> _size;
}


// std::tuple< float, float, float, float > Dataset::impurity(void) const {
//     if (this -> hardware_acceleration == true) {
//         return _kernel_impurity(Bitmask::ones(height()));
//     } else {
//         return _impurity(Bitmask::ones(height()));
//     }
// }

// std::tuple< float, float, float, float > Dataset::impurity(Bitmask const & indicator) const {
//     if (this -> hardware_acceleration == true) {
//         return _kernel_impurity(indicator);
//     } else {
//         return _impurity(indicator);
//     }
// }

// std::tuple< float, float, float, float > Dataset::_kernel_impurity(Bitmask const & indicator) const {
//     #ifdef INCLUDE_OPENCL
//     if (indicator.size() != height()) { throw "Indicator size does not match dataset height."; }

//     int block_count = indicator.dump().size();
//     int work_group_size = this -> work_group_size;
//     int work_group_count = this -> work_group_count;
//     int work_item_count = this -> work_item_count;

//     cl::Buffer filter_buffer(this -> context, CL_MEM_READ_ONLY, sizeof(unsigned long) * block_count);
//     std::vector< cl::Buffer, tbb::scalable_allocator< cl::Buffer > > output_buffers;
//     std::vector< cl::Event, tbb::scalable_allocator< cl::Event > > events;

//     for (int k = 0; k < this -> buffers.size(); ++k) {
//         cl::Kernel sum_kernel(program, "sum_kernel");
//         output_buffers.emplace_back(this -> context, CL_MEM_WRITE_ONLY, sizeof(float) * work_group_count);
//         this -> queue.enqueueWriteBuffer(filter_buffer, CL_TRUE, 0, sizeof(unsigned long) * block_count, indicator.dump().data());
//         sum_kernel.setArg(0, this -> buffers[k]);
//         sum_kernel.setArg(1, filter_buffer);
//         sum_kernel.setArg(2, output_buffers[k]);
//         sum_kernel.setArg(3, sizeof(float) * height(), NULL);

//         cl::Event event;
//         this -> queue.enqueueNDRangeKernel(sum_kernel, cl::NullRange, work_item_count, work_group_size, NULL, &event);
//         events.push_back(event);
//     }

//     for (auto iterator = events.begin(); iterator != events.end(); ++iterator) {
//         cl::Event event = * iterator;
//         event.wait();
//     }

//     std::vector< float, tbb::scalable_allocator< float > > result;
//     std::vector< float, tbb::scalable_allocator< float > > accumulator;
//     for (int k = 0; k < depth(); ++k) {
//         float sum = 0.0;
//         result.clear();
//         result.resize(work_group_count, 0.0);
//         queue.enqueueReadBuffer(output_buffers[k], CL_TRUE, 0, sizeof(float) * work_group_count, result.data());
//         for (auto iterator = result.begin(); iterator != result.end(); ++iterator) {
//             sum += * iterator;
//         }
//         accumulator.emplace_back(sum);
//     }
//     float minimum_impurity = this -> _regularization;
//     result.clear();
//     result.resize(work_group_count, 0.0);
//     queue.enqueueReadBuffer(output_buffers[depth()], CL_TRUE, 0, sizeof(float) * work_group_count, result.data());
//     for (auto iterator = result.begin(); iterator != result.end(); ++iterator) {
//         minimum_impurity += * iterator;
//     }
//     float maximum_impurity = this -> _regularization;
//     result.clear();
//     result.resize(work_group_count, 0.0);
//     queue.enqueueReadBuffer(output_buffers[depth() + 1], CL_TRUE, 0, sizeof(float) * work_group_count, result.data());
//     for (auto iterator = result.begin(); iterator != result.end(); ++iterator) {
//         maximum_impurity += * iterator;
//     }

//     float maximum = 0;
//     float total = 0;
//     for (auto iterator = accumulator.begin(); iterator != accumulator.end(); ++iterator) {
//         float frequency = *iterator;
//         maximum = std::max(frequency, maximum);
//         total += frequency;
//     }
//     float base_impurity = total - maximum + this -> _regularization;
//     std::tuple< float, float, float, float > impurity(minimum_impurity, std::min(maximum_impurity, base_impurity), total, base_impurity);
//     return impurity;
//     #else
//     throw "Error: OpenCL is not supported in this distribution";
//     #endif
// }

// std::tuple< float, float, float, float > Dataset::_impurity(Bitmask const & indicator) const {
//     if (indicator.size() != height()) { throw "Indicator size does not match dataset height."; }

//     float minimum_impurity = this -> _regularization;
//     float maximum_impurity = this -> _regularization;
//     std::vector< float, tbb::scalable_allocator< float > > accumulator;
//     accumulator.resize(depth(), 0);
//     for (int i = 0; i < height(); ++i) {
//         if (indicator[i] != 1) { continue; }

//         std::vector< float > const & element = std::get<1>(this -> rows[i]);
//         float subtotal = 0; // Total label frequency for this feature set
//         float maximum = 0; // Maximum label frequency for this feature set
//         for (int j = 0; j < element.size(); ++j) {
//             maximum = element[j] > maximum ? element[j] : maximum;
//             subtotal += element[j];
//             accumulator[j] += (float) element[j];
//         }
//         minimum_impurity += subtotal - maximum;
//         maximum_impurity += std::min(subtotal - maximum + this -> _regularization, subtotal);
//     }

//     float maximum = 0;
//     float total = 0;
//     for (auto iterator = accumulator.begin(); iterator != accumulator.end(); ++iterator) {
//         float frequency = *iterator;
//         maximum = std::max(frequency, maximum);
//         total += frequency;
//     }
//     float base_impurity = total - maximum + this -> _regularization;
//     std::tuple< float, float, float, float > impurity(minimum_impurity, std::min(maximum_impurity, base_impurity), total, base_impurity);
//     // std::tuple< float, float, float, float > impurity(minimum_impurity + this -> _regularization, base_impurity, total, base_impurity);
//     return impurity;
// }


void Dataset::initialize_kernel(unsigned int platform_index, unsigned int device_index) {
    #ifdef INCLUDE_OPENCL
    try {
        // Set up the platform
        std::vector< cl::Platform > platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << " No OpenCL platforms detected on this system." << std::endl;
            return;
        }
        if (platform_index < 0 || platform_index> platforms.size()) {
            std::cout << "Please specify an index for this list of OpenCL platforms:" << std::endl;
            for (int i = 0; i < platforms.size(); ++i) {
                std::cout << "    " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
            }
            return;
        }
        cl::Platform platform = platforms[platform_index];
        if (configuration["verbose"]) {
            std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        }

        std::vector< cl::Device > devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.size() == 0) {
            std::cout << " No OpenCL devices detected on this system." << std::endl;
            return;
        }
        if (device_index < 0 || device_index > devices.size()) {
            std::cout << "Please specify an index for this list of OpenCL devices:" << std::endl;
            for (int i = 0; i < devices.size(); ++i) {
                std::cout << "    " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
            }
            return;
        }
        cl::Device device = devices[device_index];
        if (configuration["verbose"]) {
            std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        }
        
        cl::Context context({device});
        cl::CommandQueue queue(context, device);

        // Set up the program
        cl::Program::Sources sources;
        std::stringstream dataset_encoding;
        for (int i = 0; i < height(); ++i) {
            std::vector< float > element = std::get<1>(this -> rows[i]);
            for (int j = 0; j < element.size(); ++j) {
                if ( i > 0 || j > 0) {
                    dataset_encoding << ',';
                }
                dataset_encoding << std::to_string(element[j]);
            }
        }

        std::string sum_source =
            "void kernel sum_kernel("
            "global const float * input,"
            "global const unsigned long * filter,"
            "global float * output,"
            "local float * stage) {\n"
            "    int global_id = get_global_id(0);\n" // Globally unique ID
            "    int local_id = get_local_id(0);\n"   // Local ID within work-group
            "    int group_id = get_group_id(0);\n"   // Global ID of the work-group
            "    int group_size = get_local_size(0);\n"   // Size of work-group

            "    int block_id = global_id / " + std::to_string(sizeof(unsigned long) * 8) + ";\n" // Block ID into the filter vector
            "    int block_offset = global_id % " + std::to_string(sizeof(unsigned long) * 8) + ";\n" // Offset within the block of the filter vector
            "    int filter_value = (filter[block_id] >> block_offset) % 2;\n" // Extract filter bit

            "    if (global_id < " + std::to_string(height()) + ") {\n"
            "        stage[local_id] = filter_value * input[global_id];\n"
            "    } else {\n"
            "        stage[local_id] = 0;\n"
            "    }\n"

            "    barrier(CLK_LOCAL_MEM_FENCE);\n" // Memory Fence

            "    for (int window_size = group_size; window_size > 1; window_size = window_size >> 1) {\n"
            "        if (local_id < window_size / 2) {\n"
            "            stage[local_id] += stage[local_id + window_size / 2];\n" // Fold elements from the second half of the window
            "        };\n"
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" // Memory Fence
            "    }\n"
            "    if (local_id == 0) {\n"
            "        output[group_id] = stage[local_id];\n"
            "    }\n"
            "    return;\n"
            "}";
        std::string min_source =
            "void kernel min_kernel("
            "global const float * input,"
            "global const unsigned long * filter,"
            "global float * output,"
            "local float * stage) {\n"
            "    int global_id = get_global_id(0);\n" // Globally unique ID
            "    int local_id = get_local_id(0);\n"   // Local ID within work-group
            "    int group_id = get_group_id(0);\n"   // Global ID of the work-group
            "    int group_size = get_local_size(0);\n"   // Size of work-group

            "    int block_id = global_id / " + std::to_string(sizeof(unsigned long)) + ";\n" // Block ID into the filter vector
            "    int block_offset = global_id % " + std::to_string(sizeof(unsigned long)) + ";\n" // Offset within the block of the filter vector
            "    int filter_value = (filter[block_id] >> block_offset) % 2;\n" // Extract filter bit

            "    if (global_id < " + std::to_string(height()) + ") {\n"
            "        stage[local_id] = filter_value * input[global_id];\n"
            "    } else {\n"
            "        stage[local_id] = 1;\n"
            "    }\n"

            "    barrier(CLK_LOCAL_MEM_FENCE);\n" // Memory Fence

            "    for (int window_size = group_size; window_size > 1; window_size = window_size >> 1) {\n"
            "        if (local_id < window_size / 2 && stage[local_id + window_size / 2] < stage[local_id]) {\n"
            "            stage[local_id] = stage[local_id + window_size / 2];\n" // Fold elements from the second half of the window
            "        };\n"
            "        barrier(CLK_LOCAL_MEM_FENCE);\n" // Memory Fence
            "    }\n"
            "    if (local_id == 0) {\n"
            "        output[group_id] = stage[local_id];\n"
            "    }\n"
            "    return;\n"
            "}";
        sources.push_back({sum_source.c_str(), sum_source.length()});
        // sources.push_back({min_source.c_str(), min_source.length()});
        cl::Program program(context, sources);

        // Build the program
        try {
            program.build({device});
        } catch (cl::Error& error) {
            if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
                std::string name = device.getInfo<CL_DEVICE_NAME>();
                std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << "Build log for " << name << ":" << std::endl << buildlog << std::endl;
            }
            throw error;
        }

        // Store necessary references for future kernel calls
        this -> context = context;
        this -> queue = queue;
        this -> program = program;
        this -> work_group_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
        this -> work_group_count = height() / this -> work_group_size + int((height() % this -> work_group_size) != 0);
        this -> work_item_count = work_group_count * work_group_size;

        // Preload input buffers
        int j = 0;
        for (auto iterator = this -> label_distributions.begin(); iterator != this -> label_distributions.end(); ++iterator) {
            std::vector< float > label_distribution = * iterator;
            this -> buffers.emplace_back(context, CL_MEM_READ_ONLY, sizeof(float) * label_distribution.size());
            queue.enqueueWriteBuffer(this -> buffers[j], CL_TRUE, 0, sizeof(float) * label_distribution.size(), label_distribution.data());
            ++j;
        }
        cl::Buffer non_majority_buffer(context, CL_MEM_READ_ONLY, sizeof(float) * height());
        queue.enqueueWriteBuffer(non_majority_buffer, CL_TRUE, 0, sizeof(float) * height(), this -> non_majority_distribution.data());
        this -> buffers.push_back(non_majority_buffer);

        cl::Buffer sample_buffer(context, CL_MEM_READ_ONLY, sizeof(float) * height());
        queue.enqueueWriteBuffer(sample_buffer, CL_TRUE, 0, sizeof(float) * height(), this -> sample_distribution.data());
        this -> buffers.push_back(sample_buffer);

        this -> hardware_acceleration = true;
    } catch( cl::Error error ) {
        std::cout << "OpenCLError. Error Code: " << int(error.err()) << " | API: " << error.what() << std::endl;
    }
    #endif
}
