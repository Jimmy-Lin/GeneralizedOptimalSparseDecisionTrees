#include "optimizer.hpp"
#include "kernels/execute.hpp"
#include "kernels/utility.hpp"

Optimizer::Optimizer(void) {
    return;
}

Optimizer::Optimizer(Dataset & dataset, json const & conf) : _configuration(conf), dataset(dataset) {
    // Load some configuration variables to improve access speed
    if (configuration().contains("regularization")) { this -> regularization = configuration()["regularization"]; }
    if (configuration().contains("uncertainty_tolerance")) { this -> uncertainty_tolerance = configuration()["uncertainty_tolerance"]; }
    if (configuration().contains("time_limit")) { this -> time_limit = configuration()["time_limit"]; }
    if (configuration().contains("output_limit")) { this -> output_limit = configuration()["output_limit"]; }

    if (configuration().contains("optimism")) { this -> optimism = configuration()["optimism"]; }
    if (configuration().contains("equity")) { this -> equity = configuration()["equity"]; }
    if (configuration().contains("sample_depth")) { this -> sample_depth = configuration()["sample_depth"]; }
    if (configuration().contains("similarity_threshold")) { this -> similarity_threshold = configuration()["similarity_threshold"]; }

    if (configuration().contains("profile_output")) { this -> profile_output = configuration()["profile_output"]; }
    if (configuration().contains("timing_output")) { this -> timing_output = configuration()["timing_output"]; }

    if (configuration().contains("workers")) { this -> workers = configuration()["workers"]; }

    int const n = this -> dataset.height();
    int const m = this -> dataset.width();

    // Initialize the queues
    unsigned int partition_count = n;
    this -> queue.initialize(this -> workers, partition_count, this -> equity);

    // Seed the optimizer with a root task
    Key key(Bitmask::ones(n));
    Bitmask sensitivity = Bitmask::ones(m);
    this -> queue.push(key);
    this -> graph.tasks.insert(std::make_pair(key, new_task(key, sensitivity, this -> dataset.similarity_index))); // Attempt to insert a new task
    this -> graph.backward_index.insert(std::make_pair(key, index_type())); // Initializes a new index

    this -> start_time = tbb::tick_count::now();
    this -> last_tick = this -> start_time;
    return;
}

Optimizer::~Optimizer(void) {
    return;
}

json const & Optimizer::configuration(void) const {
    return this -> _configuration;
}

std::tuple< float, float> Optimizer::objective_boundary(void) const {
    Key key(Bitmask::ones(this -> dataset.height()));
    task_table::const_accessor task_accessor;
    if (this -> graph.tasks.find(task_accessor, key)) {
        Task const & task = task_accessor -> second;
        return std::tuple< float, float>(task.lowerbound(), task.upperbound());
    } else {
        return std::tuple< float, float>(0.0, 1.0);
    }
}

float const Optimizer::uncertainty(void) {
    Key key(Bitmask::ones(this -> dataset.height()));
    task_table::const_accessor task_accessor;
    if (this -> graph.tasks.find(task_accessor, key)) {
        Task const & task = task_accessor -> second;
        return task.uncertainty();
    } else {
        return 1.0;
    }
}

float const Optimizer::elapsed(void) {
    auto now = tbb::tick_count::now();
    float duration = (now - this -> start_time).seconds();
    return duration;
}

bool const Optimizer::timeout(void) {
    return (this -> time_limit > 0 && elapsed() > this -> time_limit);
}

bool const Optimizer::tick(int const id) {
    if (this -> profile_output == "") { return false; }
    if (this -> ticks == 0) {
        std::ofstream profile_output(this -> profile_output);
        profile_output << "time,lowerbound,upperbound,graph_size,graph_hits,queue_size";
        for (int k = 0; k < this -> queue.width(); ++k) {
            profile_output << ",local_queue_" << k << "_size";
        }
        profile_output << std::endl;
        profile_output.flush();
    }
    this -> ticks += 1;
    if (((this -> ticks) % (this -> tick_duration)) == 0 || complete() || timeout()) {
        std::ofstream profile_output(this -> profile_output, std::ios_base::app);
        std::tuple< float, float> boundary = objective_boundary();
        profile_output << 
            elapsed() << ", " << std::get<0>(boundary) << ", " << std::get<1>(boundary) << "," << this -> graph.tasks.size() << "," << this -> graph.hits << "," << this -> queue.size();
        for (int k = 0; k < this -> queue.width(); ++k) {
            profile_output << "," << this -> queue.local_size(k);
        }
        profile_output << std::endl;
        profile_output.flush();
        return true;
    } else {
        return false;
    }
}

bool const Optimizer::complete(void) {
    Key key(Bitmask::ones(this -> dataset.height()));
    task_table::const_accessor task_accessor;
    if (this -> graph.tasks.find(task_accessor, key)) {
        Task const & task = task_accessor -> second;
        if (this -> uncertainty_tolerance == 0) {
            return task.resolved();
        } else {
            return uncertainty() < this -> uncertainty_tolerance;
        }
    } else {
        return false;
    }
}

void Optimizer::iterate(int const id) {
    std::tuple< Key, float, float, float > item;
    if (this -> queue.pop(item, id)) {
        Key key = std::get< 0 >(item);
        // std::cout << key.indicator().count() << std::endl;
        execute(key);      
    }
    return;
}

std::unordered_set< Model > Optimizer::models(unsigned int output_limit) {
    if (!complete()) {
        std::cout << "Non-Convergence Detected. Beginning Diagnosis" << std::endl;
        // diagnose_non_convergence();
    }
    if (output_limit == 0) {
        std::unordered_set< Model > results;
        return results;
    }
    // std::vector< Model > results;
    Key key(Bitmask::ones(this -> dataset.height()));
    task_table::accessor task_accessor;
    this -> graph.tasks.find(task_accessor, key);
    Task & task = task_accessor -> second;
    std::unordered_set< Model > results = models(key, task, output_limit);
    if (results.size() == 0 && output_limit > 0) {
        std::cout << "False Convergence Detected. Beginning Diagnosis" << std::endl;
        // diagnose_false_convergence();
    }
    if (results.size() <=  output_limit) {
        return results;
    } else {
        std::unordered_set< Model > limited_results;
        for (auto iterator = results.begin(); iterator != results.end(); ++iterator) {
            if (limited_results.size() == output_limit) { break; }
            limited_results.insert(* iterator);
        }
        return limited_results;
    }
}

std::unordered_set< Model > Optimizer::models(Key const & key, Task & task, unsigned int output_limit) {
    // std::cout << "Task(" << key.indicator().to_string() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "]" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl;
    float const epsilon = std::numeric_limits<float>::epsilon();
    if (task.cancelled() == true) {
        std::unordered_set< Model > results;
        return results;
    } else { 
        std::unordered_set< Model > results;
        int const m = this -> dataset.width();
        float task_upperbound = task.upperbound();

        if (task_upperbound < task.base_objective() + epsilon) {
            for (int j = 0; j < m; ++j) {
                if (!task.sensitive(j) || results.size() >= output_limit) { continue; }

                float combined_lowerbound = task.combined_bounds[j].first;
                float combined_upperbound = task.combined_bounds[j].second;
                bool subtask_cancelled = false;

                auto partitions = this -> dataset.partition(key.indicator(), j);
                for (auto iterator = partitions.begin(); iterator != partitions.end() && subtask_cancelled == false; ++iterator) {
                    Key subkey(* iterator);
                    if (subtask_cancelled == true || subkey.indicator().count() == key.indicator().count() || subkey.indicator().count() == 0) {
                        subtask_cancelled = true;
                        continue;
                    }
                    if (this -> graph.tasks.count(subkey) == 0) { 
                        subtask_cancelled = true;
                        continue;
                    }
                }

                if (combined_upperbound <= task_upperbound + epsilon && subtask_cancelled == false) {
                    std::unordered_set< Model > subresults;
                    // transfer the models from the qualifying summation into overall list of models
                    /**********************************************************************************/
                    // std::vector< bool > values;
                    std::vector< std::unordered_set< Model > > model_space;
                    // Create a model space such that each subtask is a dimension
                    // and the models from that subtask are the values of that dimension
                    for (auto iterator = partitions.begin(); iterator != partitions.end(); ++iterator) {
                        Key subkey(* iterator);
                        task_table::accessor subtask_accessor;
                        if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { throw "Failed Access to Subtask Subtask (Extraction)"; }
                        Task & subtask = subtask_accessor -> second;
                        model_space.push_back(models(subkey, subtask, output_limit));
                    }
                    bool empty_dimension = false;
                    for (auto iterator = model_space.begin(); iterator != model_space.end(); ++iterator) {
                        std::unordered_set< Model > submodels = * iterator;
                        if (submodels.size() == 0) { empty_dimension = true; }
                    }
                    if (empty_dimension == false) {
                        std::vector< std::vector< Model > > combinations;
                        // Compute possible submodel combinations as coordinates in the model space defined earlier
                        for (auto iterator = model_space.begin(); iterator != model_space.end(); ++iterator) {
                            std::unordered_set< Model > submodels = * iterator;
                            if (combinations.size() == 0) {
                                for (auto iterator = submodels.begin(); iterator != submodels.end(); ++iterator) {
                                    std::vector< Model > combination;
                                    combination.push_back(* iterator);
                                    combinations.push_back(combination);
                                }
                            } else {
                                std::vector< std::vector< Model > > new_combinations;
                                for (auto iterator = combinations.begin(); iterator != combinations.end(); ++iterator) {
                                    std::vector< Model > combination = * iterator;
                                    for (auto iterator = submodels.begin(); iterator != submodels.end(); ++iterator) {
                                        std::vector< Model > prefix(combination);
                                        prefix.push_back(* iterator);
                                        new_combinations.push_back(prefix);
                                    }
                                }
                                combinations = new_combinations;
                            }
                        }
                        // Extract each submodel combination and map them to the feature_value that leads into each one. 
                        for (auto iterator = combinations.begin(); iterator != combinations.end(); ++iterator) {
                            std::vector< Model > combination = * iterator;
                            std::map< bool, Model > submodel_map;
                            for (int k = 0; k < model_space.size(); ++k) {
                                submodel_map[(bool)(k)] = combination[k];
                            }
                            std::tuple< unsigned int, unsigned int > decoding = this -> dataset.encoder.decode(j);
                            unsigned int feature_index = std::get<0>(decoding);
                            unsigned int offset = std::get<1>(decoding);                        
                            std::string const & feature_name = this -> dataset.encoder.header(feature_index);
                            std::vector< std::string > const & rule = this -> dataset.encoder.decoder(feature_index, offset);
                            // unsigned int feature, std::string feature_name, std::string type, std::string relation, std::string reference, std::map< bool, Model > const & submodels
                            subresults.insert(Model(feature_index, feature_name, rule[0], rule[1], rule[2], submodel_map));
                        }
                    }

                    /**********************************************************************************/

                    for (auto iterator = subresults.begin(); iterator != subresults.end(); ++iterator) {
                        Model submodel = * iterator;
                        results.insert(submodel);
                    }
                }
            }
        }
        

        if (task.base_objective() <= task_upperbound + epsilon) {
            // Optimal models include a stump
            // Use the dataset encoder to transform the graph results into a 
            // Terminal Node Constructor requires (Bitmask const & capture, std::string prediction, float loss, float complexity)
            auto leaf = this -> dataset.leaf(key.indicator()); // < Prediction, Loss >
            std::string const & feature_name = this -> dataset.encoder.header();
            std::string type = this -> dataset.encoder.decoder()[0];
            results.insert(Model(feature_name, type, std::get<0>(leaf), task.base_objective() - (this -> regularization), this -> regularization, key.indicator()));
        }

        std::unordered_set< Model > bounded_results;
        for (auto iterator = results.begin(); iterator != results.end(); ++iterator) {
            if (bounded_results.size() >= output_limit) { break; }
            bounded_results.insert(* iterator);
        }
        return bounded_results;
    }
}


void Optimizer::diagnose_non_convergence(void) {
    Key key(Bitmask::ones(this -> dataset.height()));
    diagnose_non_convergent_task(key);
    return;
}
void Optimizer::diagnose_non_convergent_task(Key const & key) {
    task_table::const_accessor task_accessor;
    this -> graph.tasks.find(task_accessor, key);
    Task const & task = task_accessor -> second;
    if (task.resolved()) {
        return;
    }
    std::cout << "Task(" << key.indicator().to_string() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "]" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl; 
    int const m = this -> dataset.width();
    int reasons = 0;
    for (int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }
        auto partitions = this -> dataset.partition(key.indicator(), j);
        for (auto iterator = partitions.begin(); iterator != partitions.end(); ++iterator) {
            Key subkey(* iterator);
            if (subkey.indicator().count() == key.indicator().count() || subkey.indicator().count() == 0) {
                continue;
            }
            task_table::const_accessor subtask_accessor;
            if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { throw "Failed Access to Subtask (Diagnosis)"; }
            Task const & subtask = subtask_accessor -> second;
            if (subtask.resolved() == false) {
                ++reasons;
                if (subtask.lowerbound() == task.lowerbound()) {
                    std::cout << "Task(" << key.indicator().to_string() << ")'s lowerbound depends on Task(" << subkey.indicator().to_string() << ")" << std::endl;
                }
                if (subtask.upperbound() == task.upperbound()) {
                    std::cout << "Task(" << key.indicator().to_string() << ")'s upperbound depends on Task(" << subkey.indicator().to_string() << ")" << std::endl;
                }
                if (subtask.lowerbound() == task.lowerbound() || subtask.upperbound() == task.upperbound()) {
                    diagnose_non_convergent_task(subkey);
                }
            }
        }
    }
    if (reasons == 0) {
        if (task.explored() == false || task.delegated() == false) {
            std::cout << "Task(" << key.indicator().to_string() << ") is missing a downward call." << std::endl;
        } else {
            std::cout << "Task(" << key.indicator().to_string() << ") is missing an upward call." << std::endl;
        }
    }
}

void Optimizer::diagnose_false_convergence(void) {
    Key key(Bitmask::ones(this -> dataset.height()));
    diagnose_falsely_convergent_task(key);
    return;
}
void Optimizer::diagnose_falsely_convergent_task(Key const & key) {
    float const epsilon = std::numeric_limits<float>::epsilon();
    task_table::const_accessor task_accessor;
    this -> graph.tasks.find(task_accessor, key);
    Task const & task = task_accessor -> second;

    std::unordered_set< Model > results;
    int const m = this -> dataset.width();
    float task_upperbound = task.base_objective();
    for (int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }

        float combined_lowerbound = 0.0;
        float combined_upperbound = 0.0;
        bool subtask_cancelled = false;

        auto partitions = this -> dataset.partition(key.indicator(), j);
        for (auto iterator = partitions.begin(); iterator != partitions.end() && subtask_cancelled == false; ++iterator) {
            Key subkey(* iterator);
            if (subtask_cancelled == true || subkey.indicator().count() == key.indicator().count() || subkey.indicator().count() == 0) {
                subtask_cancelled = true;
                continue;
            }
            task_table::const_accessor subtask_accessor;
            if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { 
                subtask_cancelled = true;
                continue;
            }
            Task const & subtask = subtask_accessor -> second;
            combined_lowerbound += subtask.lowerbound();
            combined_upperbound += subtask.upperbound();
        }
        if (subtask_cancelled == false) {
            task_upperbound = std::min(task_upperbound, combined_upperbound);
        }
    }
    for (int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }

        float combined_lowerbound = 0.0;
        float combined_upperbound = 0.0;
        bool subtask_cancelled = false;

        auto partitions = this -> dataset.partition(key.indicator(), j);
        for (auto iterator = partitions.begin(); iterator != partitions.end() && subtask_cancelled == false; ++iterator) {
            Key subkey(* iterator);
            if (subtask_cancelled == true || subkey.indicator().count() == key.indicator().count() || subkey.indicator().count() == 0) {
                subtask_cancelled = true;
                continue;
            }
            task_table::const_accessor subtask_accessor;
            if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { 
                subtask_cancelled = true;
                continue;
            }
            Task const & subtask = subtask_accessor -> second;
            combined_lowerbound += subtask.lowerbound();
            combined_upperbound += subtask.upperbound();
        }

        if (combined_upperbound <= task_upperbound + epsilon && subtask_cancelled == false) {
            std::unordered_set< Model > subresults;
            // transfer the models from the qualifying summation into overall list of models
            /**********************************************************************************/
            // std::vector< bool > values;
            std::vector< std::unordered_set< Model > > model_space;
            // Create a model space such that each subtask is a dimension
            // and the models from that subtask are the values of that dimension
            for (auto iterator = partitions.begin(); iterator != partitions.end(); ++iterator) {
                Key subkey(* iterator);
                task_table::accessor subtask_accessor;
                if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { throw "Failed Access to Subtask Subtask (Extraction)"; }
                Task & subtask = subtask_accessor -> second;
                model_space.push_back(models(subkey, subtask, output_limit));
            }
            bool empty_dimension = false;
            for (auto iterator = model_space.begin(); iterator != model_space.end(); ++iterator) {
                std::unordered_set< Model > submodels = * iterator;
                if (submodels.size() == 0) { empty_dimension = true; }
            }
            if (empty_dimension == false) {
                std::vector< std::vector< Model > > combinations;
                // Compute possible submodel combinations as coordinates in the model space defined earlier
                for (auto iterator = model_space.begin(); iterator != model_space.end(); ++iterator) {
                    std::unordered_set< Model > submodels = * iterator;
                    if (combinations.size() == 0) {
                        for (auto iterator = submodels.begin(); iterator != submodels.end(); ++iterator) {
                            std::vector< Model > combination;
                            combination.push_back(* iterator);
                            combinations.push_back(combination);
                        }
                    } else {
                        std::vector< std::vector< Model > > new_combinations;
                        for (auto iterator = combinations.begin(); iterator != combinations.end(); ++iterator) {
                            std::vector< Model > combination = * iterator;
                            for (auto iterator = submodels.begin(); iterator != submodels.end(); ++iterator) {
                                std::vector< Model > prefix(combination);
                                prefix.push_back(* iterator);
                                new_combinations.push_back(prefix);
                            }
                        }
                        combinations = new_combinations;
                    }
                }
                // Extract each submodel combination and map them to the feature_value that leads into each one. 
                for (auto iterator = combinations.begin(); iterator != combinations.end(); ++iterator) {
                    std::vector< Model > combination = * iterator;
                    std::map< bool, Model > submodel_map;
                    for (int k = 0; k < model_space.size(); ++k) {
                        submodel_map[(bool)(k)] = combination[k];
                    }
                    std::tuple< unsigned int, unsigned int > decoding = this -> dataset.encoder.decode(j);
                    unsigned int feature_index = std::get<0>(decoding);
                    unsigned int offset = std::get<1>(decoding);                        
                    std::string const & feature_name = this -> dataset.encoder.header(feature_index);
                    std::vector< std::string > const & rule = this -> dataset.encoder.decoder(feature_index, offset);
                    // unsigned int feature, std::string feature_name, std::string type, std::string relation, std::string reference, std::map< bool, Model > const & submodels
                    subresults.insert(Model(feature_index, feature_name, rule[0], rule[1], rule[2], submodel_map));
                }
            }

            /**********************************************************************************/

            for (auto iterator = subresults.begin(); iterator != subresults.end(); ++iterator) {
                Model submodel = * iterator;
                results.insert(submodel);
            }
        }

    }

    if (task.base_objective() <= task_upperbound + epsilon) {
        // Optimal models include a stump
        // Use the dataset encoder to transform the graph results into a 
        // Terminal Node Constructor requires (Bitmask const & capture, std::string prediction, float loss, float complexity)
        auto leaf = this -> dataset.leaf(key.indicator()); // < Prediction, Loss >
        std::string const & feature_name = this -> dataset.encoder.header();
        std::string type = this -> dataset.encoder.decoder()[0];
        results.insert(Model(feature_name, type, std::get<0>(leaf), std::get<1>(leaf), this -> regularization, key.indicator()));
    }
    if (results.size() > 0) { return; }
    std::cout << "Task(" << key.indicator().to_string() << ") falsely converged to bounds [" << task.lowerbound() << ", " << task_upperbound + epsilon << "], Base Objective = " << task.base_objective() << ", Base Match = " << (int)(task.base_objective() <= task_upperbound + epsilon) << std::endl;
    for (int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }

        float combined_lowerbound = 0.0;
        float combined_upperbound = 0.0;
        bool subtask_cancelled = false;

        auto partitions = this -> dataset.partition(key.indicator(), j);
        for (auto iterator = partitions.begin(); iterator != partitions.end() && subtask_cancelled == false; ++iterator) {
            Key subkey(* iterator);
            if (subkey.indicator().count() == key.indicator().count() || subkey.indicator().count() == 0) {
                std::cout << "Subtask " << j << " cancelled due to circular dependency" << std::endl;
                subtask_cancelled = true;
                continue;
            }
            task_table::const_accessor subtask_accessor;
            if (this -> graph.tasks.find(subtask_accessor, subkey) == false) {
                std::cout << "Subtask " << j << " cancelled due to missing dependency" << std::endl;
                subtask_cancelled = true;
                continue;
            }
            Task const & subtask = subtask_accessor -> second;
            combined_lowerbound += subtask.lowerbound();
            combined_upperbound += subtask.upperbound();
        }
        std::cout << "Subtask " << j << " reports bounds [" << combined_lowerbound << ", " << combined_upperbound << "], Cancellation = " << (int) subtask_cancelled << std::endl;

        if (combined_upperbound <= task_upperbound && subtask_cancelled == false) {
            for (auto iterator = partitions.begin(); iterator != partitions.end() && subtask_cancelled == false; ++iterator) {
                Key subkey(* iterator);
                std::cout << "Task(" << key.indicator().to_string() << ") points to Task(" << subkey.indicator().to_string() << ")" << std::endl;
            }
            std::cout << "Task(" << key.indicator().to_string() << ") falsely converged." << std::endl;
            for (auto iterator = partitions.begin(); iterator != partitions.end() && subtask_cancelled == false; ++iterator) {
                Key subkey(* iterator);
                diagnose_falsely_convergent_task(subkey);
            }
        }
    }
}

unsigned int Optimizer::size(void) {
    return this -> graph.tasks.size();
}