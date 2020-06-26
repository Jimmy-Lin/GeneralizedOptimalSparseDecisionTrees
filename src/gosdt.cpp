#include "gosdt.hpp"

#define _DEBUG true
#define THROTTLE false

float GOSDT::time = 0.0;
unsigned int GOSDT::size = 0;
unsigned int GOSDT::iterations = 0;
unsigned int GOSDT::status = 0;

GOSDT::GOSDT(void) {}

GOSDT::~GOSDT(void) {
    return;
}

void GOSDT::configure(std::istream & config_source) { Configuration::configure(config_source); }

void GOSDT::fit(std::istream & data_source, std::string & result) {
    std::unordered_set< Model > models;
    fit(data_source, models);
    json output = json::array();
    for (auto iterator = models.begin(); iterator != models.end(); ++iterator) {
        Model model = * iterator;
        json object = json::object();
        model.to_json(object);
        output.push_back(object);
    }
    result = output.dump(2);
}

void GOSDT::fit(std::istream & data_source, std::unordered_set< Model > & models) {
    if(Configuration::verbose) { std::cout << "Using configuration: " << Configuration::to_string(2) << std::endl; }

    if(Configuration::verbose) { std::cout << "Initializing Optimization Framework" << std::endl; }
    Optimizer optimizer;
    optimizer.load(data_source);

    GOSDT::time = 0.0;
    GOSDT::size = 0;
    GOSDT::iterations = 0;
    GOSDT::status = 0;

    std::vector< std::thread > workers;
    std::vector< int > iterations(Configuration::worker_limit);

    if(Configuration::verbose) { std::cout << "Starting Optimization" << std::endl; }
    auto start = std::chrono::high_resolution_clock::now();

    optimizer.initialize();
    for (unsigned int i = 0; i < Configuration::worker_limit; ++i) {
        workers.emplace_back(work, i, std::ref(optimizer), std::ref(iterations[i]));
        #ifndef __APPLE__
        if (Configuration::worker_limit > 1) {
            // If using Ubuntu Build, we can pin each thread to a specific CPU core to improve cache locality
            cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(i, &cpuset);
            int error = pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (error != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << error << std::endl; }
        }
        #endif
    }
    for (auto iterator = workers.begin(); iterator != workers.end(); ++iterator) { (* iterator).join(); } // Wait for the thread pool to terminate
    
    auto stop = std::chrono::high_resolution_clock::now(); // Stop measuring training time
    GOSDT::time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0;
    if(Configuration::verbose) { std::cout << "Optimization Complete" << std::endl; }

    for (auto iterator = iterations.begin(); iterator != iterations.end(); ++iterator) { GOSDT::iterations += * iterator; }    
    GOSDT::size = optimizer.size();

    if (Configuration::timing != "") {
        std::ofstream timing_output(Configuration::timing, std::ios_base::app);
        timing_output << GOSDT::time;
        timing_output.flush();
        timing_output.close();
    }

    if(Configuration::verbose) {
        std::cout << "Training Duration: " << GOSDT::time << " seconds" << std::endl;
        std::cout << "Number of Iterations: " << GOSDT::iterations << " iterations" << std::endl;
        std::cout << "Size of Graph: " << GOSDT::size << " nodes" << std::endl;
        float lowerbound, upperbound;
        optimizer.objective_boundary(& lowerbound, & upperbound);
        std::cout << "Objective Boundary: [" << lowerbound << ", " << upperbound << "]" << std::endl;
        std::cout << "Optimality Gap: " << optimizer.uncertainty() << std::endl;
    }

    // try 
    { // Model Extraction
        if (!optimizer.complete()) {
            GOSDT::status = 1;
            if (Configuration::diagnostics) {
                std::cout << "Non-convergence Detected. Beginning Diagnosis" << std::endl;
                optimizer.diagnose_non_convergence();
                std::cout << "Diagnosis complete" << std::endl;
            }
        }

        optimizer.models(models);

        if (Configuration::model_limit > 0 && models.size() == 0) {
            GOSDT::status = 1;
            if (Configuration::diagnostics) {
                std::cout << "False-convergence Detected. Beginning Diagnosis" << std::endl;
                optimizer.diagnose_false_convergence();
                std::cout << "Diagnosis complete" << std::endl;
            }
        }

        if (Configuration::verbose) {
            std::cout << "Models Generated: " << models.size() << std::endl;
            if (optimizer.uncertainty() == 0.0 && models.size() > 0) {
                std::cout << "Loss: " << models.begin() -> loss() << std::endl;
                std::cout << "Complexity: " << models.begin() -> complexity() << std::endl;
            } 
        }
        if (Configuration::model != "") {
            json output = json::array();
            for (auto iterator = models.begin(); iterator != models.end(); ++iterator) {
                Model model = * iterator;
                json object = json::object();
                model.to_json(object);
                output.push_back(object);
            }
            std::string result = output.dump(2);
            if(Configuration::verbose) { std::cout << "Storing Models in: " << Configuration::model << std::endl; }
            std::ofstream out(Configuration::model);
            out << result;
            out.close();
        }
    }
    //  catch (IntegrityViolation exception) {
    //     GOSDT::status = 1;
    //     std::cout << exception.to_string() << std::endl;
    // }
}

void GOSDT::work(int const id, Optimizer & optimizer, int & return_reference) {
    unsigned int iterations = 0;
    try {
        while (optimizer.iterate(id)) { iterations += 1; }
    } catch( IntegrityViolation exception ) {
        GOSDT::status = 1;
        std::cout << exception.to_string() << std::endl;
        throw std::move(exception);
    }
    return_reference = iterations;
}