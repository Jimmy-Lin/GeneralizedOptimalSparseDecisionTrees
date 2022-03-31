#include "gosdt.hpp"

//#include <sys/time.h>
//#include <sys/resource.h>  // FIREWOLF: Incompatible with Windows

#define _DEBUG true
#define THROTTLE false

float GOSDT::time = 0.0;
unsigned int GOSDT::size = 0;
unsigned int GOSDT::iterations = 0;
unsigned int GOSDT::status = 0;

double GOSDT::lower_bound;
double GOSDT::upper_bound;
float GOSDT::model_loss;

float  GOSDT::ru_utime;
float  GOSDT::ru_stime;
long   GOSDT::ru_maxrss;
long   GOSDT::ru_nswap;
long   GOSDT::ru_nivcsw;

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

    int const n = State::dataset.size();
    if(Configuration::regularization < (float) 1/n) {
        std::cout << "Regularization smaller than 1/(num_samples) - this may lead to longer training time if not adjusted." << std::endl;
        if (!Configuration::allow_small_reg) {
            std::cout << "Regularization increased to 1/(num_samples) = " << (1/n) << ". To allow regularization below this, set allow_small_reg to true" << std::endl;
            Configuration::regularization = (float) 1/n;
        }
    }

    GOSDT::time = 0.0;
    GOSDT::size = 0;
    GOSDT::iterations = 0;
    GOSDT::status = 0;

    std::vector< std::thread > workers;
    std::vector< int > iterations(Configuration::worker_limit);

    if(Configuration::verbose) { std::cout << "Starting Optimization" << std::endl; }

//    static struct rusage usage_start, usage_end;
//    if (getrusage(RUSAGE_SELF, &usage_start)) {
//         std::cout << "WARNING: rusage returned non-zero value" << std::endl;
//    }

    auto start = std::chrono::high_resolution_clock::now();

    optimizer.initialize();
    if (Configuration::worker_limit > 1) {
        for (unsigned int i = 0; i < Configuration::worker_limit; ++i) {
            workers.emplace_back(work, i, std::ref(optimizer), std::ref(iterations[i]));
//            #ifndef __APPLE__
//            // If using Ubuntu Build, we can pin each thread to a specific CPU core to improve cache locality
//            cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(i, &cpuset);
//            int error = pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
//            if (error != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << error << std::endl; }
//            #endif
        }
        for (auto iterator = workers.begin(); iterator != workers.end(); ++iterator) { (* iterator).join(); } // Wait for the thread pool to terminate
    }else { 
        work(0, optimizer, iterations[0]);
    }

    auto stop = std::chrono::high_resolution_clock::now(); // Stop measuring training time
// FIREWOLF: Incompatible with Windows MSVC
//    if (getrusage(RUSAGE_SELF, &usage_end)) {
//        std::cout << "WARNING: rusage returned non-zero value" << std::endl;
//        GOSDT::ru_utime = -1;
//        GOSDT::ru_stime = -1;
//        GOSDT::ru_maxrss = -1;
//        GOSDT::ru_nswap = -1;
//        GOSDT::ru_nivcsw = -1;
//    } else {
//        struct timeval delta;
//        timersub(&usage_end.ru_utime, &usage_start.ru_utime, &delta);
//        GOSDT::ru_utime = (float)delta.tv_sec + (((float)delta.tv_usec) / 1000000);
//        timersub(&usage_end.ru_stime, &usage_start.ru_stime, &delta);
//        GOSDT::ru_stime = (float)delta.tv_sec + (((float)delta.tv_usec) / 1000000);
//        GOSDT::ru_maxrss = usage_end.ru_maxrss;
//        GOSDT::ru_nswap = usage_end.ru_nswap - usage_start.ru_nswap;
//        GOSDT::ru_nivcsw = usage_end.ru_nivcsw - usage_start.ru_nivcsw;
//    }
    GOSDT::time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() / 1000.0;
    if(Configuration::verbose) { std::cout << "Optimization Complete" << std::endl; }

    for (auto iterator = iterations.begin(); iterator != iterations.end(); ++iterator) { GOSDT::iterations += * iterator; }
    GOSDT::size = optimizer.size();
    float lowerbound, upperbound;
    optimizer.objective_boundary(& lowerbound, & upperbound);
    GOSDT::lower_bound = lowerbound;
    GOSDT::upper_bound = upperbound;

    if(Configuration::verbose) { std::cout << "Optimization Complete" << std::endl; }


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
        std::cout << "Training Duration (user time): " <<  GOSDT::ru_utime << " seconds" << std::endl;
        std::cout << "Training Duration (system time)" << GOSDT::ru_stime << " seconds" << std::endl;
        std::cout << "Maximim memory: " << GOSDT::ru_utime  << " kB" << std::endl;
        std::cout << "Number of swaps" << GOSDT::ru_nswap << std::endl;
        std::cout << "Size of Search Graph: " << optimizer.size() << " nodes" << std::endl;
        std::cout << "Objective Boundary: [" << lowerbound << ", " << upperbound << "]" << std::endl;
        std::cout << "Optimality Gap: " << optimizer.uncertainty() << std::endl;
    }

    try {
        if (!optimizer.complete()) {
            // there might be a timeout here...
            if (GOSDT::time > (float)Configuration::time_limit || !State::queue.empty()) {
                std::cout << "possible timeout: " << GOSDT::time << " " << Configuration::time_limit << " queue emtpy: "  << State::queue.empty() << std::endl;
                GOSDT::status = 2;
            } else {
                std::cout << "possible non-convergence: [" << lowerbound << " .. " << upperbound << "]" << std::endl;
                GOSDT::status = 1;
            }

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
            throw IntegrityViolation("No model","No model found - either user-provided upper bound assumption was too strong, or there was a false convergence");
        }

        if (Configuration::verbose) {
            std::cout << "Models Generated: " << models.size() << std::endl;
            if (optimizer.uncertainty() == 0.0 && models.size() > 0) {
                std::cout << "Loss: " << models.begin() -> loss() << std::endl;
                std::cout << "Complexity: " << models.begin() -> complexity() << std::endl;
            }
        }
        GOSDT::model_loss = models.begin() -> loss();
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
    } catch (IntegrityViolation exception) {
        GOSDT::status = 1;
        std::cout << exception.to_string() << std::endl;
    }
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
