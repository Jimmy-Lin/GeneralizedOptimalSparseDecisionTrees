#include "gosdt.hpp"

#define _DEBUG true
#define THROTTLE false

GOSDT::GOSDT(void) {
    configure(json::object());
}

GOSDT::GOSDT(std::istream & configuration) {
    configure(configuration);
}

GOSDT::~GOSDT(void) {
    return;
}

void GOSDT::configure(std::istream & configuration) {
    // The configuration is provided as an input stream
    // We need to first parse it into a JSON object before using it for configuration
    json config;
    configuration >> config;
    configure(config);
}

void GOSDT::configure(json configuration) {
    if (!configuration.contains("regularization")) { configuration["regularization"] = 0.1; }
    if (!configuration.contains("precision")) { configuration["precision"] = 0; }
    if (!configuration.contains("uncertainty_tolerance")) { configuration["uncertainty_tolerance"] = 0.0; }
    if (!configuration.contains("time_limit")) { configuration["time_limit"] = 300; }
    if (!configuration.contains("output_limit")) { configuration["output_limit"] = 10; }

    if (!configuration.contains("verbose")) { configuration["verbose"] = false; }

    if (!configuration.contains("optimism")) { configuration["optimism"] = 0.7; }
    if (!configuration.contains("equity")) { configuration["equity"] = 0.5; }
    if (!configuration.contains("sample_depth")) { configuration["sample_depth"] = 1; }
    if (!configuration.contains("similarity_threshold")) { configuration["similarity_threshold"] = 0.0; }
    if (!configuration.contains("workers")) { configuration["workers"] = std::thread::hardware_concurrency(); }

    this -> configuration = configuration;
}

std::string GOSDT::get_configuration(unsigned int indentation) const {
    return this -> configuration.dump(indentation);
}

bool const GOSDT::verbose(void) const {
    return this -> configuration["verbose"];
}

void GOSDT::work(int const id, Optimizer & optimizer, json configuration, int & return_reference) {
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 0;

    while (optimizer.complete() == false && optimizer.timeout() == false) {
        if (THROTTLE) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
        try {
            if (id == 0) { optimizer.tick(); }
            optimizer.iterate(id);
            ++iterations;
        } catch( const char * exception ) {
            std::cout << exception << std::endl;
            optimizer.diagnose_non_convergence();
            optimizer.diagnose_false_convergence();
            break;
        }
    }
    if (id == 0) { optimizer.tick(); }

    return_reference = iterations;
}

std::string GOSDT::fit(std::istream & data_source) {
    std::string result;
    json const & configuration = this -> configuration;

    if(verbose()) { std::cout << "Preprocessing Data" << std::endl; }
    // Create dataset object to deal with calculations over the training data
    Dataset dataset(data_source, configuration["precision"], configuration["regularization"], verbose());

    // // Initialize the similar support index which will get passed down recursively
    // dataset.initialize_similarity_index(configuration["similarity_threshold"]);
    // // Initialize the OpenCL kernel which can be used to accelerate data set calculations
    // if (configuration.contains("opencl_platform_index") && configuration.contains("opencl_device_index")) {
    //     dataset.initialize_kernel(configuration["opencl_platform_index"], configuration["opencl_device_index"]);
    // }

    if(verbose()) { std::cout << "Initializing Optimization Framework" << std::endl; }
    // Create optimizer object to track subproblem graph and execute graph updates
    Optimizer optimizer(dataset, configuration);

    if(verbose()) { std::cout << "Starting Optimization" << std::endl; }
    // Start measuring training time
    auto start = std::chrono::high_resolution_clock::now();

    // if(verbose()) { std::cout << "Starting Thread Pool" << std::endl; }
    // Create a pool of worker threads. Threads start running immediately after creation
    unsigned int pool_size = configuration["workers"];
    std::vector< std::thread > workers;
    std::vector< int > results(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        workers.emplace_back(work, i, std::ref(optimizer), configuration, std::ref(results[i]));

        #ifndef __APPLE__
            // If using Ubuntu Build, we can pin each thread to a specific CPU core to improve cache locality
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(i, &cpuset);
            int error = pthread_setaffinity_np(workers[i].native_handle(), sizeof(cpu_set_t), &cpuset);
            if (error != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << error << std::endl; }
        #endif
    }
    // Wait for the thread pool to terminate
    for (auto iterator = workers.begin(); iterator != workers.end(); ++iterator) {
        std::thread & worker = * iterator;
        worker.join();
    }

    // Stop measuring training time
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
    float execution_time = duration.count() / 1000.0;
    if(verbose()) { std::cout << "Optimization Complete" << std::endl; }

    if (configuration.contains("timing_output")) {
        std::ofstream timing_output(configuration["timing_output"], std::ios_base::app);
        timing_output << execution_time;
        timing_output.flush();
        timing_output.close();
    }

    if(verbose()) {
        std::cout << "Training Duration: " << execution_time << " seconds" << std::endl;
        int iterations = 0;
        for (auto iterator = results.begin(); iterator != results.end(); ++iterator) {
            iterations += * iterator;
        }
        std::cout << "Training Iterations: " << iterations << " iterations" << std::endl;
        std::cout << "Size of Explored Search Space: " << optimizer.size() << " problems" << std::endl;
        std::tuple< float, float > objective_boundary = optimizer.objective_boundary();
        std::cout << "Objective Boundary: [" << std::get<0>(objective_boundary) << ", " << std::get<1>(objective_boundary) << "]" << std::endl;
        std::cout << "Optimality Gap: " << optimizer.uncertainty() << std::endl;
    }

    try{

        std::unordered_set< Model > const & models = optimizer.models(configuration["output_limit"]);

        if(verbose()) { std::cout << "Models Generated: " << models.size() << std::endl; }

        json output = json::array();
        for (auto iterator = models.begin(); iterator != models.end(); ++iterator) {
            Model model = * iterator;
            output.push_back(model.to_json());
        }
        result = output.dump(16);

        if (configuration.contains("output")) {
            if(verbose()) { std::cout << "Storing Models in: " << configuration["output"] << std::endl; }
            std::ofstream out(configuration["output"]);
            out << result;
            out.close();
        }

        return result;
    } catch(const char * exception) {
        std::cout << exception << std::endl;
        result = std::string(exception);
        return result;
    }
}