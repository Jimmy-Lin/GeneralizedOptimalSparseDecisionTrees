#ifndef GOSDT_H
#define GOSDT_H

#define SIMDPP_ARCH_X86_SSE4_1

#include <iostream>

#include <thread>
//#include <pthread.h>
//#include <sched.h>
//#include <unistd.h>
#include <chrono>

//#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <string>

//#include <alloca.h>

#include <json/json.hpp>

#include "encoder.hpp"
#include "dataset.hpp"
#include "integrity_violation.hpp"
#include "model.hpp"
#include "optimizer.hpp"

using json = nlohmann::json;

// The main interface of the library
// Note that the algorithm behaviour is modified using the static configuration object using the Configuration class
class GOSDT {
    public:
        GOSDT(void);
        ~GOSDT(void);

        static float time;
        static unsigned int size;
        static unsigned int iterations;
        static unsigned int status;
        static double lower_bound;
        static double upper_bound;
        static float model_loss; //loss of tree(s) returned

        static float  ru_utime;         /* user CPU time used */
        static float  ru_stime;         /* system CPU time used */
        static long   ru_maxrss;        /* maximum resident set size in KB */
        static long   ru_nswap;         /* swaps */
        static long   ru_nivcsw;        /* involuntary context switches */

        // @param config_source: string stream containing a JSON object of configuration parameters
        // @note: See the Configuration class for details about each parameter
        static void configure(std::istream & config_source);

        // @require: The CSV must contain a header.
        // @require: Scientific notation is currently not supported by the parser, use long form decimal notation
        // @require: All rows must have the same number of entries
        // @require: all entries are comma-separated
        // @require: Wrapping quotations are not stripped
        // @param data_source: string containing a CSV of training_data
        // @modifies result: Contains a JSON array of all optimal models extracted
        void fit(std::istream & data_source, std::string & result);

        // @require: The CSV must contain a header.
        // @require: Scientific notation is currently not supported by the parser, use long form decimal notation
        // @require: All rows must have the same number of entries
        // @require: all entries are comma-separated
        // @require: Wrapping quotations are not stripped
        // @param data_source: string containing a CSV of training_data
        // @modifies models: Set of models extracted from the optimization
        void fit(std::istream & data_source, std::unordered_set< Model > & models);
    private:
        // @param id: The worker ID of the current thread
        // @param optimizer: optimizer object which will assign work to the thread
        // @modifies return_reference: reference for returning values to the main thread
        static void work(int const id, Optimizer & optimizer, int & return_reference);
};

#endif
