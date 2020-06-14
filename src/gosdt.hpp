#ifndef GOSDT_H
#define GOSDT_H

#include <iostream>

#include <thread>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <chrono>

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <string>

#include <json/json.hpp>

#include "encoder.hpp"
#include "dataset.hpp"
#include "model.hpp"
#include "optimizer.hpp"

using json = nlohmann::json;

class GOSDT {
    public:
        GOSDT(void);
        GOSDT(std::istream & configuration);
        ~GOSDT(void);

        bool const verbose(void) const;
        void configure(std::istream & configuration);
        void configure(json configuration);
        std::string get_configuration(unsigned int indentation) const;

        std::string fit(std::istream & data_source);
    private:
        json configuration;
        static void work(int const id, Optimizer & optimizer, json configuration, int & return_reference);
};

#endif
