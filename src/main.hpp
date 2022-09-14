#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
//#include <sys/poll.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include <json/json.hpp>
#include <csv/csv.h>

#include "configuration.hpp"
#include "gosdt.hpp"

using json = nlohmann::json;

// Main entry point for the CLI
// The data set is entered either through the standard input stream or as a file path in the first argument
// The configuration file is entered as a file path in the first argument if the data set is entered through standard input
//   otherwise the configuration file is entered as a file path in the second argument
// Example:
// cat data.csv | gosdt config.json
// or
// gosdt data.csv config.json
int main(int argc, char *argv[]);

#endif