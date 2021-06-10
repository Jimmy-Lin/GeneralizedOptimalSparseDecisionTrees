#include "configuration.hpp"

float Configuration::uncertainty_tolerance = 0.0;
float Configuration::regularization = 0.05;
float Configuration::upperbound = 0.0;

unsigned int Configuration::time_limit = 0;
unsigned int Configuration::worker_limit = 1;
unsigned int Configuration::stack_limit = 0;
unsigned int Configuration::precision_limit = 0;
unsigned int Configuration::model_limit = 1;

bool Configuration::verbose = false;
bool Configuration::diagnostics = false;

unsigned char Configuration::depth_budget = 0;

bool Configuration::balance = false;
bool Configuration::look_ahead = true;
bool Configuration::similar_support = true;
bool Configuration::cancellation = true;
bool Configuration::continuous_feature_exchange = true;
bool Configuration::feature_exchange = true;
bool Configuration::feature_transform = true;
bool Configuration::rule_list = false;
bool Configuration::non_binary = false;

std::string Configuration::costs = "";
std::string Configuration::model = "";
std::string Configuration::timing = "";
std::string Configuration::trace = "";
std::string Configuration::tree = "";
std::string Configuration::profile = "";

void Configuration::configure(std::istream & source) {
    json config;
    source >> config;
    Configuration::configure(config);
};

void Configuration::configure(json config) {
    if (config.contains("uncertainty_tolerance")) { Configuration::uncertainty_tolerance = config["uncertainty_tolerance"]; }
    if (config.contains("regularization")) { Configuration::regularization = config["regularization"]; }
    if (config.contains("upperbound")) { Configuration::upperbound = config["upperbound"]; }

    if (config.contains("time_limit")) { Configuration::time_limit = config["time_limit"]; }
    if (config.contains("worker_limit")) { Configuration::worker_limit = config["worker_limit"]; }
    if (config.contains("stack_limit")) { Configuration::stack_limit = config["stack_limit"]; }
    if (config.contains("precision_limit")) { Configuration::precision_limit = config["precision_limit"]; }
    if (config.contains("model_limit")) { Configuration::model_limit = config["model_limit"]; }

    if (config.contains("verbose")) { Configuration::verbose = config["verbose"]; }
    if (config.contains("diagnostics")) { Configuration::diagnostics = config["diagnostics"]; }

    if (config.contains("depth_budget")) { Configuration::depth_budget = config["depth_budget"]; }

    if (config.contains("balance")) { Configuration::balance = config["balance"]; }
    if (config.contains("look_ahead")) { Configuration::look_ahead = config["look_ahead"]; }
    if (config.contains("similar_support")) { Configuration::similar_support = config["similar_support"]; }
    if (config.contains("cancellation")) { Configuration::cancellation = config["cancellation"]; }
    if (config.contains("continuous_feature_exchange")) { Configuration::continuous_feature_exchange = config["continuous_feature_exchange"]; }
    if (config.contains("feature_exchange")) { Configuration::feature_exchange = config["feature_exchange"]; }
    if (config.contains("feature_transform")) { Configuration::feature_transform = config["feature_transform"]; }
    if (config.contains("rule_list")) { Configuration::rule_list = config["rule_list"]; }
    if (config.contains("non_binary")) { Configuration::non_binary = config["non_binary"]; }

    if (config.contains("costs")) { Configuration::costs = config["costs"]; }
    if (config.contains("model")) { Configuration::model = config["model"]; }
    if (config.contains("timing")) { Configuration::timing = config["timing"]; }
    if (config.contains("trace")) { Configuration::trace = config["trace"]; }
    if (config.contains("tree")) { Configuration::tree = config["tree"]; }
    if (config.contains("profile")) { Configuration::profile = config["profile"]; }
}

std::string Configuration::to_string(unsigned int spacing) {
    json obj = json::object();
    obj["uncertainty_tolerance"] = Configuration::uncertainty_tolerance;
    obj["regularization"] = Configuration::regularization;
    obj["upperbound"] = Configuration::upperbound;

    obj["time_limit"] = Configuration::time_limit;
    obj["worker_limit"] = Configuration::worker_limit;
    obj["stack_limit"] = Configuration::stack_limit;
    obj["precision_limit"] = Configuration::precision_limit;
    obj["model_limit"] = Configuration::model_limit;

    obj["verbose"] = Configuration::verbose;
    obj["diagnostics"] = Configuration::diagnostics;

    obj["depth_budget"] = Configuration::depth_budget;

    obj["balance"] = Configuration::balance;
    obj["look_ahead"] = Configuration::look_ahead;
    obj["similar_support"] = Configuration::similar_support;
    obj["cancellation"] = Configuration::cancellation;
    obj["continuous_feature_exchange"] = Configuration::continuous_feature_exchange;
    obj["feature_exchange"] = Configuration::feature_exchange;
    obj["feature_transform"] = Configuration::feature_transform;
    obj["rule_list"] = Configuration::rule_list;
    obj["non_binary"] = Configuration::non_binary;

    obj["costs"] = Configuration::costs;
    obj["model"] = Configuration::model;
    obj["timing"] = Configuration::timing;
    obj["trace"] = Configuration::trace;
    obj["tree"] = Configuration::tree;
    obj["profile"] = Configuration::profile;
    return obj.dump(spacing);
}
