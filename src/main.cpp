#include "main.hpp"

int main(int argc, char *argv[]) {
	// Check program input
	if (argc < 2 || argc > 3) {
		std::cout << "Usage: gosdt [path to feature set] ?[path to config]" << std::endl;
		return 0;
	}
	if (!std::ifstream(argv[1]).good()) {
		std::cout << "File Not Found: " << argv[1] << std::endl;
		return 1;
	}
	if (argc >= 3 && !std::ifstream(argv[2]).good()) {
		std::cout << "File Not Found: " << argv[2] << std::endl;
		return 1;
	}

	// Initialize the library interface
	GOSDT model;
	if (argc == 2) {
		model = GOSDT();
	} else if (argc == 3) {
		// Use custom configuration if provided
		std::ifstream configuration(argv[2]);
		model = GOSDT(configuration);
	}

	// Print messages to help user ensure they've provided the correct inputs
	if (model.verbose()) {
		std::cout << "Generalized Optimal Sparse Decision Tree" << std::endl;
		std::cout << "Using data set: " << argv[1] << std::endl;
		std::cout << "Using configuration: " << model.get_configuration(2) << std::endl;
	}
	
	// Compute and print results
	std::ifstream data(argv[1]);
	std::string result = model.fit(data);
	if (model.verbose()) {
		std::cout << result << std::endl;
	}
	return 0;
}
