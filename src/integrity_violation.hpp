#ifndef INTEGRITY_VIOLATION_H
#define INTEGRITY_VIOLATION_H

#include <stdexcept>
#include <string>
#include <sstream>

// Implementation of run time error for displaying any detected integrity violations during the algorithm
// These exceptions indicate that a logical error in the code has caused the algorithm to reach an incorrect state
// The correct response to an integrity violation is to report any diagnosis, then terminate the program.
class IntegrityViolation : public std::runtime_error {
public:
    IntegrityViolation(std::string error, std::string reason) : std::runtime_error(error), error(error), reason(reason) {}
    std::string error;
    std::string reason;
    std::string to_string(void) const {
        std::stringstream message;
        message << "\033[1;31mIntegrityViolation Detected during Optimization:\n"
                << "  ErrorContext: " << this -> error << "\n"
                << "  Reason: " << this -> reason << "\033[0m" << std::endl;
        return message.str();
    }
};

#endif
