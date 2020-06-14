#include "../src/task.hpp"

int test_task(void) {
    int failures = 0;
    bool throws;

    Task task(0.21, 0.32, 0.5, 0.76);

    failures += expect(0.21, task.lowerbound(), "Test Task::lowerbound fails equality with constructor argument.");
    failures += expect(0.32, task.upperbound(), "Test Task::upperbound fails equality with constructor argument.");
    failures += expect(0.5, task.support(), "Test Task::support fails equality with constructor argument.");
    failures += expect(0.11, task.uncertainty(), "Test Task::uncertainty fails equality with constructor argument.");
    failures += expect(0.11, task.potential(), "Test Task::potential fails equality with constructor argument.");
    failures += expect(0.76, task.base_objective(), "Test Task::potential fails equality with constructor argument.");

    throws = false;
    try { task.objective(); } catch (const char * e) { throws = true; }
    failures += expect(true, throws, "Test Task::objective doesn't throw on invalid objective query.");

    task.inform(0.22, 0.31);
    failures += expect(0.22, task.lowerbound(), "Test Task::inform fails to update lowerbound.");
    failures += expect(0.31, task.upperbound(), "Test Task::inform fails to update upperbound.");
    failures += expect(0.09, task.uncertainty(), "Test Task::uncertainty fails to reflect updated state.");
    failures += expect(0.11, task.potential(), "Test Task::potential fails remain constant.");

    throws = false;
    try { task.inform(0.88, 0.31); } catch (const char * e) { throws = true; }
    failures += expect(true, throws, "Test Task::inform doesn't throw on invalid update.");

    return failures;

}