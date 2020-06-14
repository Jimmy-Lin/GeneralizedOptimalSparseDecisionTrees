#include "../src/dataset.hpp"

int test_dataset(void) {
    int failures = 0;

    std::ifstream in("test/fixtures/dataset.csv");
    Dataset dataset(in, 0.0, false);
    in.close();

    // Create indicator
    bitset indicator(5);
    indicator[3] = 1;
    indicator[4] = 1;

    // Test Datase::shape
    int height = dataset.height();
    int width = dataset.width();
    int depth = dataset.depth();
    int size = dataset.size();
    failures += expect(5, height, "Test Dataset Height Failed.");
    failures += expect(5, width, "Test Dataset Width Failed.");
    failures += expect(4, depth, "Test Dataset Depth Failed.");
    failures += expect(10, size, "Test Dataset Size Failed.");

    // Test Dataset::impurity()
    // lowerbound, upperbound, support, stump
    failures += expect(0.1, dataset.loss_lowerbound(), "Test Dataset Lower Bound Failed.");
    failures += expect(0.6, std::get<1>(dataset.leaf()), "Test Dataset Upper Bound Failed.");
    failures += expect(1.0, dataset.support(), "Test Dataset Support Failed.");
    failures += expect(0.6, std::get<1>(dataset.leaf()), "Test Dataset Stump Loss Failed.");

    // Test Dataset::impurity(indicator)
    failures += expect(0.1, dataset.loss_lowerbound(indicator), "Test Dataset Subset Lower Bound Failed.");
    failures += expect(0.2, std::get<1>(dataset.leaf(indicator)), "Test Dataset Subset Upper Bound Failed.");
    failures += expect(0.4, dataset.support(indicator), "Test Dataset Subset Support Failed.");
    failures += expect(0.2, std::get<1>(dataset.leaf(indicator)), "Test Dataset Subset Stump Loss Failed.");

    // Test Dataset::partition(feature_index)
    auto partition = dataset.partition(0);
    failures += expect("11110", partition[0].to_string(), "Test Dataset Partition(0) negative Failed.");
    failures += expect("00001", partition[1].to_string(), "Test Dataset Partition(0) positive Failed.");

    partition = dataset.partition(1);
    failures += expect("11101", partition[0].to_string(), "Test Dataset Partition(1) negative Failed.");
    failures += expect("00010", partition[1].to_string(), "Test Dataset Partition(1) positive Failed.");

    partition = dataset.partition(2);
    failures += expect("11011", partition[0].to_string(), "Test Dataset Partition(2) negative Failed.");
    failures += expect("00100", partition[1].to_string(), "Test Dataset Partition(2) positive Failed.");

    partition = dataset.partition(3);
    failures += expect("10111", partition[0].to_string(), "Test Dataset Partition(3) negative Failed.");
    failures += expect("01000", partition[1].to_string(), "Test Dataset Partition(3) positive Failed.");

    partition = dataset.partition(4);
    failures += expect("01111", partition[0].to_string(), "Test Dataset Partition(4) negative Failed.");
    failures += expect("10000", partition[1].to_string(), "Test Dataset Partition(4) positive Failed.");

    // Test Dataset::partition(indicaor, feature_index)
    partition = dataset.partition(indicator, 0);
    failures += expect("11000", partition[0].to_string(), "Test Dataset Subset Partition(0) negative Failed.");
    failures += expect("00000", partition[1].to_string(), "Test Dataset Subset Partition(0) positive Failed.");

    partition = dataset.partition(indicator, 1);
    failures += expect("11000", partition[0].to_string(), "Test Dataset Subset Partition(1) negative Failed.");
    failures += expect("00000", partition[1].to_string(), "Test Dataset Subset Partition(1) positive Failed.");

    partition = dataset.partition(indicator, 2);
    failures += expect("11000", partition[0].to_string(), "Test Dataset Subset Partition(2) negative Failed.");
    failures += expect("00000", partition[1].to_string(), "Test Dataset Subset Partition(2) positive Failed.");

    partition = dataset.partition(indicator, 3);
    failures += expect("10000", partition[0].to_string(), "Test Dataset Subset Partition(3) negative Failed.");
    failures += expect("01000", partition[1].to_string(), "Test Dataset Subset Partition(3) positive Failed.");

    partition = dataset.partition(indicator, 4);
    failures += expect("01000", partition[0].to_string(), "Test Dataset Subset Partition(4) negative Failed.");
    failures += expect("10000", partition[1].to_string(), "Test Dataset Subset Partition(4) positive Failed.");

    return failures;
}