# Frequently Asked Questions
- [Does Gosdt (implicitly) restrict the depth of the resulting tree?](##depth)
- [Why does GOSDT run for a long time when the regularization parameter (lambda) is set to zero?](##long_run)
- [Is there a way to limit the size of produced tree?](##limit_tree_size)
- [In general, how does GOSDT set the regularization parameter?](##set_lambda)

---

## Does Gosdt (implicitly) restrict the depth of the resulting tree? 

New as of 2022, GOSDT With Guesses can now restrict the depth of the resulting tree both implicitly and explicitly. Our primary sparsity constraint is the regularization parameter (lambda) which is used to penalize the number of leaves. As lambda becomes smaller, the generated trees will have more leaves, but the number of leaves doesn't guarantee what depth a tree has since GOSDT generates trees of any shape. As of our 2022 AAAI paper, though, we've allowed users to restrict the depth of the tree. This provides more control over the tree's shape and reduces the runtime. However, the depth constraint is not a substitute for having a nonzero regularization parameter! Our algorithm achieves a better sparsity-accuracy tradeoff, and saves time, with a well-chosen lambda. 

## Why does GOSDT run for a long time when the regularization parameter (lambda) is set to zero?

The running time depends on the dataset itself and the regularization parameter (lambda). In general, setting lambda to 0 will make the running time longer. Setting lambda to 0 is kind of deactivating the branch-and-bound in GOSDT. In other words, we are kind of using brute force to search over the whole space without effective pruning, though dynamic programming can help for computational reuse. 
In GOSDT, we compare the difference between the upper and lower bound of a subproblem with lambda to determine whether this subproblem needs to be further split. If lambda=0, we can always split a subproblem. Therefore, it will take more time to run. It usually does not make sense to set lambda smaller than 1/n, where n is the number of samples.

## Is there a way to limit the size of the produced tree?

Regularization parameter (lambda) is used to limit the size of the produced tree (specifically, in GOSDT, it limits the number of leaves of the produced tree). We usually set lambda to [0.1, 0.05, 0.01, 0.005, 0.001], but the value really depends on the dataset. One thing that might be helpful is considering how many samples should be captured by each leaf node. Suppose you want each leaf node to contain at least 10 samples. Then setting the regularization parameter to 10/n is reasonable. In general, the larger the value of lambda is, the sparser a tree you will get.


## In general, how does GOSDT set the regularization parameter? 

GOSDT aims to find an optimal tree that minimizes the training loss with a penalty on the number of leaves. The mathematical description is min loss+lambda*# of leaves. When we run GOSDT, we usually set lambda to different non-zero values and usually not smaller than 1/n. On page 31 Appendix I.6 in [our ICML paper](#https://arxiv.org/abs/2006.08690), we provide detailed information about the configuration we used to run accuracy vs. sparsity experiments.  
