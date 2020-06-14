from lib.data_structures.interval import Interval

class Objective: 
    def __init__(self, name, P, N, lamb, w = None, theta = None): 
        self.name = name
        self.P = P
        self.N = N
        self.lamb=lamb
        self.w = w
        self.theta = theta

    def evaluate(self, confusion):
        (tp, tn, fp, fn) = confusion
        if self.name == 'acc':
            return self.accuracy(fp, fn)
        elif self.name == 'bacc':
            return self.balanced_accuracy(fp, fn)
        elif self.name == 'wacc':
            return self.weighted_accuracy(fp, fn)
        # elif self.name == 'f1':
        #     loss = self.f1_loss(FP, FN)
        # elif self.name == 'auc_convex':
        #     _, loss = self.ach_loss(leaves)
        # elif self.name == 'partial_auc':
        #     _, loss = self.pach_loss(leaves, theta)
        else:
            raise "Unimplemented objective type '#{self.name}'"
    
    # def leaf_predict(self, p, n, w=None):
    #     predict = 1
    #     if self.name == 'acc':
    #         if p<n:
    #             predict = 0
    #     elif self.name == 'bacc':
    #         if p/self.P <= n/self.N:
    #             predict = 0
    #     elif self.name == 'wacc':
    #         if p/(p+n) <= 1/(1+self.w):
    #             predict = 0
    #     elif self.name == 'f1':
    #         if self.w*p <= n:
    #             predict = 0      
    #     return predict
    
    # def loss(self, FP, FN, w=None, leaves=None, theta=None):
    #     if self.name == 'acc':
    #         loss = self.acc_loss(FP, FN)
    #     elif self.name == 'bacc':
    #         loss = self.bacc_loss(FP, FN)
    #     elif self.name == 'wacc':
    #         loss = self.wacc_loss(FP, FN, w)
    #     elif self.name == 'f1':
    #         loss = self.f1_loss(FP, FN)
    #     elif self.name == 'auc_convex':
    #         _, loss = self.ach_loss(leaves)
    #     elif self.name == 'partial_auc':
    #         _, loss = self.pach_loss(leaves, theta)
    #     return loss
    
    # def risk(self, leaves, w=None, theta=None):
    #     FP, FN = get_false(leaves)
    #     risk = self.loss(FP, FN, w, leaves, theta) +  self.lamb*len(leaves)
    #     return risk
    
    # def lower_bound(self, leaves, splitleaf, w=None, theta=None):
    #     FP, FN = get_fixed_false(leaves, splitleaf)
    #     if self.name == 'acc':
    #         loss = self.acc_loss(FP, FN)
    #     elif self.name == 'bacc':
    #         loss = self.bacc_loss(FP, FN)
    #     elif self.name == 'wacc':
    #         loss = self.wacc_loss(FP, FN, w)
    #     elif self.name == 'f1':
    #         loss = self.f1_loss(FP, FN)
    #     elif self.name == 'auc_convex':
    #         ordered_fixed = order_fixed_leaves(leaves, splitleaf, len(leaves))
    #         if len(ordered_fixed) == 0:
    #             loss = 0
    #         else:
    #             Ps, Ns = get_split(leaves, splitleaf)
    #             loss = self.ach_lb(ordered_fixed, Ps, Ns)
    #     elif self.name == 'partial_auc':
    #         ordered_fixed = order_fixed_leaves(leaves, splitleaf, len(leaves))
    #         if len(ordered_fixed) == 0:
    #             loss = 1-theta
    #         else:
    #             Ps, Ns = get_split(leaves, splitleaf)
    #             loss = self.pach_lb(ordered_fixed, theta, Ps, Ns)
    #     lb = loss + self.lamb*len(leaves)
    #     return lb
     
    
    # def incre_accu_bound(self, removed_l, new_l1, new_l2, unchanged_leaves=None, 
    #                      removed_leaves=None, w=None, FPu=None, FNu=None, theta=None, ld=None):
    #     if self.name == 'acc':
    #         a = self.acc_loss(removed_l.fp-new_l1.fp-new_l2.fp, 
    #                           removed_l.fn-new_l1.fn-new_l2.fn)
    #     elif self.name == 'bacc':
    #         a = self.bacc_loss(removed_l.fp-new_l1.fp-new_l2.fp, 
    #                           removed_l.fn-new_l1.fn-new_l2.fn)
    #     elif self.name == 'wacc':
    #         a = self.wacc_loss(removed_l.fp-new_l1.fp-new_l2.fp, 
    #                            removed_l.fn-new_l1.fn-new_l2.fn, w)
    #     elif self.name == 'f1':
    #         a = self.f1_loss(FPu+removed_l.fp, FNu+removed_l.fn) - \
    #             self.f1_loss(FPu+new_l1.fp+new_l2.fp, FNu+new_l1.fn+new_l2.fn)
    #     elif self.name == 'auc_convex':
    #         removed = removed_leaves.copy()
    #         removed.remove(removed_l)
    #         _, ld_new = self.ach_loss(unchanged_leaves+removed+\
    #                                   [new_l1]+[new_l2])
    #         a = ld-ld_new
    #     elif self.name == 'partial_auc':
    #         removed = removed_leaves.copy()
    #         removed.remove(removed_l)
    #         _, ld_new = self.pach_loss(unchanged_leaves+removed+\
    #                                    [new_l1]+[new_l2], theta)
    #         a = ld-ld_new
    #     return a
    
    # def leaf_support_bound(self, leaf, w=None, theta=None, 
    #                        FP=None, FN=None, ld=None, ordered_leaves=None):
    #     if self.name == 'acc':
    #         tau = self.acc_loss(leaf.fp, leaf.fn)
    #     elif self.name == 'bacc':
    #         tau = self.bacc_loss(leaf.fp, leaf.fn)
    #     elif self.name == 'wacc':
    #         tau = self.wacc_loss(leaf.fp, leaf.fn, w)
    #     elif self.name == 'f1':
    #         tau = ld - self.f1_loss(FP-leaf.fp, FN-leaf.fn)
    #     elif self.name == 'auc_convex':
    #         if leaf.p == 0 or leaf.n == 0:
    #             tau = -1
    #         else:
    #             leaves = ordered_leaves.copy()
    #             leaves.remove(leaf)
    #             ld_new = self.ach_lb(leaves, leaf.p, leaf.n)
    #             tau = ld-ld_new
    #     elif self.name == 'partial_auc':
    #         if leaf.p == 0 or leaf.n == 0:
    #             tau = -1
    #             ld_new = ld
    #         else:
    #             leaves = ordered_leaves.copy()
    #             leaves.remove(leaf)
    #             ld_new = self.pach_lb(leaves, theta, leaf.p, leaf.n)
    #             tau = ld-ld_new
    #     #print('leaf split feature', leaf.rules)
    #     #print('tau:',round(tau,4), 'ld:', round(ld,4), 'ld_new:', round(ld_new, 4))
    #     return tau
    
    def accuracy(self, FP, FN):
        lowerbound = (FP.lowerbound + FN.lowerbound) / (self.P + self.N)
        upperbound = min(FP.upperbound, FN.upperbound) / (self.P + self.N)
        return Interval(lowerbound, upperbound), (0 if FP.upperbound >= FN.upperbound else 1)
    
    def balanced_accuracy(self, FP, FN):
        lowerbound = 0.5 * FN.lowerbound / self.P + 0.5 * FP.lowerbound / self.N
        upperbound = 0.5 * min(FN.upperbound / self.P, FP.upperbound / self.N)
        if 0.5 * FP.lowerbound / self.N >= 0.5 * FN.lowerbound / self.P:
            prediction = 0
        else:
            prediction = 1
        return Interval(lowerbound, upperbound), prediction
    
    def weighted_accuracy(self, FP, FN, w):
        lowerbound = (self.w * FN.lowerbound + FP.lowerbound) / (self.w * self.P + self.N)
        upperbound = min(self.w * FN.upperbound + FP.upperbound)  / (self.w * self.P + self.N)
        if FP.lowerbound / self.N >= sefl.w * FN.lowerbound / self.P:
            prediction = 0
        else:
            prediction = 1
        return Interval(lowerbound, upperbound), prediction
    
    # def f1_score(self, FP, FN): # I don't think we can actually implement it this way?
    #     lowerbound = (FP.lowerbound + FN.lowerbound) / (2 * self.P + FP.upperbound - FN.upperbound)
    #     upperbound = min( FP.upperbound / (2 * self.P + FP.upperbound), FN.lowerbound / (2 * self.P + FN.upperbound) )
    #     if FP.upperbound / (2 * self.P + FP.upperbound) >= FN.lowerbound / (2 * self.P + FN.upperbound):
    #         prediction = 0
    #     else:
    #         prediction = 1
    #     return Interval(lowerbound, upperbound), prediction
    
    # def aucch(self, leaves):
    #     # This is kind of tricky... we don't actually have precisely defined leaves to sort
    #     # We don't even know how many leaves there will be?
    #     # The best I can think of is a trivial lowerbound of 0 and an upperbound of the single best leaf
    #     ordered_leaves = order_leaves(leaves)
    #     tp = fp = np.array([0])
    #     if len(leaves) > 1:
    #         for i in range(0, len(leaves)):
    #             tp = np.append(tp, tp[i]+ordered_leaves[i].p)
    #             fp = np.append(fp, fp[i]+ordered_leaves[i].n)
    #     else:
    #         tp = np.append(tp, self.P)
    #         fp = np.append(fp, self.N)
    
    #     loss = 1-0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) \
    #                       for i in range(1,len(tp))])
    #     # return ordered_leaves, loss

    #     return 0, 1, 0
    
    # def paucch(self, leaves, theta):
    #     ordered_leaves = order_leaves(leaves)
    #     tp = fp = np.array([0], dtype=float)
    #     if len(leaves) > 1:
    #         i = 0
    #         while fp[i] < self.N*theta and i < len(leaves):
    #             tp = np.append(tp, tp[i]+ordered_leaves[i].p)
    #             fp = np.append(fp, fp[i]+ordered_leaves[i].n)
    #             i += 1
    #         tp[i] = ((tp[i]-tp[i-1])/(fp[i]-fp[i-1]))*(self.N*self.theta-fp[i])+tp[i]
    #         fp[i] = self.N*self.theta
    #     else:
    #         tp = np.append(tp, self.P*self.theta)
    #         fp = np.append(fp, self.N*self.theta)
    #     loss = 1-0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) \
    #                       for i in range(1,len(tp))])
    #     # return ordered_leaves, loss

    #     return 0, self.theta, 0
    
    # def ach_lb(self, leaves, Ps, Ns):
    #     tp = np.array([0, Ps], dtype=float)
    #     fp = np.array([0,0], dtype=float)
    #     for i in range(len(leaves)):
    #         tp = np.append(tp, tp[i+1]+leaves[i].p)
    #         fp = np.append(fp, fp[i+1]+leaves[i].n)
    #     tp = np.append(tp, tp[len(leaves)+1]+0)
    #     fp = np.append(fp, fp[len(leaves)+1]+Ns)
    #     loss = 1- 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) for i in range(1,len(tp))])
        
    #     return loss
    
    # def pach_lb(self, leaves, theta, Ps, Ns):
    #     tp = np.array([0, Ps], dtype=float)
    #     fp = np.array([0, 0], dtype=float)
    #     i = 0
    #     while fp[i+1] < self.N*theta:
    #         if i < len(leaves):
    #             tp = np.append(tp, tp[i+1]+leaves[i].p)
    #             fp = np.append(fp, fp[i+1]+leaves[i].n)
    #             i += 1
    #         else:
    #             tp = np.append(tp, tp[i+1]+0)
    #             fp = np.append(fp, fp[i+1]+Ns)
    #             break                         
    #     tp[len(tp)-1] = ((tp[len(tp)-1]-tp[len(tp)-2])/(fp[len(fp)-1]-fp[len(fp)-2]))*\
    #                     (self.N*theta-fp[len(fp)-1])+tp[len(tp)-1]
    #     fp[len(fp)-1] = self.N*theta
    #     loss = 1- 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) \
    #            for i in range(1,len(tp))])
    #     return loss
    