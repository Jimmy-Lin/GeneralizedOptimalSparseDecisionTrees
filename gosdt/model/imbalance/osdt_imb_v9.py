import numpy as np
import heapq
import math
import time
import copy
import sklearn.tree
import sklearn.metrics
from itertools import product, compress
from gmpy2 import mpz
from matplotlib import pyplot as plt
import pickle

from gosdt.model.imbalance.rule import make_all_ones, rule_vand, rule_vectompz, count_ones, rule_mpztovec
from gosdt.model.imbalance.osdt_sup import log, gini_reduction, get_code, cart, get_z

class Objective: 
    def __init__(self, name, P, N, lamb): 
        self.name = name
        self.P = P
        self.N = N
        self.lamb=lamb
    
    def leaf_predict(self, p, n, w=None):
        predict = 1
        if self.name == 'acc':
            if p<n:
                predict = 0
        elif self.name == 'bacc':
            if p/self.P <= n/self.N:
                predict = 0
        elif self.name == 'wacc':
            if w*p <= n:
                predict = 0
        elif self.name == 'f1':
            if w*p <= n:
                predict = 0      
        return predict
    
    def loss(self, FP, FN, w=None, leaves=None, theta=None):
        if self.name == 'acc':
            loss = self.acc_loss(FP, FN)
        elif self.name == 'bacc':
            loss = self.bacc_loss(FP, FN)
        elif self.name == 'wacc':
            loss = self.wacc_loss(FP, FN, w)
        elif self.name == 'f1':
            loss = self.f1_loss(FP, FN)
        elif self.name == 'auc':
            _, loss = self.ach_loss(leaves)
        elif self.name == 'pauc':
            _, loss = self.pach_loss(leaves, theta)
        return loss
    
    def risk(self, leaves, w=None, theta=None):
        FP, FN = get_false(leaves)
        risk = self.loss(FP, FN, w, leaves, theta) +  self.lamb*len(leaves)
        return risk
    
    def lower_bound(self, leaves, splitleaf, w=None, theta=None):
        FP, FN = get_fixed_false(leaves, splitleaf)
        if self.name == 'acc':
            loss = self.acc_loss(FP, FN)
        elif self.name == 'bacc':
            loss = self.bacc_loss(FP, FN)
        elif self.name == 'wacc':
            loss = self.wacc_loss(FP, FN, w)
        elif self.name == 'f1':
            loss = self.f1_loss(FP, FN)
        elif self.name == 'auc':
            ordered_fixed = order_fixed_leaves(leaves, splitleaf)
            if len(ordered_fixed) == 0:
                loss = 0
            else:
                Ps, Ns = get_split(leaves, splitleaf)
                loss = self.ach_lb(ordered_fixed, Ps, Ns)
        elif self.name == 'pauc':
            ordered_fixed = order_fixed_leaves(leaves, splitleaf)
            if len(ordered_fixed) == 0:
                loss = 1-theta
            else:
                Ps, Ns = get_split(leaves, splitleaf)
                loss = self.pach_lb(ordered_fixed, theta, Ps, Ns)
        lb = loss + self.lamb*len(leaves)
        return lb
     
    
    def incre_accu_bound(self, removed_l, new_l1, new_l2, unchanged_leaves=None, 
                         removed_leaves=None, w=None, FPu=None, FNu=None, theta=None, ld=None):
        if self.name == 'acc':
            a = self.acc_loss(removed_l.fp-new_l1.fp-new_l2.fp, 
                              removed_l.fn-new_l1.fn-new_l2.fn)
        elif self.name == 'bacc':
            a = self.bacc_loss(removed_l.fp-new_l1.fp-new_l2.fp, 
                              removed_l.fn-new_l1.fn-new_l2.fn)
        elif self.name == 'wacc':
            a = self.wacc_loss(removed_l.fp-new_l1.fp-new_l2.fp, 
                               removed_l.fn-new_l1.fn-new_l2.fn, w)
        elif self.name == 'f1':
            a = self.f1_loss(FPu+removed_l.fp, FNu+removed_l.fn) - \
                self.f1_loss(FPu+new_l1.fp+new_l2.fp, FNu+new_l1.fn+new_l2.fn)
        elif self.name == 'auc':
            removed = removed_leaves.copy()
            removed.remove(removed_l)
            _, ld_new = self.ach_loss(unchanged_leaves+removed+\
                                      [new_l1]+[new_l2])
            a = ld-ld_new
        elif self.name == 'pauc':
            removed = removed_leaves.copy()
            removed.remove(removed_l)
            _, ld_new = self.pach_loss(unchanged_leaves+removed+\
                                       [new_l1]+[new_l2], theta)
            a = ld-ld_new
        return a
    
    def leaf_support_bound(self, leaf, w=None, theta=None, 
                           FP=None, FN=None, ld=None, ordered_leaves=None):
        if self.name == 'acc':
            tau = self.acc_loss(leaf.fp, leaf.fn)
        elif self.name == 'bacc':
            tau = self.bacc_loss(leaf.fp, leaf.fn)
        elif self.name == 'wacc':
            tau = self.wacc_loss(leaf.fp, leaf.fn, w)
        elif self.name == 'f1':
            tau = ld - self.f1_loss(FP-leaf.fp, FN-leaf.fn)
        elif self.name == 'auc':
            if leaf.p == 0 or leaf.n == 0:
                tau = -1
            else:
                leaves = ordered_leaves.copy()
                leaves.remove(leaf)
                ld_new = self.ach_lb(leaves, leaf.p, leaf.n)
                tau = ld-ld_new
        elif self.name == 'pauc':
            if leaf.p == 0 or leaf.n == 0:
                tau = -1
                ld_new = ld
            else:
                leaves = ordered_leaves.copy()
                leaves.remove(leaf)
                ld_new = self.pach_lb(leaves, theta, leaf.p, leaf.n)
                tau = ld-ld_new
        #print('leaf split feature', leaf.rules)
        #print('tau:',round(tau,4), 'ld:', round(ld,4), 'ld_new:', round(ld_new, 4))
        return tau
    
    def acc_loss(self, FP, FN):
        return (FP+FN)/(self.P+self.N)
    
    def bacc_loss(self, FP, FN):
        return 0.5*(FN/self.P + FP/self.N)
    
    def wacc_loss(self, FP, FN, w):
        return (FP+w*FN)/(w*self.P+self.N)
    
    def f1_loss(self, FP, FN):
        return (FP+FN)/(2*self.P+FP-FN)
    
    def ach_loss(self, leaves):
        ordered_leaves = order_leaves(leaves)
        tp = fp = np.array([0])
        if len(leaves) > 1:
            for i in range(0, len(leaves)):
                tp = np.append(tp, tp[i]+ordered_leaves[i].p)
                fp = np.append(fp, fp[i]+ordered_leaves[i].n)
        else:
            tp = np.append(tp, self.P)
            fp = np.append(fp, self.N)
    
        loss = 1-0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) \
                          for i in range(1,len(tp))])
        return ordered_leaves, loss
    
    def pach_loss(self, leaves, theta):
        ordered_leaves = order_leaves(leaves)
        tp = fp = np.array([0], dtype=float)
        if len(leaves) > 1:
            i = 0
            while fp[i] < self.N*theta and i < len(leaves):
                tp = np.append(tp, tp[i]+ordered_leaves[i].p)
                fp = np.append(fp, fp[i]+ordered_leaves[i].n)
                i += 1
            tp[i] = ((tp[i]-tp[i-1])/(fp[i]-fp[i-1]))*(self.N*theta-fp[i])+tp[i]
            fp[i] = self.N*theta
        else:
            tp = np.append(tp, self.P*theta)
            fp = np.append(fp, self.N*theta)
        loss = 1-0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) \
                          for i in range(1,len(tp))])
        return ordered_leaves, loss
    
    def ach_lb(self, leaves, Ps, Ns):
        tp = np.array([0, Ps], dtype=float)
        fp = np.array([0,0], dtype=float)
        for i in range(len(leaves)):
            tp = np.append(tp, tp[i+1]+leaves[i].p)
            fp = np.append(fp, fp[i+1]+leaves[i].n)
        tp = np.append(tp, tp[len(leaves)+1]+0)
        fp = np.append(fp, fp[len(leaves)+1]+Ns)
        loss = 1- 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) for i in range(1,len(tp))])
        
        return loss
    
    def pach_lb(self, leaves, theta, Ps, Ns):
        tp = np.array([0, Ps], dtype=float)
        fp = np.array([0, 0], dtype=float)
        i = 0
        while fp[i+1] < self.N*theta:
            if i < len(leaves):
                tp = np.append(tp, tp[i+1]+leaves[i].p)
                fp = np.append(fp, fp[i+1]+leaves[i].n)
                i += 1
            else:
                tp = np.append(tp, tp[i+1]+0)
                fp = np.append(fp, fp[i+1]+Ns)
                break                         
        tp[len(tp)-1] = ((tp[len(tp)-1]-tp[len(tp)-2])/(fp[len(fp)-1]-fp[len(fp)-2]))*\
                        (self.N*theta-fp[len(fp)-1])+tp[len(tp)-1]
        fp[len(fp)-1] = self.N*theta
        loss = 1- 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(self.P*self.N) \
               for i in range(1,len(tp))])
        return loss
      

class CacheTree:
    def __init__(self, name, P, N, lamb, leaves, w=None, theta=None):
        self.name = name
        self.P = P
        self.N = N
        self.leaves = leaves
        self.w = w
        self.theta = theta
            
        bound = Objective(name, P, N, lamb)
        self.risk = bound.risk(self.leaves, self.w, self.theta)
    
    def sorted_leaves(self):
        return tuple(sorted(leaf.rules for leaf in self.leaves))
        

class Tree:
    def __init__(self, cache_tree, n, lamb, splitleaf=None, prior_metric=None):
        
        self.cache_tree = cache_tree
        self.splitleaf = splitleaf
        leaves = cache_tree.leaves
        self.H = len(leaves)
        self.risk = cache_tree.risk
        
        bound = Objective(cache_tree.name, cache_tree.P, cache_tree.N, lamb)
        self.lb = bound.lower_bound(leaves, splitleaf, cache_tree.w, cache_tree.theta)
            
        if leaves[0].num_captured == n:
            self.metric = 0                #null tree
        elif prior_metric == "objective":
            self.metric = self.risk
        elif prior_metric == "bound":
            self.metric = self.lb
        elif prior_metric == "curiosity":
            removed_leaves = list(compress(leaves, splitleaf)) #dsplit
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves) # num captured by dsplit
            if num_cap_rm < n:
                self.metric = self.lb / ((n - num_cap_rm) / n) # supp(dun, xn)
            else:
                self.metric = self.lb / (0.01 / n) # null tree
        elif prior_metric == "entropy":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # entropy weighted by number of points captured
            self.entropy = [
                (-leaves[i].p * math.log2(leaves[i].p) - (1 - leaves[i].p) * math.log2(1 - leaves[i].p)) * leaves[
                    i].num_captured if leaves[i].p != 0 and leaves[i].p != 1 else 0 for i in range(self.H)]
            if num_cap_rm < n:
                self.metric = sum(self.entropy[i] for i in range(self.H) if splitleaf[i] == 0) / (
                        n - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.entropy[i] for i in range(self.H) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "gini":
            removed_leaves = list(compress(leaves, splitleaf))
            num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
            # gini index weighted by number of points captured
            self.giniindex = [(2 * leaves[i].p * (1 - leaves[i].p))
                              * leaves[i].num_captured for i in range(self.H)]
            if num_cap_rm < n:
                self.metric = sum(self.giniindex[i] for i in range(self.H) if splitleaf[i] == 0) / (
                        n - sum(leaf.num_captured for leaf in removed_leaves))
            else:
                self.metric = sum(self.giniindex[i] for i in range(self.H) if splitleaf[i] == 0) / 0.01
        elif prior_metric == "FIFO":
            self.metric = 0
        elif prior_metric == "random":
            self.metric = np.random.uniform(0.0,1.0)
        
    def __lt__(self, other):
        # define <, which will be used in the priority queue
        return self.metric < other.metric
    

class CacheLeaf:
    def __init__(self, name, n, P, N, rules, x, y, y_mpz, z_mpz, points_cap, 
                 num_captured, lamb, support, is_feature_dead, w=None):
        self.rules = rules
        self.points_cap = points_cap
        self.num_captured = num_captured
        self.is_feature_dead = is_feature_dead
        
        _, num_ones = rule_vand(points_cap, y_mpz) #return vand and cnt
        _, num_errors = rule_vand(points_cap, z_mpz)
        '''
        print('rules:', rules)
        print("points_cap:", points_cap, "vec:", rule_mpztovec(points_cap))
        print('_:', _, "vec:", rule_mpztovec(_))
        print('num_errors',num_errors)
        '''
        self.delta = num_errors
        self.p = num_ones
        self.n = self.num_captured - num_ones
        if self.num_captured > 0 :
            self.r = num_ones/self.num_captured
        else:
            self.r = 0
        bound = Objective(name, P, N, lamb)

        if name != 'pauc':
            if num_errors > 0:
                cap = np.array(rule_mpztovec(points_cap))
                cap_i = np.where(cap == 1)[0]
                x_cap = x[cap_i]
                y_cap = y[cap_i]
                
                v = rule_mpztovec(_)
                equiv_i = np.where(np.array(v) == 1)[0]
                idx = [i for i,c in enumerate(cap_i) if c in equiv_i]
                idx = np.array(idx)
                
                unique_rows, counts = np.unique(x_cap[idx,], axis=0, return_counts=True)
                nrow = unique_rows.shape[0]
                self.equiv = np.zeros((3, nrow+2))
    
                for i in range(nrow):
                    comp = np.all(np.equal(x_cap, unique_rows[i,]), axis=1)
                    eu = np.sum(comp)
                    j = np.where(comp==True)
                    n_neg = np.sum(y_cap[j]==0)
                    n_pos = eu-n_neg
                    self.equiv[0,i] = n_pos/eu    #r = n_pos/eu
                    self.equiv[1,i] = n_pos
                    self.equiv[2,i] = n_neg
            
                self.equiv[0, nrow] = 1
                #y_i = np.where(np.array(v)==0)[0]
                #equiv_not_i = [i for i,c in enumerate(cap_i) if c not in equiv_i]
                self.equiv[1, nrow] = sum(y_cap==1) - sum(self.equiv[1,i] for i in range(nrow))
                self.equiv[2, nrow+1] = sum(y_cap==0) - sum(self.equiv[2,i] for i in range(nrow))
            else:
                self.equiv = np.zeros((3, 2))
                self.equiv[0,0] = 1
                self.equiv[1,0] = self.p
                self.equiv[2,1] = self.n
    
        if self.num_captured:
            self.pred = bound.leaf_predict(self.p, self.n, w)
            if self.pred == 0:
                self.fp = 0
                self.fn = self.p
            else:
                self.fp = self.n
                self.fn = 0
        else:
            self.pred = 0
            self.fp = 0
            self.fn = self.p
                
class EquivPoints:
    def __init__(self, r, p, n):
        self.r = r
        self.p = p
        self.n = n


def get_false(leaves):
    FP = sum([l.fp for l in leaves])
    FN = sum([l.fn for l in leaves])
    return FP, FN
    
def get_fixed_false(leaves, splitleaf):
    FPu = sum([leaves[i].fp for i in range(len(leaves)) if splitleaf[i]==0])
    FNu = sum([leaves[i].fn for i in range(len(leaves)) if splitleaf[i]==0])
    return FPu, FNu
    
def get_split(leaves, splitleaf):
    Ps = sum([leaves[i].p for i in range(len(leaves)) if splitleaf[i]==1])
    Ns = sum([leaves[i].n for i in range(len(leaves)) if splitleaf[i]==1])
    return Ps, Ns
                
def order_leaves(leaves):
    return sorted([l for l in leaves], key=lambda x:x.r, reverse=True)

def order_fixed_leaves(leaves, splitleaf):
    leaf_fixed = [not split for split in splitleaf]
    fixed_set = list(compress(leaves, leaf_fixed))
    ordered_fixed = order_leaves(fixed_set)
    return ordered_fixed

def equiv_lb(name, leaves, splitleaf, P, N, lamb, w):
    split_set = list(compress(leaves, splitleaf))
    #leaf_fixed = [not split for split in splitleaf]
    
    leaf_equiv_fp = 0
    leaf_equiv_fn = 0
    
    bound = Objective(name, P, N, lamb)
    
    for i in split_set:
        #print(rule_mpztovec(i.points_cap))
        equiv = i.equiv # equiv is the array row=3 and col=equivset+2
        if equiv.shape[1]-2 > 0:
            for j in range(equiv.shape[1]-2):
                pred = bound.leaf_predict(equiv[1,j], equiv[2,j], w)
                if pred == 0:
                    leaf_equiv_fp += 0
                    leaf_equiv_fn += equiv[1,j]
                else:
                    leaf_equiv_fp += equiv[2,j]
                    leaf_equiv_fn += 0
        #print('delta_fp:', leaf_equiv_fp, 'delta_fn:', leaf_equiv_fn)
        
    return leaf_equiv_fp, leaf_equiv_fn

    
def ach_equiv_lb(leaves, splitleaf, P, N, lamb):
    split_set = list(compress(leaves, splitleaf))
    leaf_fixed = [not split for split in splitleaf]
    fixed_set = list(compress(leaves, leaf_fixed))
    Ps1 = 0
    Ns1 = 0
    for i in split_set:
        equiv = i.equiv # equiv is the array row=3 and col=equivset+2
        if equiv.shape[1]-2 > 0:
            for j in range(equiv.shape[1]-2):
                leaf_equiv = EquivPoints(equiv[0,j], equiv[1,j], equiv[2,j])
                fixed_set.append(leaf_equiv)
        Ps1 += equiv[1, equiv.shape[1]-2]
        Ns1 += equiv[2, equiv.shape[1]-1]
    ordered_leaves = order_leaves(fixed_set)
    
    tp = np.array([0, Ps1], dtype=float)
    fp = np.array([0,0], dtype=float)
    for i in range(len(ordered_leaves)):
        tp = np.append(tp, tp[i+1]+ordered_leaves[i].p)
        fp = np.append(fp, fp[i+1]+ordered_leaves[i].n)
    tp = np.append(tp, tp[len(ordered_leaves)+1]+0)
    fp = np.append(fp, fp[len(ordered_leaves)+1]+Ns1)
    
    loss = 1- 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(P*N) for i in range(1,len(tp))])
    return loss + lamb*len(leaves)



def generate_new_splitleaf(name, P, N, unchanged_leaves, removed_leaves, new_leaves, lamb,
                           incre_support, w, theta):
    
    n_removed_leaves = len(removed_leaves)  #dsplit
    n_unchanged_leaves = len(unchanged_leaves) #dun
    n_new_leaves = len(new_leaves)
    n_new_tree_leaves = n_unchanged_leaves + n_new_leaves #H'
    splitleaf1 = [0] * n_unchanged_leaves + [1] * n_new_leaves
    
    bound = Objective(name, P, N, lamb)
    FPu = None
    FNu = None
    ld = None
    if name == 'f1':
        FPu, FNu = get_false(unchanged_leaves)
    if name == 'auc':
        _, ld = bound.ach_loss(unchanged_leaves+removed_leaves)
    if name == 'pauc':
        _, ld = bound.pach_loss(unchanged_leaves+removed_leaves, theta)
    
    sl = []
    for i in range(n_removed_leaves):
        splitleaf = [0]*n_new_tree_leaves
        removed_l = removed_leaves[i]
        new_l1 = new_leaves[2*i]
        new_l2 = new_leaves[2*i+1]
        a = bound.incre_accu_bound(removed_l, new_l1, new_l2, unchanged_leaves, 
                         removed_leaves, w, FPu, FNu, theta, ld)
        if not incre_support:
            a = float('Inf')
        if a <= lamb:
            splitleaf[n_unchanged_leaves + 2*i] = 1
            splitleaf[n_unchanged_leaves + 2*i + 1] = 1
            sl.append(splitleaf)
        else:
            sl.append(splitleaf1)
    return sl

def get_cannot_split(name, P, N, lamb, m, new_tree_leaves, MAXDEPTH, w, theta):
    bound = Objective(name, P, N, lamb)
    FP = None
    FN = None
    ld = None
    ordered_leaves = None
    if name == 'f1':
        FP, FN = get_false(new_tree_leaves)
        ld = bound.f1_loss(FP, FN)
    if name == 'auc':
        ordered_leaves, ld = bound.ach_loss(new_tree_leaves)
    if name == 'pauc':
        ordered_leaves, ld = bound.pach_loss(new_tree_leaves, theta)
        
    cannot_split = [len(l.rules) >= MAXDEPTH or bound.leaf_support_bound(l, w, theta, \
                    FP, FN, ld, ordered_leaves) < lamb or\
                    all([l.is_feature_dead[r-1] for r in range(1, m+1) if r not in \
                    map(abs, l.rules)]) for l in new_tree_leaves]
    return cannot_split



def bbound(x, y, name, lamb, prior_metric=None, w=None, theta=None, MAXDEPTH=float('Inf'), 
           MAX_NLEAVES=float('Inf'), niter=float('Inf'), logon=False,
           support=True, incre_support=True, accu_support=True, equiv_points=True,
           lookahead=True, lenbound=True, R_c0 = 1, timelimit=float('Inf'), init_cart = True,
           saveTree = False, readTree = False):

    x0 = copy.deepcopy(x)
    y0 = copy.deepcopy(y)

    tic = time.time()

    m = x.shape[1] # number of features
    n = len(y)
    P = np.count_nonzero(y)
    N = n-P

    x_mpz = [rule_vectompz(x[:, i]) for i in range(m)]
    y_mpz = rule_vectompz(y)

    # order the columns by descending gini reduction
    idx, dic = gini_reduction(x_mpz, y_mpz, n, range(m))
    #idx, dic = get_variable_importance(x, y)
    
    x = x[:, idx]
    x_mpz = [x_mpz[i] for i in idx]
    
    z_mpz = get_z(x,y,n,m)


    lines = []  # a list for log
    leaf_cache = {}  # cache leaves
    tree_cache = {}  # cache trees

    # initialize the queue to include just empty root
    queue = []
    root_leaf = CacheLeaf(name, n, P, N, (), x, y, y_mpz, z_mpz, make_all_ones(n + 1), 
                          n, lamb, support, [0] * m, w)
    d_c = CacheTree(name, P, N, lamb=lamb, leaves=[root_leaf], w=w, theta=theta)
    R_c = d_c.risk
    tree0 = Tree(cache_tree=d_c, n=n, lamb=lamb,splitleaf=[1], prior_metric=prior_metric)
    heapq.heappush(queue, (tree0.metric, tree0))
    
    best_is_cart = False  # a flag for whether or not the best is the initial CART
    if init_cart: 
        clf, nleaves_CART, trainout_CART, R_c, d_c, C_c = cart(x0, y0, name, n, P, N, lamb, w, theta, MAXDEPTH)
        time_c = time.time() - tic
        best_is_cart = True
        print('risk of cart:', R_c)
    else:
        C_c=0
        clf=None
        time_c = time.time()
        
    if readTree:
        with open('tree.pkl', 'rb') as f:
            d_c = pickle.load(f)
        R_c = d_c.risk

        with open('leaf_cache.pkl', 'rb') as f:
            leaf_cache = pickle.load(f)

        sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in d_c.leaves))
        tree_cache[sorted_new_tree_rules] = True

        tree_p = Tree(cache_tree=d_c, n=n, lamb=lamb, 
                      splitleaf=[1]*len(d_c.leaves), prior_metric=prior_metric)

        heapq.heappush(queue, (tree_p.metric, tree_p))
        '''
        print("PICKEL>>>>>>>>>>>>>", [leaf.rules for leaf in d_c.leaves])
        print('R_c:', R_c)
        print('lower_bound:', tree_p.lb)
        print('lookahead:',tree_p.lb+lamb*sum(tree_p.splitleaf))
        '''
        #print("leaf_cache:", leaf_cache)

        C_c = 0
        time_c = time.time() - tic
        
    if R_c0 < R_c:
        R_c = R_c0

    
    leaf_cache[()] = root_leaf

    COUNT = 0  # count the total number of trees in the queue
    COUNT_POP = 0 # number of tree poped from queue (# of tree checked)
    COUNT_UNIQLEAVES = 0
    COUNT_LEAFLOOKUPS = 0
    
    if logon:
        header = ['time', '#pop', '#push', 'queue_size', 'metric', 'R_c',
                  'the_old_tree', 'the_old_tree_splitleaf', 'the_old_tree_objective', 'the_old_tree_lbound',
                  'the_new_tree', 'the_new_tree_splitleaf',
                  'the_new_tree_objective', 'the_new_tree_lbound', 'the_new_tree_length', 'the_new_tree_depth', 'queue']

        fname = "_".join([name, str(m), str(n), prior_metric,
                          str(lamb), str(MAXDEPTH), str(init_cart), ".txt"])
        with open(fname, 'w') as f:
            f.write('%s\n' % ";".join(header))
    
    bound = Objective(name, P, N, lamb)
    
    #len_queue=[]
    #time_queue=[]
    #count_tree = []
    #time_realize_best_tree=[time_c]
    #R_best_tree=[R_c]
    #best_tree = [d_c]

    while queue and COUNT < niter and time.time() - tic < timelimit:
        '''
        print(len(queue))
        for metric, t in queue:
            print(metric, [l.rules for l in t.cache_tree.leaves], t.splitleaf)
        '''
        metric, tree = heapq.heappop(queue)
        

        COUNT_POP = COUNT_POP + 1
        #count_tree.append(COUNT_POP)
        
        leaves = tree.cache_tree.leaves
        leaf_split = tree.splitleaf       
        removed_leaves = list(compress(leaves, leaf_split))
        old_tree_length = len(leaf_split)
        new_tree_length = len(leaf_split) + sum(leaf_split)
        
        # prefix-specific upper bound on number of leaves
        if lenbound and new_tree_length >= min(old_tree_length + math.floor((R_c - tree.lb) / lamb),
                                               2**m):
            continue

        n_removed_leaves = sum(leaf_split)
        n_unchanged_leaves = old_tree_length - n_removed_leaves
        
        #print("num in queue:", len(queue))
        #print(time.time()-tic)
        #len_queue.append(len(queue))
        #time_queue.append(time.time()-tic)
        
        
        '''equivalent points bound + lookahead bound'''        
        lambbb = lamb if lookahead else 0
        
        if (name != 'auc') and (name != 'pauc'):
            FPu, FNu = get_fixed_false(leaves, leaf_split)
            if equiv_points:
                delta_fp, delta_fn = equiv_lb(name, leaves, leaf_split, P, N, lamb, w) 
                #print('delta_fp:', delta_fp)
                #print('delta_fn:', delta_fn)
            else:
                delta_fp=0
                delta_fn=0
            
            if (bound.loss(FPu+delta_fp, FNu+delta_fn, w)+ (old_tree_length+n_removed_leaves) * lambbb >= R_c):
                continue
  
        if (name == 'auc'): 
            if (ach_equiv_lb(leaves, leaf_split, P, N, lamb) + n_removed_leaves*lambbb >= R_c):
                continue

        
        if (name == 'pauc') and (tree.lb + n_removed_leaves * lambbb>= R_c):
            continue

        leaf_no_split = [not split for split in leaf_split]
        unchanged_leaves = list(compress(leaves, leaf_no_split))

        # Generate all assignments of rules to the leaves that are due to be split
        rules_for_leaf = [set(range(1, m + 1)) - set(map(abs, l.rules)) -
                          set([i+1 for i in range(m) if l.is_feature_dead[i] == 1]) for l in removed_leaves]

        for leaf_rules in product(*rules_for_leaf):

            if time.time() - tic >= timelimit:
                break

            new_leaves = []
            flag_increm = False  # a flag for jump out of the loops (incremental support bound)
            for rule, removed_leaf in zip(leaf_rules, removed_leaves):

                rule_index = rule - 1
                tag = removed_leaf.points_cap  # points captured by the leaf's parent leaf

                for new_rule in (-rule, rule):
                    new_rule_label = int(new_rule > 0)
                    new_rules = tuple(
                        sorted(removed_leaf.rules + (new_rule,)))
                    if new_rules not in leaf_cache:

                        COUNT_UNIQLEAVES = COUNT_UNIQLEAVES+1

                        tag_rule = x_mpz[rule_index] if new_rule_label == 1 else ~(x_mpz[rule_index]) | mpz(pow(2, n))
                        new_points_cap, new_num_captured = rule_vand(tag, tag_rule)

                        #parent_is_feature_dead =
                        new_leaf = CacheLeaf(name, n, P, N, new_rules, x, y, y_mpz, z_mpz, new_points_cap, new_num_captured,
                                             lamb, support, removed_leaf.is_feature_dead.copy(), w)
                        leaf_cache[new_rules] = new_leaf
                        new_leaves.append(new_leaf)
                    else:

                        COUNT_LEAFLOOKUPS = COUNT_LEAFLOOKUPS+1

                        new_leaf = leaf_cache[new_rules]
                        new_leaves.append(new_leaf)

                    '''
                    # Lower bound on classification accuracy
                    # if (new_leaf.num_captured) / n <= lamb:
                    # accu_support == theorem 9 in OSDT, check if feature dead, not derived yet
                    
                    if accu_support == True and (new_leaf.num_captured - new_leaf.num_captured_incorrect) / n <= lamb:

                        removed_leaf.is_feature_dead[rule_index] = 1

                        flag_increm = True
                        break
                    '''    

                if flag_increm:
                    break

            if flag_increm:
                continue

            new_tree_leaves = unchanged_leaves + new_leaves

            sorted_new_tree_rules = tuple(sorted(leaf.rules for leaf in new_tree_leaves))

            if sorted_new_tree_rules in tree_cache:
                continue
            else:
                tree_cache[sorted_new_tree_rules] = True

            child = CacheTree(name, P, N, lamb, new_tree_leaves, w=w, theta=theta)
            
            #print([l.rules for l in child.leaves])

            R = child.risk
            
            #print("R:", R, "R_c:", R_c)
            #time_realize_best_tree.append(time.time()-tic)
            #R_best_tree.append(R)
            
            
            if R < R_c:
                d_c = child
                #best_tree.append([leaf.rules for leaf in d_c.leaves])
                #R_best_tree.append(R)
                #time_realize_best_tree.append(time.time()-tic)
                R_c = R
                C_c = COUNT + 1
                time_c = time.time() - tic
                

                best_is_cart = False

            # generate the new splitleaf for the new tree
            sl = generate_new_splitleaf(name, P, N, unchanged_leaves, removed_leaves, new_leaves,
                                        lamb, incre_support, w, theta) # a_j

            cannot_split = get_cannot_split(name, P, N, lamb, m, new_tree_leaves, 
                                            MAXDEPTH, w, theta)
                

            # For each copy, we don't split leaves which are not split in its parent tree.
            # In this way, we can avoid duplications.
            can_split_leaf = [(0,)] * n_unchanged_leaves + \
                             [(0,) if cannot_split[i]
                              else (0, 1) for i in range(n_unchanged_leaves, new_tree_length)]
            # Discard the first element of leaf_splits, since we must split at least one leaf
            new_leaf_splits0 = np.array(list(product(*can_split_leaf))[1:])#sorted(product(*can_split_leaf))[1:]
            len_sl = len(sl)
            if len_sl == 1:
                # Filter out those which split at least one leaf in dp (d0)
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if np.dot(ls, sl[0]) > 0]
            else:
                # Filter out those which split at least one leaf in dp and split at least one leaf in d0
                new_leaf_splits = [ls for ls in new_leaf_splits0
                                   if all([np.dot(ls, sl[i]) > 0 for i in range(len_sl)])]

            for new_leaf_split in new_leaf_splits:
                # construct the new tree
                tree_new = Tree(cache_tree=child, n=n, lamb=lamb,
                                splitleaf=new_leaf_split, prior_metric=prior_metric)
                '''
                print('tree_lb:', round(tree_new.lb, 4), 
                      'tree_risk:', round(tree.cache_tree.risk, 4))
                '''
                #print('tree_rules_x8:', [l.rules for l in tree.cache_tree.leaves])
                
                
                # MAX Number of leaves
                if len(new_leaf_split)+sum(new_leaf_split) > MAX_NLEAVES:
                    continue

                COUNT = COUNT + 1
                #print([l.rules for l in tree_new.cache_tree.leaves], tree_new.splitleaf)
                '''
                if (COUNT <= 22):
                    print([l.rules for l in tree_new.cache_tree.leaves], 
                          tree_new.splitleaf, round(tree_new.lb, 4), 
                          round(tree_new.cache_tree.risk,4), round(tree_new.metric, 4), 
                          round(metric,4), [l.rules for l in tree.cache_tree.leaves])
                
                if (COUNT ==22)|(COUNT == 21)|(COUNT==20):
                    for metric, t in queue:
                        print(metric, [l.rules for l in t.cache_tree.leaves], t.splitleaf)
                   
                if COUNT == 22:
                    print('123455667677')
                    return
                '''
                # heapq.heappush(queue, (2*tree_new.metric - R_c, tree_new))
                heapq.heappush(queue, (tree_new.metric, tree_new))
                
                if logon:
                    log(tic, lines, COUNT_POP, COUNT, queue, metric, R_c, tree, tree_new, sorted_new_tree_rules, fname)
                
                if COUNT % 1000000 == 0:
                    print("COUNT:", COUNT)
        #print('COUNT:', COUNT)

    totaltime = time.time() - tic

    if not best_is_cart:

        accu = 1-(R_c-lamb*len(d_c.leaves))

        leaves_c = [leaf.rules for leaf in d_c.leaves]
        pred_c = [leaf.pred for leaf in d_c.leaves]

        num_captured = [leaf.num_captured for leaf in d_c.leaves]

        #num_captured_incorrect = [leaf.num_captured_incorrect for leaf in d_c.leaves]

        nleaves = len(leaves_c)
    else:
        accu = trainout_CART
        leaves_c = 'NA'
        pred_c = 'NA'
        get_code(d_c, ['x'+str(i) for i in range(1, m+1)], [0, 1])
        num_captured = 'NA'
        #num_captured_incorrect = 'NA'
        nleaves = nleaves_CART
        
    if saveTree:
        with open('tree.pkl', 'wb') as f:
            pickle.dump(d_c, f)
        with open('leaf_cache.pkl', 'wb') as f:
            pickle.dump(leaf_cache, f)
        

    '''
    print(">>> log:", logon)
    print(">>> support bound:", support)
    print(">>> accu_support:", accu_support)
    print(">>> accurate support bound:", incre_support)
    print(">>> equiv points bound:", equiv_points)
    print(">>> lookahead bound:", lookahead)
    print("prior_metric=", prior_metric)
    '''
    print("loss function:", name)
    print("lambda: ", lamb)
    print("COUNT_UNIQLEAVES:", COUNT_UNIQLEAVES)
    print("COUNT_LEAFLOOKUPS:", COUNT_LEAFLOOKUPS)
    print("total time: ", totaltime)   
    print("leaves: ", leaves_c)
    print("num_captured: ", num_captured)
    print("prediction: ", pred_c)
    print("Objective: ", R_c)
    print(name, ": ", accu)
    print("COUNT of the best tree: ", C_c)
    print("time when the best tree is achieved: ", time_c)
    print("TOTAL COUNT: ", COUNT)

    return leaves_c, pred_c, dic, nleaves, m, n, totaltime, time_c, R_c, COUNT, C_c, \
            accu, best_is_cart, clf#, len_queue, time_queue, \
            #time_realize_best_tree, R_best_tree, count_tree, best_tree


def predict(name, leaves_c, prediction_c, nleaves, dic, x, y, best_is_cart, clf, w=None, theta=None, logon=False):
    """
    :param leaves_c:
    :param dic:
    :return:
    """
    P = np.count_nonzero(y)
    N = len(y) - P
    if best_is_cart:
        yhat = clf.predict(x)
        
        n_fp = sum((yhat == 1) & (yhat != y))
        n_fn = sum((yhat == 0) & (yhat != y))
        n_tp = sum((yhat == 1) & (yhat == y))
        
        if name == 'acc':
            out = sklearn.metrics.accuracy_score(y, yhat)
        elif name == "bacc":
            out = sklearn.metrics.balanced_accuracy_score(y, yhat)
        elif name == 'wacc':
            #print(n_fp,n_fn,P,N,w)
            out = 1-(n_fp + w*n_fn)/(w*P+N)
        elif name == 'f1':
            out = sklearn.metrics.f1_score(y, yhat)
        elif name == 'auc':
            nodes = clf.apply(x)
            metric = np.empty([3, int(nleaves)], dtype=float)
            for i, num in enumerate(set(nodes)):
                idx = np.where(nodes == num)
                metric[0,i] = sum(y[idx]==1)
                metric[1,i] = sum(y[idx]==0)
                metric[2,i] = sum(y[idx]==1)/len(y[idx])
            metric = metric[:,np.argsort(metric[2,])]
            metric = np.flip(metric, axis=1)
            metric = np.cumsum(metric,axis=1)
            init = np.array([[0], [0], [0]])
            metric = np.append(init, metric, axis=1)
            tp = metric[0, :]
            fp = metric[1, :]
            out = 0.5*sum([(metric[0,i]/P+metric[0,i-1]/P)*\
                                 (metric[1,i]/N-metric[1,i-1]/N) for i in \
                                 range(1,metric.shape[1])])
        elif name == 'pauc':
            nodes = clf.apply(x)
            metric = np.zeros([3, int(nleaves)], dtype=float)
            for i, num in enumerate(set(nodes)):
                idx = np.where(nodes == num)
                metric[0,i] = sum(y[idx]==1)
                metric[1,i] = sum(y[idx]==0)
                metric[2,i] = sum(y[idx]==1)/len(y[idx])
                #print(metric[0,i], metric[1,i], metric[2,i])
            metric = metric[:,np.argsort(metric[2,])]
            metric = np.flip(metric, axis=1)
            metric = np.cumsum(metric,axis=1)
            init = np.array([[0], [0], [0]])
            metric = np.append(init, metric, axis=1)
            '''
            tp = [0]
            fp = [0]
            i = 1
            while fp[i-1] < N*theta and i <= int(nleaves)+1:
                tp.append(metric[0,i])
                fp.append(metric[1,i])
                i += 1
            
            tp[i-1] = ((tp[i-1]-tp[i-2])/(fp[i-1]-fp[i-2]))*(N*theta-fp[i-1])+tp[i-1]
            fp[i-1] = N*theta
            out = 0.5*sum([(tp[i]+tp[i-1])*(fp[i]-fp[i-1])/(P*N) for i in range(1,len(tp))])   
            '''
            tp = metric[0, :]
            fp = metric[1, :]
            out = 0.5*sum([(metric[0,i]/P+metric[0,i-1]/P)*\
                           (metric[1,i]/N-metric[1,i-1]/N) for i in \
                           range(1,metric.shape[1])])
        
        print("Best is cart! Testing", name, round(out,4))
        print("P=", P, "N=", N, "FP=", n_fp, "FN=", n_fn)
        if (name=='auc') or (name == 'pauc'):
            print('tp=', tp)
            print('fp=',fp)
            plt.plot([f/N for f in fp], [f/P for f in tp], 'go-', linewidth=2)
        print(">>>>>>>>>>>>>>>>>>>>>>>")

        return yhat, out

    n = x.shape[0]

    caps = []

    for leaf in leaves_c:
        cap = np.array([1] * n)
        for feature in leaf:
            idx = dic[abs(feature)]
            feature_label = int(feature > 0)
            cap = (x[:, idx] == feature_label) * cap
        caps.append(cap)

    yhat = np.array([1] * n)
    cap_pos = []
    cap_neg = []

    for j in range(len(caps)):
        idx_cap = [i for i in range(n) if caps[j][i] == 1]
        yhat[idx_cap] = prediction_c[j]
        cap_pos.append(sum(y[idx_cap]))
        cap_neg.append(len(idx_cap)-sum(y[idx_cap]))

    n_fp = sum((yhat == 1) & (yhat != y))
    n_fn = sum((yhat == 0) & (yhat != y))
    #n_tp = sum((yhat == 1) & (yhat == y))
    
    
    if name == 'acc':
        out = sklearn.metrics.accuracy_score(y, yhat)
    elif name == "bacc":
        out = sklearn.metrics.balanced_accuracy_score(y, yhat)
    elif name == 'wacc':
        out = 1-(n_fp + w*n_fn)/(w*P+N)
    elif name == 'f1':
        out = sklearn.metrics.f1_score(y, yhat)
    elif name == 'auc':
        '''
        n_p = []
        n_n = []
        for i in range(len(caps)):
            idx = np.where(caps[i]==1)
            n_p.append(sum(y[idx]==1))
            n_n.append(sum(y[idx]==0))
        '''    
        metric = np.empty([3, len(caps)], dtype=float)
        for i in range(len(caps)):
            idx = np.where(caps[i]==1)
            metric[0,i] = sum(y[idx]==1)
            metric[1,i] = sum(y[idx]==0)
            metric[2,i] = sum(y[idx]==1)/len(y[idx])
        
        metric = metric[:,np.argsort(metric[2,])]
        metric = np.flip(metric, axis=1)
        metric = np.cumsum(metric,axis=1)
        init = np.array([[0], [0], [0]])
        metric = np.append(init, metric, axis=1)  
        tp = metric[0, :]
        fp = metric[1, :]
        out = 0.5*sum([(metric[0,i]+metric[0,i-1])*(metric[1,i]-metric[1,i-1])/(P*N) for i in range(1,len(tp))])
    
        
    elif name == 'pauc':
        metric = np.empty([3, len(caps)], dtype=float)
        for i in range(len(caps)):
            idx = np.where(caps[i]==1)
            metric[0,i] = sum(y[idx]==1)
            metric[1,i] = sum(y[idx]==0)
            metric[2,i] = sum(y[idx]==1)/len(y[idx])
        
        metric = metric[:,np.argsort(metric[2,])]
        metric = np.flip(metric, axis=1)
        metric = np.cumsum(metric,axis=1)
        init = np.array([[0], [0], [0]])
        metric = np.append(init, metric, axis=1)
        '''
        tp = [0]
        fp = [0]
        i = 1
        while fp[i-1] < N*theta and i <= len(caps)+1:
            tp.append(metric[0,i])
            fp.append(metric[1,i])
            i += 1          
        tp[i-1] = ((tp[i-1]-tp[i-2])/(fp[i-1]-fp[i-2]))*(N*theta-fp[i-1])+tp[i-1]
        fp[i-1] = N*theta
        out = 0.5*sum([(tp[i]/P+tp[i-1]/P)*(fp[i]/N-fp[i-1]/N) for i in range(1,len(tp))])
        '''
        tp = metric[0, :]
        fp = metric[1, :]
        out = 0.5*sum([(metric[0,i]+metric[0,i-1])*(metric[1,i]-metric[1,i-1])/(P*N) for i in range(1,len(tp))])
    
    if logon==True:
        fname = '_'.join(['pred', name, str(P+N), '.txt'])    
        with open(fname, 'w') as f:
            f.write(';'.join(["P=", str(P), "N=", str(N), 
                              "FP=", str(n_fp), "FN=", str(n_fn), 
                    name, "=", str(round(out, 4))]))
    print('Leaf:', leaves_c)
    print('captured pos:', cap_pos)
    print('captured neg:', cap_neg)        
    print("Testing", name, round(out,4))
    #print("Testing", "auc:", round(out,4))
    if name != 'auc' and name != 'pauc':
        print("P=", P, "N=", N, "FP=", n_fp, "FN=", n_fn)
        plt.plot([0, n_fp/N, 1], [0, (P-n_fn)/P, 1], 'go-', linewidth=2)
    else:
        #print("P=", P, "N=", N, "l.p=", n_p, "l.n=", n_n)
        print('P=',P, "N=", N, 'tp=', tp, "fp=", fp)
        plt.plot([f/N for f in fp], [f/P for f in tp], 'go-', linewidth=2)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    return yhat, out


# class Objective:
#     def __init__(self, name, P, N, lamb):
#         self.name = name
#         self.P = P
#         self.N = N
#         self.lamb = lamb

#     def leaf_predict(self, p, n, w=None):
#         predict = 1
#         if self.name == "acc":
#             if p < n:
#                 predict = 0
#         elif self.name == "bacc":
#             if p / self.P <= n / self.N:
#                 predict = 0
#         elif self.name == "wacc":
#             if w * p <= n:
#                 predict = 0
#         elif self.name == "f1":
#             if w * p <= n:
#                 predict = 0
#         return predict

#     def loss(self, FP, FN, w=None, leaves=None, theta=None):
#         if self.name == "acc":
#             loss = self.acc_loss(FP, FN)
#         elif self.name == "bacc":
#             loss = self.bacc_loss(FP, FN)
#         elif self.name == "wacc":
#             loss = self.wacc_loss(FP, FN, w)
#         elif self.name == "f1":
#             loss = self.f1_loss(FP, FN)
#         elif self.name == "auc_convex":
#             _, loss = self.ach_loss(leaves)
#         elif self.name == "partial_auc":
#             _, loss = self.pach_loss(leaves, theta)
#         return loss

#     def risk(self, leaves, w=None, theta=None):
#         FP, FN = get_false(leaves)
#         risk = self.loss(FP, FN, w, leaves, theta) + self.lamb * len(leaves)
#         return risk

#     def lower_bound(self, leaves, splitleaf, w=None, theta=None):
#         FP, FN = get_fixed_false(leaves, splitleaf)
#         if self.name == "acc":
#             loss = self.acc_loss(FP, FN)
#         elif self.name == "bacc":
#             loss = self.bacc_loss(FP, FN)
#         elif self.name == "wacc":
#             loss = self.wacc_loss(FP, FN, w)
#         elif self.name == "f1":
#             loss = self.f1_loss(FP, FN)
#         elif self.name == "auc_convex":
#             ordered_fixed = order_fixed_leaves(leaves, splitleaf)
#             if len(ordered_fixed) == 0:
#                 loss = 0
#             else:
#                 Ps, Ns = get_split(leaves, splitleaf)
#                 loss = self.ach_lb(ordered_fixed, Ps, Ns)
#         elif self.name == "partial_auc":
#             ordered_fixed = order_fixed_leaves(leaves, splitleaf)
#             if len(ordered_fixed) == 0:
#                 loss = 1 - theta
#             else:
#                 Ps, Ns = get_split(leaves, splitleaf)
#                 loss = self.pach_lb(ordered_fixed, theta, Ps, Ns)
#         lb = loss + self.lamb * len(leaves)
#         return lb

#     def incre_accu_bound(
#         self,
#         removed_l,
#         new_l1,
#         new_l2,
#         unchanged_leaves=None,
#         removed_leaves=None,
#         w=None,
#         FPu=None,
#         FNu=None,
#         theta=None,
#         ld=None,
#     ):
#         if self.name == "acc":
#             a = self.acc_loss(
#                 removed_l.fp - new_l1.fp - new_l2.fp,
#                 removed_l.fn - new_l1.fn - new_l2.fn,
#             )
#         elif self.name == "bacc":
#             a = self.bacc_loss(
#                 removed_l.fp - new_l1.fp - new_l2.fp,
#                 removed_l.fn - new_l1.fn - new_l2.fn,
#             )
#         elif self.name == "wacc":
#             a = self.wacc_loss(
#                 removed_l.fp - new_l1.fp - new_l2.fp,
#                 removed_l.fn - new_l1.fn - new_l2.fn,
#                 w,
#             )
#         elif self.name == "f1":
#             a = self.f1_loss(FPu + removed_l.fp, FNu + removed_l.fn) - self.f1_loss(
#                 FPu + new_l1.fp + new_l2.fp, FNu + new_l1.fn + new_l2.fn
#             )
#         elif self.name == "auc_convex":
#             removed = removed_leaves.copy()
#             removed.remove(removed_l)
#             _, ld_new = self.ach_loss(unchanged_leaves + removed + [new_l1] + [new_l2])
#             a = ld - ld_new
#         elif self.name == "partial_auc":
#             removed = removed_leaves.copy()
#             removed.remove(removed_l)
#             _, ld_new = self.pach_loss(
#                 unchanged_leaves + removed + [new_l1] + [new_l2], theta
#             )
#             a = ld - ld_new
#         return a

#     def leaf_support_bound(
#         self, leaf, w=None, theta=None, FP=None, FN=None, ld=None, ordered_leaves=None
#     ):
#         if self.name == "acc":
#             tau = self.acc_loss(leaf.fp, leaf.fn)
#         elif self.name == "bacc":
#             tau = self.bacc_loss(leaf.fp, leaf.fn)
#         elif self.name == "wacc":
#             tau = self.wacc_loss(leaf.fp, leaf.fn, w)
#         elif self.name == "f1":
#             tau = ld - self.f1_loss(FP - leaf.fp, FN - leaf.fn)
#         elif self.name == "auc_convex":
#             if leaf.p == 0 or leaf.n == 0:
#                 tau = -1
#             else:
#                 leaves = ordered_leaves.copy()
#                 leaves.remove(leaf)
#                 ld_new = self.ach_lb(leaves, leaf.p, leaf.n)
#                 tau = ld - ld_new
#         elif self.name == "partial_auc":
#             if leaf.p == 0 or leaf.n == 0:
#                 tau = -1
#                 ld_new = ld
#             else:
#                 leaves = ordered_leaves.copy()
#                 leaves.remove(leaf)
#                 ld_new = self.pach_lb(leaves, theta, leaf.p, leaf.n)
#                 tau = ld - ld_new
#         # print('leaf split feature', leaf.rules)
#         # print('tau:',round(tau,4), 'ld:', round(ld,4), 'ld_new:', round(ld_new, 4))
#         return tau

#     def acc_loss(self, FP, FN):
#         return (FP + FN) / (self.P + self.N)

#     def bacc_loss(self, FP, FN):
#         return 0.5 * (FN / self.P + FP / self.N)

#     def wacc_loss(self, FP, FN, w):
#         return (FP + w * FN) / (w * self.P + self.N)

#     def f1_loss(self, FP, FN):
#         return (FP + FN) / (2 * self.P + FP - FN)

#     def ach_loss(self, leaves):
#         ordered_leaves = order_leaves(leaves)
#         tp = fp = np.array([0])
#         if len(leaves) > 1:
#             for i in range(0, len(leaves)):
#                 tp = np.append(tp, tp[i] + ordered_leaves[i].p)
#                 fp = np.append(fp, fp[i] + ordered_leaves[i].n)
#         else:
#             tp = np.append(tp, self.P)
#             fp = np.append(fp, self.N)

#         loss = 1 - 0.5 * sum(
#             [
#                 (tp[i] + tp[i - 1]) * (fp[i] - fp[i - 1]) / (self.P * self.N)
#                 for i in range(1, len(tp))
#             ]
#         )
#         return ordered_leaves, loss

#     def pach_loss(self, leaves, theta):
#         ordered_leaves = order_leaves(leaves)
#         tp = fp = np.array([0], dtype=float)
#         if len(leaves) > 1:
#             i = 0
#             while fp[i] < self.N * theta and i < len(leaves):
#                 tp = np.append(tp, tp[i] + ordered_leaves[i].p)
#                 fp = np.append(fp, fp[i] + ordered_leaves[i].n)
#                 i += 1
#             tp[i] = ((tp[i] - tp[i - 1]) / (fp[i] - fp[i - 1])) * (
#                 self.N * theta - fp[i]
#             ) + tp[i]
#             fp[i] = self.N * theta
#         else:
#             tp = np.append(tp, self.P * theta)
#             fp = np.append(fp, self.N * theta)
#         loss = 1 - 0.5 * sum(
#             [
#                 (tp[i] + tp[i - 1]) * (fp[i] - fp[i - 1]) / (self.P * self.N)
#                 for i in range(1, len(tp))
#             ]
#         )
#         return ordered_leaves, loss

#     def ach_lb(self, leaves, Ps, Ns):
#         tp = np.array([0, Ps], dtype=float)
#         fp = np.array([0, 0], dtype=float)
#         for i in range(len(leaves)):
#             tp = np.append(tp, tp[i + 1] + leaves[i].p)
#             fp = np.append(fp, fp[i + 1] + leaves[i].n)
#         tp = np.append(tp, tp[len(leaves) + 1] + 0)
#         fp = np.append(fp, fp[len(leaves) + 1] + Ns)
#         loss = 1 - 0.5 * sum(
#             [
#                 (tp[i] + tp[i - 1]) * (fp[i] - fp[i - 1]) / (self.P * self.N)
#                 for i in range(1, len(tp))
#             ]
#         )

#         return loss

#     def pach_lb(self, leaves, theta, Ps, Ns):
#         tp = np.array([0, Ps], dtype=float)
#         fp = np.array([0, 0], dtype=float)
#         i = 0
#         while fp[i + 1] < self.N * theta:
#             if i < len(leaves):
#                 tp = np.append(tp, tp[i + 1] + leaves[i].p)
#                 fp = np.append(fp, fp[i + 1] + leaves[i].n)
#                 i += 1
#             else:
#                 tp = np.append(tp, tp[i + 1] + 0)
#                 fp = np.append(fp, fp[i + 1] + Ns)
#                 break
#         tp[len(tp) - 1] = (
#             (tp[len(tp) - 1] - tp[len(tp) - 2]) / (fp[len(fp) - 1] - fp[len(fp) - 2])
#         ) * (self.N * theta - fp[len(fp) - 1]) + tp[len(tp) - 1]
#         fp[len(fp) - 1] = self.N * theta
#         loss = 1 - 0.5 * sum(
#             [
#                 (tp[i] + tp[i - 1]) * (fp[i] - fp[i - 1]) / (self.P * self.N)
#                 for i in range(1, len(tp))
#             ]
#         )
#         return loss


# class CacheTree:
#     def __init__(self, name, P, N, lamb, leaves, w=None, theta=None):
#         self.name = name
#         self.P = P
#         self.N = N
#         self.leaves = leaves
#         self.w = w
#         self.theta = theta

#         bound = Objective(name, P, N, lamb)
#         self.risk = bound.risk(self.leaves, self.w, self.theta)

#     def sorted_leaves(self):
#         return tuple(sorted(leaf.points_cap for leaf in self.leaves))


# class Tree:
#     def __init__(self, cache_tree, n, lamb, splitleaf=None, prior_metric=None):

#         self.cache_tree = cache_tree
#         self.splitleaf = splitleaf
#         leaves = cache_tree.leaves
#         self.H = len(leaves)
#         self.risk = cache_tree.risk

#         bound = Objective(cache_tree.name, cache_tree.P, cache_tree.N, lamb)
#         self.lb = bound.lower_bound(leaves, splitleaf, cache_tree.w, cache_tree.theta)

#         if leaves[0].num_captured == n:
#             self.metric = 0  # null tree
#         elif prior_metric == "objective":
#             self.metric = self.risk
#         elif prior_metric == "bound":
#             self.metric = self.lb
#         elif prior_metric == "curiosity":
#             removed_leaves = list(compress(leaves, splitleaf))  # dsplit
#             num_cap_rm = sum(
#                 leaf.num_captured for leaf in removed_leaves
#             )  # num captured by dsplit
#             if num_cap_rm < n:
#                 self.metric = self.lb / ((n - num_cap_rm) / n)  # supp(dun, xn)
#             else:
#                 self.metric = self.lb / (0.01 / n)  # null tree
#         elif prior_metric == "entropy":
#             removed_leaves = list(compress(leaves, splitleaf))
#             num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
#             # entropy weighted by number of points captured
#             self.entropy = [
#                 (
#                     -leaves[i].p * math.log2(leaves[i].p)
#                     - (1 - leaves[i].p) * math.log2(1 - leaves[i].p)
#                 )
#                 * leaves[i].num_captured
#                 if leaves[i].p != 0 and leaves[i].p != 1
#                 else 0
#                 for i in range(self.H)
#             ]
#             if num_cap_rm < n:
#                 self.metric = sum(
#                     self.entropy[i] for i in range(self.H) if splitleaf[i] == 0
#                 ) / (n - sum(leaf.num_captured for leaf in removed_leaves))
#             else:
#                 self.metric = (
#                     sum(self.entropy[i] for i in range(self.H) if splitleaf[i] == 0)
#                     / 0.01
#                 )
#         elif prior_metric == "gini":
#             removed_leaves = list(compress(leaves, splitleaf))
#             num_cap_rm = sum(leaf.num_captured for leaf in removed_leaves)
#             # gini index weighted by number of points captured
#             self.giniindex = [
#                 (2 * leaves[i].p * (1 - leaves[i].p)) * leaves[i].num_captured
#                 for i in range(self.H)
#             ]
#             if num_cap_rm < n:
#                 self.metric = sum(
#                     self.giniindex[i] for i in range(self.H) if splitleaf[i] == 0
#                 ) / (n - sum(leaf.num_captured for leaf in removed_leaves))
#             else:
#                 self.metric = (
#                     sum(self.giniindex[i] for i in range(self.H) if splitleaf[i] == 0)
#                     / 0.01
#                 )
#         elif prior_metric == "FIFO":
#             self.metric = 0
#         elif prior_metric == "random":
#             self.metric = np.random.uniform(0.0, 1.0)

#     def __lt__(self, other):
#         # define <, which will be used in the priority queue
#         return self.metric < other.metric


# class CacheLeaf:
#     def __init__(
#         self,
#         name,
#         n,
#         P,
#         N,
#         rules,
#         x,
#         y,
#         y_mpz,
#         z_mpz,
#         points_cap,
#         num_captured,
#         lamb,
#         support,
#         is_feature_dead,
#         w=None,
#     ):
#         self.rules = rules
#         self.points_cap = points_cap
#         self.num_captured = num_captured
#         self.is_feature_dead = is_feature_dead

#         _, num_ones = rule_vand(points_cap, y_mpz)  # return vand and cnt
#         _, num_errors = rule_vand(points_cap, z_mpz)
#         """
#         print('rules:', rules)
#         print("points_cap:", points_cap, "vec:", rule_mpztovec(points_cap))
#         print('_:', _, "vec:", rule_mpztovec(_))
#         print('num_errors',num_errors)
#         """
#         self.delta = num_errors
#         self.p = num_ones
#         self.n = self.num_captured - num_ones
#         if self.num_captured > 0:
#             self.r = num_ones / self.num_captured
#         else:
#             self.r = 0
#         bound = Objective(name, P, N, lamb)

#         if name == "auc_convex":
#             if num_errors > 0:
#                 cap = np.array(rule_mpztovec(points_cap))
#                 cap_i = np.where(cap == 1)[0]
#                 x_cap = x[cap_i]
#                 y_cap = y[cap_i]

#                 v = rule_mpztovec(_)
#                 equiv_i = np.where(np.array(v) == 1)[0]
#                 idx = [i for i, c in enumerate(cap_i) if c in equiv_i]
#                 idx = np.array(idx)

#                 unique_rows, counts = np.unique(x_cap[idx,], axis=0, return_counts=True)
#                 """
#                 print('cap_i:', cap_i)
#                 print("x_cap:", x_cap)
#                 print("y_cap:", y_cap)
#                 print("v:", v)
#                 print("idx:", idx)
#                 """
#                 nrow = unique_rows.shape[0]
#                 self.equiv = np.zeros((3, nrow + 2))

#                 for i in range(nrow):
#                     comp = np.all(np.equal(x_cap, unique_rows[i,]), axis=1)
#                     eu = np.sum(comp)
#                     j = np.where(comp == True)
#                     n_neg = np.sum(y_cap[j] == 0)
#                     n_pos = eu - n_neg
#                     self.equiv[0, i] = n_pos / eu  # r = n_pos/eu
#                     self.equiv[1, i] = n_pos
#                     self.equiv[2, i] = n_neg

#                 self.equiv[0, nrow] = 1
#                 # y_i = np.where(np.array(v)==0)[0]
#                 # equiv_not_i = [i for i,c in enumerate(cap_i) if c not in equiv_i]
#                 self.equiv[1, nrow] = sum(y_cap == 1) - sum(
#                     self.equiv[1, i] for i in range(nrow)
#                 )
#                 self.equiv[2, nrow + 1] = sum(y_cap == 0) - sum(
#                     self.equiv[2, i] for i in range(nrow)
#                 )
#             else:
#                 self.equiv = np.zeros((3, 2))
#                 self.equiv[0, 0] = 1
#                 self.equiv[1, 0] = self.p
#                 self.equiv[2, 1] = self.n

#         if self.num_captured:
#             self.pred = bound.leaf_predict(self.p, self.n, w)
#             if self.pred == 0:
#                 self.fp = 0
#                 self.fn = self.p
#                 self.delta_fp = 0
#                 self.delta_fn = self.delta
#             else:
#                 self.fp = self.n
#                 self.fn = 0
#                 self.delta_fp = self.delta
#                 self.delta_fn = 0
#         else:
#             self.pred = 0
#             self.fp = 0
#             self.fn = self.p


# class EquivPoints:
#     def __init__(self, r, p, n):
#         self.r = r
#         self.p = p
#         self.n = n


# def get_false(leaves):
#     FP = sum([l.fp for l in leaves])
#     FN = sum([l.fn for l in leaves])
#     return FP, FN


# def get_fixed_false(leaves, splitleaf):
#     FPu = sum([leaves[i].fp for i in range(len(leaves)) if splitleaf[i] == 0])
#     FNu = sum([leaves[i].fn for i in range(len(leaves)) if splitleaf[i] == 0])
#     return FPu, FNu


# def get_split(leaves, splitleaf):
#     Ps = sum([leaves[i].p for i in range(len(leaves)) if splitleaf[i] == 1])
#     Ns = sum([leaves[i].n for i in range(len(leaves)) if splitleaf[i] == 1])
#     return Ps, Ns


# def order_leaves(leaves):
#     return sorted([l for l in leaves], key=lambda x: x.r, reverse=True)


# def order_fixed_leaves(leaves, splitleaf):
#     leaf_fixed = [not split for split in splitleaf]
#     fixed_set = list(compress(leaves, leaf_fixed))
#     ordered_fixed = order_leaves(fixed_set)
#     return ordered_fixed


# def ach_equiv_lb(leaves, splitleaf, P, N, lamb):
#     split_set = list(compress(leaves, splitleaf))
#     leaf_fixed = [not split for split in splitleaf]
#     fixed_set = list(compress(leaves, leaf_fixed))
#     Ps1 = 0
#     Ns1 = 0
#     for i in split_set:
#         equiv = i.equiv  # equiv is the array row=3 and col=equivset+2
#         if equiv.shape[1] - 2 > 0:
#             for j in range(equiv.shape[1] - 2):
#                 leaf_equiv = EquivPoints(equiv[0, j], equiv[1, j], equiv[2, j])
#                 fixed_set.append(leaf_equiv)
#         Ps1 += equiv[1, equiv.shape[1] - 2]
#         Ns1 += equiv[2, equiv.shape[1] - 1]
#     ordered_leaves = order_leaves(fixed_set)

#     tp = np.array([0, Ps1], dtype=float)
#     fp = np.array([0, 0], dtype=float)
#     for i in range(len(ordered_leaves)):
#         tp = np.append(tp, tp[i + 1] + ordered_leaves[i].p)
#         fp = np.append(fp, fp[i + 1] + ordered_leaves[i].n)
#     tp = np.append(tp, tp[len(ordered_leaves) + 1] + 0)
#     fp = np.append(fp, fp[len(ordered_leaves) + 1] + Ns1)

#     loss = 1 - 0.5 * sum(
#         [(tp[i] + tp[i - 1]) * (fp[i] - fp[i - 1]) / (P * N) for i in range(1, len(tp))]
#     )
#     return loss + lamb * len(leaves)


# def generate_new_splitleaf(
#     name,
#     P,
#     N,
#     unchanged_leaves,
#     removed_leaves,
#     new_leaves,
#     lamb,
#     incre_support,
#     w,
#     theta,
# ):

#     n_removed_leaves = len(removed_leaves)  # dsplit
#     n_unchanged_leaves = len(unchanged_leaves)  # dun
#     n_new_leaves = len(new_leaves)
#     n_new_tree_leaves = n_unchanged_leaves + n_new_leaves  # H'
#     splitleaf1 = [0] * n_unchanged_leaves + [1] * n_new_leaves

#     bound = Objective(name, P, N, lamb)
#     FPu = None
#     FNu = None
#     ld = None
#     if name == "f1":
#         FPu, FNu = get_false(unchanged_leaves)
#     if name == "auc_convex":
#         _, ld = bound.ach_loss(unchanged_leaves + removed_leaves)
#     if name == "partial_auc":
#         _, ld = bound.pach_loss(unchanged_leaves + removed_leaves, theta)

#     sl = []
#     for i in range(n_removed_leaves):
#         splitleaf = [0] * n_new_tree_leaves
#         removed_l = removed_leaves[i]
#         new_l1 = new_leaves[2 * i]
#         new_l2 = new_leaves[2 * i + 1]
#         a = bound.incre_accu_bound(
#             removed_l,
#             new_l1,
#             new_l2,
#             unchanged_leaves,
#             removed_leaves,
#             w,
#             FPu,
#             FNu,
#             theta,
#             ld,
#         )
#         if not incre_support:
#             a = float("Inf")
#         if a <= lamb:
#             splitleaf[n_unchanged_leaves + 2 * i] = 1
#             splitleaf[n_unchanged_leaves + 2 * i + 1] = 1
#             sl.append(splitleaf)
#         else:
#             sl.append(splitleaf1)
#     return sl


# def get_cannot_split(name, P, N, lamb, m, new_tree_leaves, MAXDEPTH, w, theta):
#     bound = Objective(name, P, N, lamb)
#     FP = None
#     FN = None
#     ld = None
#     ordered_leaves = None
#     if name == "f1":
#         FP, FN = get_false(new_tree_leaves)
#         ld = bound.f1_loss(FP, FN)
#     if name == "auc_convex":
#         ordered_leaves, ld = bound.ach_loss(new_tree_leaves)
#     if name == "partial_auc":
#         ordered_leaves, ld = bound.pach_loss(new_tree_leaves, theta)

#     cannot_split = [
#         len(l.rules) >= MAXDEPTH
#         or bound.leaf_support_bound(l, w, theta, FP, FN, ld, ordered_leaves) < lamb
#         or all(
#             [
#                 l.is_feature_dead[r - 1]
#                 for r in range(1, m + 1)
#                 if r not in map(abs, l.rules)
#             ]
#         )
#         for l in new_tree_leaves
#     ]
#     return cannot_split


# def bbound(
#     x,
#     y,
#     name,
#     lamb,
#     prior_metric=None,
#     w=None,
#     theta=None,
#     MAXDEPTH=float("Inf"),
#     MAX_NLEAVES=float("Inf"),
#     niter=float("Inf"),
#     logon=False,
#     support=True,
#     incre_support=True,
#     accu_support=True,
#     equiv_points=True,
#     lookahead=True,
#     lenbound=True,
#     R_c0=1,
#     timelimit=float("Inf"),
#     init_cart=True,
#     saveTree=False,
#     readTree=False,
# ):

#     x0 = copy.deepcopy(x)
#     y0 = copy.deepcopy(y)

#     tic = time.time()

#     m = x.shape[1]  # number of features
#     n = len(y)
#     P = np.count_nonzero(y)
#     N = n - P

#     x_mpz = [rule_vectompz(x[:, i]) for i in range(m)]
#     y_mpz = rule_vectompz([i for i in y])

#     # order the columns by descending gini reduction
#     idx, dic = gini_reduction(x_mpz, y_mpz, n, range(m))
#     x = x[:, idx]
#     x_mpz = [x_mpz[i] for i in idx]

#     z_mpz = get_z(x, y, n, m)

#     # lines = []  # a list for log
#     leaf_cache = {}  # cache leaves
#     tree_cache = {}  # cache trees

#     # initialize the queue to include just empty root
#     queue = []
#     root_leaf = CacheLeaf(
#         name,
#         n,
#         P,
#         N,
#         (),
#         x,
#         y,
#         y_mpz,
#         z_mpz,
#         make_all_ones(n + 1),
#         n,
#         lamb,
#         support,
#         [0] * m,
#         w,
#     )
#     d_c = CacheTree(name, P, N, lamb=lamb, leaves=[root_leaf], w=w, theta=theta)
#     R_c = d_c.risk
#     tree0 = Tree(
#         cache_tree=d_c, n=n, lamb=lamb, splitleaf=[1], prior_metric=prior_metric
#     )
#     heapq.heappush(queue, (tree0.metric, tree0))

#     best_is_cart = False  # a flag for whether or not the best is the initial CART
#     if init_cart:
#         clf, nleaves_CART, trainout_CART, R_c, d_c, C_c = cart(
#             x0, y0, name, n, P, N, lamb, w, theta, MAXDEPTH
#         )
#         time_c = time.time() - tic
#         best_is_cart = True
#         # print("risk of cart:", R_c)
#     else:
#         C_c = 0
#         clf = None
#         time_c = time.time()

#     if readTree:
#         with open("tree.pkl", "rb") as f:
#             d_c = pickle.load(f)
#         R_c = d_c.risk

#         with open("leaf_cache.pkl", "rb") as f:
#             leaf_cache = pickle.load(f)

#         sorted_new_tree_rules = tuple(sorted(leaf.points_cap for leaf in d_c.leaves))
#         tree_cache[sorted_new_tree_rules] = True

#         tree_p = Tree(
#             cache_tree=d_c,
#             n=n,
#             lamb=lamb,
#             splitleaf=[1] * len(d_c.leaves),
#             prior_metric=prior_metric,
#         )

#         heapq.heappush(queue, (tree_p.metric, tree_p))
#         """
#         print("PICKEL>>>>>>>>>>>>>", [leaf.rules for leaf in d_c.leaves])
#         print('R_c:', R_c)
#         print('lower_bound:', tree_p.lb)
#         print('lookahead:',tree_p.lb+lamb*sum(tree_p.splitleaf))
#         """
#         # print("leaf_cache:", leaf_cache)

#         C_c = 0
#         time_c = time.time() - tic

#     if R_c0 < R_c:
#         R_c = R_c0

#     leaf_cache[make_all_ones(n + 1)] = root_leaf

#     COUNT = 0  # count the total number of trees in the queue
#     COUNT_POP = 0  # number of tree poped from queue (# of tree checked)
#     COUNT_UNIQLEAVES = 0
#     COUNT_LEAFLOOKUPS = 0

#     bound = Objective(name, P, N, lamb)

#     len_queue = []
#     time_queue = []
#     count_tree = []
#     time_realize_best_tree = [time_c]
#     R_best_tree = [R_c]

#     while queue and COUNT < niter and time.time() - tic < timelimit:
#         """
#         print(len(queue))
#         for metric, t in queue:
#             print(metric, [l.rules for l in t.cache_tree.leaves], t.splitleaf)
#         """
#         metric, tree = heapq.heappop(queue)

#         COUNT_POP = COUNT_POP + 1
#         count_tree.append(COUNT_POP)

#         leaves = tree.cache_tree.leaves
#         leaf_split = tree.splitleaf
#         removed_leaves = list(compress(leaves, leaf_split))
#         old_tree_length = len(leaf_split)
#         new_tree_length = len(leaf_split) + sum(leaf_split)

#         # prefix-specific upper bound on number of leaves
#         if lenbound and new_tree_length >= min(
#             old_tree_length + math.floor((R_c - tree.lb) / lamb) if lamb > 0 else 2**m, 2 ** m
#         ):
#             continue

#         n_removed_leaves = sum(leaf_split)
#         n_unchanged_leaves = old_tree_length - n_removed_leaves

#         # print("num in queue:", len(queue))
#         # print(time.time()-tic)
#         len_queue.append(len(queue))
#         time_queue.append(time.time() - tic)

#         """equivalent points bound + lookahead bound"""
#         lambbb = lamb if lookahead else 0

#         if (name != "auc_convex") & (name != "partial_auc"):
#             delta_fp = (
#                 sum([leaf.delta_fp for leaf in removed_leaves]) if equiv_points else 0
#             )
#             delta_fn = (
#                 sum([leaf.delta_fn for leaf in removed_leaves]) if equiv_points else 0
#             )
#             FPu, FNu = get_fixed_false(leaves, leaf_split)

#             # print("leaf:", [l.rules for l in leaves])
#             # print("leaf fp:", [l.p for l in leaves])
#             # print("leaf fn:", [l.n for l in leaves])
#             # print("leaf delta fp:", [l.delta_fp for l in leaves])
#             # print("leaf delta fn:", [l.delta_fn for l in leaves])
#             # print((delta_fp+delta_fn)/(P+N))
#             # print((FPu+FNu)/(P+N))
#             # print(bound.loss(FPu+delta_fp, FNu+delta_fn, w))
#             # print(n_removed_leaves * lambbb)
#             # print("R_c:", R_c)
#             # print(bound.loss(FPu+delta_fp, FNu+delta_fn, w) + (old_tree_length+n_removed_leaves) * lambbb, R_c)
#             # print(bound.loss(FPu+delta_fp, FPu+delta_fn, w)+ n_removed_leaves * lambbb >= R_c)

#         """
#         if (name != 'auc_convex') and (name != 'partial_auc'):
#             #skip.append(bound.loss(FPu+delta_fp, FNu+delta_fn, w)+ (old_tree_length+n_removed_leaves) * lambbb >= R_c)
#             print(bound.loss(FPu+delta_fp, FNu+delta_fn, w)+ (old_tree_length+n_removed_leaves) * lambbb >= R_c)
#         if (name == 'auc_convex' or name == 'partial_auc'):
#             #skip.append(tree.lb + n_removed_leaves * lambbb>= R_c)
#             print(tree.lb + n_removed_leaves * lambbb>= R_c)
#         """

#         if (
#             (name != "auc_convex")
#             and (name != "partial_auc")
#             and (
#                 bound.loss(FPu + delta_fp, FNu + delta_fn, w)
#                 + (old_tree_length + n_removed_leaves) * lambbb
#                 >= R_c
#             )
#         ):
#             continue

#         if name == "auc_convex":
#             if (
#                 ach_equiv_lb(leaves, leaf_split, P, N, lamb) + n_removed_leaves * lambbb
#                 >= R_c
#             ):
#                 continue

#         if (name == "partial_auc") and (tree.lb + n_removed_leaves * lambbb >= R_c):
#             continue

#         leaf_no_split = [not split for split in leaf_split]
#         unchanged_leaves = list(compress(leaves, leaf_no_split))

#         # Generate all assignments of rules to the leaves that are due to be split
#         rules_for_leaf = [
#             set(range(1, m + 1))
#             - set(map(abs, l.rules))
#             - set([i + 1 for i in range(m) if l.is_feature_dead[i] == 1])
#             for l in removed_leaves
#         ]

#         for leaf_rules in product(*rules_for_leaf):

#             if time.time() - tic >= timelimit:
#                 break

#             new_leaves = []
#             flag_increm = (
#                 False  # a flag for jump out of the loops (incremental support bound)
#             )
#             for rule, removed_leaf in zip(leaf_rules, removed_leaves):

#                 rule_index = rule - 1
#                 tag = (
#                     removed_leaf.points_cap
#                 )  # points captured by the leaf's parent leaf

#                 for new_rule in (-rule, rule):
#                     new_rule_label = int(new_rule > 0)
#                     new_rules = tuple(sorted(removed_leaf.rules + (new_rule,)))
#                     tag_rule = (
#                         x_mpz[rule_index]
#                         if new_rule_label == 1
#                         else ~(x_mpz[rule_index]) | mpz(pow(2, n))
#                     )
#                     new_points_cap, new_num_captured = rule_vand(tag, tag_rule)
#                     leaf_key = new_points_cap
#                     # print(leaf_key)

#                     if leaf_key not in leaf_cache:

#                         COUNT_UNIQLEAVES = COUNT_UNIQLEAVES + 1

#                         # parent_is_feature_dead =
#                         new_leaf = CacheLeaf(
#                             name,
#                             n,
#                             P,
#                             N,
#                             new_rules,
#                             x,
#                             y,
#                             y_mpz,
#                             z_mpz,
#                             new_points_cap,
#                             new_num_captured,
#                             lamb,
#                             support,
#                             removed_leaf.is_feature_dead.copy(),
#                             w,
#                         )
#                         leaf_cache[leaf_key] = new_leaf
#                         new_leaves.append(new_leaf)
#                     else:

#                         COUNT_LEAFLOOKUPS = COUNT_LEAFLOOKUPS + 1

#                         new_leaf = leaf_cache[leaf_key]
#                         new_leaves.append(new_leaf)

#                     """
#                     # Lower bound on classification accuracy
#                     # if (new_leaf.num_captured) / n <= lamb:
#                     # accu_support == theorem 9 in OSDT, check if feature dead, not derived yet
                    
#                     if accu_support == True and (new_leaf.num_captured - new_leaf.num_captured_incorrect) / n <= lamb:

#                         removed_leaf.is_feature_dead[rule_index] = 1

#                         flag_increm = True
#                         break
#                     """

#                 if flag_increm:
#                     break

#             if flag_increm:
#                 continue

#             new_tree_leaves = unchanged_leaves + new_leaves

#             sorted_new_tree_rules = tuple(
#                 sorted(leaf.points_cap for leaf in new_tree_leaves)
#             )

#             if sorted_new_tree_rules in tree_cache:
#                 continue
#             else:
#                 tree_cache[sorted_new_tree_rules] = True

#             child = CacheTree(name, P, N, lamb, new_tree_leaves, w=w, theta=theta)

#             # print([l.rules for l in child.leaves])

#             R = child.risk

#             # print("R:", R, "R_c:", R_c)
#             time_realize_best_tree.append(time.time() - tic)
#             R_best_tree.append(R)

#             if R < R_c:
#                 d_c = child
#                 R_c = R
#                 C_c = COUNT + 1
#                 time_c = time.time() - tic

#                 best_is_cart = False

#             # generate the new splitleaf for the new tree
#             sl = generate_new_splitleaf(
#                 name,
#                 P,
#                 N,
#                 unchanged_leaves,
#                 removed_leaves,
#                 new_leaves,
#                 lamb,
#                 incre_support,
#                 w,
#                 theta,
#             )  # a_j

#             cannot_split = get_cannot_split(
#                 name, P, N, lamb, m, new_tree_leaves, MAXDEPTH, w, theta
#             )

#             # For each copy, we don't split leaves which are not split in its parent tree.
#             # In this way, we can avoid duplications.
#             can_split_leaf = [(0,)] * n_unchanged_leaves + [
#                 (0,) if cannot_split[i] else (0, 1)
#                 for i in range(n_unchanged_leaves, new_tree_length)
#             ]
#             # Discard the first element of leaf_splits, since we must split at least one leaf
#             new_leaf_splits0 = np.array(
#                 list(product(*can_split_leaf))[1:]
#             )  # sorted(product(*can_split_leaf))[1:]
#             len_sl = len(sl)
#             if len_sl == 1:
#                 # Filter out those which split at least one leaf in dp (d0)
#                 new_leaf_splits = [
#                     ls for ls in new_leaf_splits0 if np.dot(ls, sl[0]) > 0
#                 ]
#             else:
#                 # Filter out those which split at least one leaf in dp and split at least one leaf in d0
#                 new_leaf_splits = [
#                     ls
#                     for ls in new_leaf_splits0
#                     if all([np.dot(ls, sl[i]) > 0 for i in range(len_sl)])
#                 ]

#             for new_leaf_split in new_leaf_splits:
#                 # construct the new tree
#                 tree_new = Tree(
#                     cache_tree=child,
#                     n=n,
#                     lamb=lamb,
#                     splitleaf=new_leaf_split,
#                     prior_metric=prior_metric,
#                 )
#                 """
#                 print('tree_lb:', round(tree_new.lb, 4), 
#                       'tree_risk:', round(tree.cache_tree.risk, 4))
#                 """
#                 # print('tree_rules_x8:', [l.rules for l in tree.cache_tree.leaves])

#                 # MAX Number of leaves
#                 if len(new_leaf_split) + sum(new_leaf_split) > MAX_NLEAVES:
#                     continue

#                 COUNT = COUNT + 1
#                 # print([l.rules for l in tree_new.cache_tree.leaves], tree_new.splitleaf)
#                 """
#                 if (COUNT <= 22):
#                     print([l.rules for l in tree_new.cache_tree.leaves], 
#                           tree_new.splitleaf, round(tree_new.lb, 4), 
#                           round(tree_new.cache_tree.risk,4), round(tree_new.metric, 4), 
#                           round(metric,4), [l.rules for l in tree.cache_tree.leaves])
                
#                 if (COUNT ==22)|(COUNT == 21)|(COUNT==20):
#                     for metric, t in queue:
#                         print(metric, [l.rules for l in t.cache_tree.leaves], t.splitleaf)
                   
#                 if COUNT == 22:
#                     print('123455667677')
#                     return
#                 """
#                 # heapq.heappush(queue, (2*tree_new.metric - R_c, tree_new))
#                 heapq.heappush(queue, (tree_new.metric, tree_new))

#                 if COUNT % 1000000 == 0:
#                     # print("COUNT:", COUNT)
#                     pass

#         # print('COUNT:', COUNT)

#     totaltime = time.time() - tic

#     if not best_is_cart:

#         accu = 1 - (R_c - lamb * len(d_c.leaves))

#         leaves_c = [leaf.rules for leaf in d_c.leaves]
#         pred_c = [leaf.pred for leaf in d_c.leaves]

#         num_captured = [leaf.num_captured for leaf in d_c.leaves]

#         # num_captured_incorrect = [leaf.num_captured_incorrect for leaf in d_c.leaves]

#         nleaves = len(leaves_c)
#     else:
#         accu = trainout_CART
#         leaves_c = "NA"
#         pred_c = "NA"
#         get_code(d_c, ["x" + str(i) for i in range(1, m + 1)], [0, 1])
#         num_captured = "NA"
#         # num_captured_incorrect = 'NA'
#         nleaves = nleaves_CART

#     if saveTree:
#         with open("tree.pkl", "wb") as f:
#             pickle.dump(d_c, f)
#         with open("leaf_cache.pkl", "wb") as f:
#             pickle.dump(leaf_cache, f)

#     """
#     print(">>> log:", logon)
#     print(">>> support bound:", support)
#     print(">>> accu_support:", accu_support)
#     print(">>> accurate support bound:", incre_support)
#     print(">>> equiv points bound:", equiv_points)
#     print(">>> lookahead bound:", lookahead)
#     print("prior_metric=", prior_metric)
#     """
#     # print("loss function:", name)
#     # print("lambda: ", lamb)
#     print("COUNT_UNIQLEAVES:", COUNT_UNIQLEAVES)
#     print("COUNT_LEAFLOOKUPS:", COUNT_LEAFLOOKUPS)
#     # print("total time: ", totaltime)
#     # print("leaves: ", leaves_c)
#     # print("num_captured: ", num_captured)
#     # print("prediction: ", pred_c)
#     # print("Objective: ", R_c)
#     # print("Accuracy: ", accu)
#     # print("COUNT of the best tree: ", C_c)
#     # print("time when the best tree is achieved: ", time_c)
#     print("TOTAL COUNT: ", COUNT)

#     return (
#         leaves_c,
#         pred_c,
#         dic,
#         nleaves,
#         m,
#         n,
#         totaltime,
#         time_c,
#         R_c,
#         COUNT,
#         C_c,
#         accu,
#         best_is_cart,
#         clf,
#         len_queue,
#         time_queue,
#         time_realize_best_tree,
#         R_best_tree,
#         count_tree,
#     )


# def predict(
#     name,
#     leaves_c,
#     prediction_c,
#     nleaves,
#     dic,
#     x,
#     y,
#     best_is_cart,
#     clf,
#     w=None,
#     theta=None,
# ):
#     """
#     :param leaves_c:
#     :param dic:
#     :return:
#     """
#     P = np.count_nonzero(y)
#     N = len(y) - P
#     if best_is_cart:
#         yhat = clf.predict(x)

#         n_fp = sum((yhat == 1) & (yhat != y))
#         n_fn = sum((yhat == 0) & (yhat != y))
#         n_tp = sum((yhat == 1) & (yhat == y))

#         if (name != "auc_convex") & (name != "partial_auc"):
#             out = 0.5 * (n_fp * n_tp + (n_tp + P) * (N - n_fp)) / (N * P)
#         elif name == "auc_convex":
#             nodes = clf.apply(x)
#             metric = np.empty([3, int(nleaves)], dtype=float)
#             for i, num in enumerate(set(nodes)):
#                 idx = np.where(nodes == num)
#                 metric[0, i] = sum(y[idx] == 1)
#                 metric[1, i] = sum(y[idx] == 0)
#                 metric[2, i] = sum(y[idx] == 1) / len(y[idx])
#             metric = metric[:, np.argsort(metric[2,])]
#             metric = np.flip(metric, axis=1)
#             metric = np.cumsum(metric, axis=1)
#             init = np.array([[0], [0], [0]])
#             metric = np.append(init, metric, axis=1)
#             tp = metric[0, :]
#             fp = metric[1, :]
#             out = 0.5 * sum(
#                 [
#                     (metric[0, i] / P + metric[0, i - 1] / P)
#                     * (metric[1, i] / N - metric[1, i - 1] / N)
#                     for i in range(1, len(leaves_c) + 1)
#                 ]
#             )
#         elif name == "partial_auc":
#             nodes = clf.apply(x)
#             metric = np.zeros([3, int(nleaves)], dtype=float)
#             for i, num in enumerate(set(nodes)):
#                 idx = np.where(nodes == num)
#                 metric[0, i] = sum(y[idx] == 1)
#                 metric[1, i] = sum(y[idx] == 0)
#                 metric[2, i] = sum(y[idx] == 1) / len(y[idx])
#                 # print(metric[0,i], metric[1,i], metric[2,i])
#             metric = metric[:, np.argsort(metric[2,])]
#             metric = np.flip(metric, axis=1)
#             metric = np.cumsum(metric, axis=1)
#             init = np.array([[0], [0], [0]])
#             metric = np.append(init, metric, axis=1)

#             tp = [0]
#             fp = [0]
#             i = 1
#             while fp[i - 1] < N * theta and i <= int(nleaves) + 1:
#                 tp.append(metric[0, i])
#                 fp.append(metric[1, i])
#                 i += 1

#             tp[i - 1] = ((tp[i - 1] - tp[i - 2]) / (fp[i - 1] - fp[i - 2])) * (
#                 N * theta - fp[i - 1]
#             ) + tp[i - 1]
#             fp[i - 1] = N * theta
#             out = 0.5 * sum(
#                 [
#                     (tp[i] + tp[i - 1]) * (fp[i] - fp[i - 1]) / (P * N)
#                     for i in range(1, len(tp))
#                 ]
#             )

#         print("Best is cart! Testing", "auc:", round(out, 4))
#         print("P=", P, "N=", N, "FP=", n_fp, "FN=", n_fn)
#         if (name == "auc_convex") or (name == "partial_auc"):
#             print("tp=", tp)
#             print("fp=", fp)
#             plt.plot([f / N for f in fp], [f / P for f in tp], "go-", linewidth=2)
#         print(">>>>>>>>>>>>>>>>>>>>>>>")

#         return yhat, out

#     n = x.shape[0]

#     caps = []

#     for leaf in leaves_c:
#         cap = np.array([1] * n)
#         for feature in leaf:
#             idx = dic[abs(feature)]
#             feature_label = int(feature > 0)
#             cap = (x[:, idx] == feature_label) * cap
#         caps.append(cap)

#     yhat = np.array([1] * n)

#     for j in range(len(caps)):
#         idx_cap = [i for i in range(n) if caps[j][i] == 1]
#         yhat[idx_cap] = prediction_c[j]

#     n_fp = sum((yhat == 1) & (yhat != y))
#     n_fn = sum((yhat == 0) & (yhat != y))
#     n_tp = sum((yhat == 1) & (yhat == y))

    
#     # if name == 'acc':
#     #     out = sklearn.metrics.accuracy_score(y, yhat)
#     # elif name == "bacc":
#     #     out = sklearn.metrics.balanced_accuracy_score(y, yhat)
#     # elif name == 'wacc':
#     #     out = 1-(n_fp + w*n_fn)/(len(y))
#     # elif name == 'f1':
#     #     out = sklearn.metrics.f1_score(y, yhat)
    
#     if (name != "auc_convex") & (name != "partial_auc"):
#         out = 0.5 * (n_fp * n_tp + (n_tp + P) * (N - n_fp)) / (N * P)
#     elif name == "auc_convex":
#         n_p = []
#         n_n = []
#         for i in range(len(caps)):
#             idx = np.where(caps[i] == 1)
#             n_p.append(sum(y[idx] == 1))
#             n_n.append(sum(y[idx] == 0))

#         metric = np.empty([3, len(caps)], dtype=float)
#         for i in range(len(caps)):
#             idx = np.where(caps[i] == 1)
#             metric[0, i] = sum(y[idx] == 1)
#             metric[1, i] = sum(y[idx] == 0)
#             metric[2, i] = sum(y[idx] == 1) / len(y[idx])

#         metric = metric[:, np.argsort(metric[2,])]
#         metric = np.flip(metric, axis=1)
#         metric = np.cumsum(metric, axis=1)
#         init = np.array([[0], [0], [0]])
#         metric = np.append(init, metric, axis=1)
#         tp = metric[0, :]
#         fp = metric[1, :]
#         out = 0.5 * sum(
#             [
#                 (metric[0, i] + metric[0, i - 1])
#                 * (metric[1, i] - metric[1, i - 1])
#                 / (P * N)
#                 for i in range(1, len(tp))
#             ]
#         )

#     elif name == "partial_auc":
#         metric = np.empty([3, len(caps)], dtype=float)
#         for i in range(len(caps)):
#             idx = np.where(caps[i] == 1)
#             metric[0, i] = sum(y[idx] == 1)
#             metric[1, i] = sum(y[idx] == 0)
#             metric[2, i] = sum(y[idx] == 1) / len(y[idx])

#         metric = metric[:, np.argsort(metric[2,])]
#         metric = np.flip(metric, axis=1)
#         metric = np.cumsum(metric, axis=1)
#         init = np.array([[0], [0], [0]])
#         metric = np.append(init, metric, axis=1)
#         tp = [0]
#         fp = [0]
#         i = 1
#         while fp[i - 1] < N * theta and i <= len(caps) + 1:
#             tp.append(metric[0, i])
#             fp.append(metric[1, i])
#             i += 1
#         tp[i - 1] = ((tp[i - 1] - tp[i - 2]) / (fp[i - 1] - fp[i - 2])) * (
#             N * theta - fp[i - 1]
#         ) + tp[i - 1]
#         fp[i - 1] = N * theta
#         out = 0.5 * sum(
#             [
#                 (tp[i] / P + tp[i - 1] / P) * (fp[i] / N - fp[i - 1] / N)
#                 for i in range(1, len(tp))
#             ]
#         )

#     print("Testing", "auc:", round(out, 4))
#     if name != "auc_convex" and name != "partial_auc":
#         print("P=", P, "N=", N, "FP=", n_fp, "FN=", n_fn)
#         plt.plot([0, n_fp / N, 1], [0, (P - n_fn) / P, 1], "go-", linewidth=2)
#     else:
#         # print("P=", P, "N=", N, "l.p=", n_p, "l.n=", n_n)
#         print("P=", P, "N=", N, "tp=", tp, "fp=", fp)
#         plt.plot([f / N for f in fp], [f / P for f in tp], "go-", linewidth=2)
#     print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

#     return yhat, out


