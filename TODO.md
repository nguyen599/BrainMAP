Problem:
1. subgraph for one node might be different for another node. -> hard for local encode
    1. direct graph message passing replace random walk
    2. random walk
2. Time consuming for such a big batch
    1. Parallel for MoE
        1. either for data or for simple implementation
    2. Try smaller datasets
    3. R


TODO:
1. node-level mamba
    1. 先找到稳定性能的node-level mamba
    2. 一定要结合上graph-mamba
    3. 
2. different combination of moe
3. rank learnable
4. loss balance



Idea:
1. unsupervised pretraining to let mamba fully capture the graph structure 
2. hierachical mamba


<!-- TODO:
1. 每一个顺序多个expert
2. 有顺序 / random
    1. MoE
3. sparse moe
4. 如果mamba-permute的效果比较差的话，考虑都重新用noise跑
    1. 如果bi/multi-bi的效果确实不是很好，那就是在训练的时候不能同时多个random的序列
    2. test1 is used to check whether the permutation in test give power. whether the low per is caused by this

Problem:
1. permutation 有用。random的permutation有用吗？random不就是在permutation嘛？
2. 有顺序的怎么加permutation？ bidirectional

Finding:
1. if trained on degree, it performs worse on random
2. permute 比noise的结果差？ -->

<!-- problems:
1. 加multi的结果很差
    1.  如果self.self_attn 里面加的是self.self_attn_那是好的结果
    2. 可能是device的问题 -->
<!-- 1. why do attn (mamba) need batch? for graph-level task?
    1. 128 * 389 (max graph node num in a batch) * 96
    2. mamba is nothing to do with node / graph level task -->



<!-- TODO:
1. 分配跑baseline结果。original / multi-head / average / MoE
    2. MoE should be n*hi, W is shared for all hi, 
        1. hi * W -->
