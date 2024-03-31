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
1. different combination of moe
    1. each moe with one random walk, each length with 20 <Done>
    2. each moe with several random walk, each length with 10 <Done>
    3. each moe with the same several random walk, each length with 10 <Done>
    3. each moe with a random subgraph, each length with 10 <Done>
    4. each moe with several random walk, each moe the random walk length is different <Done>
    5. above with a global model <Done>
    6. not random walk, only CNN of the subgraph, no moe. 
1. run exp
    1. Mamba_RandomGraph_NodeMulti
    2. Mamba_RandomWalk_NodeMulti
    3. Mamba_RandomWalk_MeanPooling_NodeMulti
    4. Mamba_RandomWalk_Staircase_NodeMulti
    5. Mamba_Several_RandomWalk_Staircase_NodeMulti
    6. Mamba_Several_Same_RandomWalk_NodeMulti
    7. Mamba_Hybrid_Degree_Noise_RandomWalk_NodeMulti
    8. Mamba_NodeGCN
    9. Mamba_Several_RandomWalk_NodeMulti
2. rank learnable
3. loss balance



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
