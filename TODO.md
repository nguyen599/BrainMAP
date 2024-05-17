# 5.7
## TODO
### Bad results
1. reason: 
   1. noisy
      1. co-train remove
   2. some high score
   3. metric bad 
      1. hit 10
      2. weighted 
   4. change explain



# 5.6
## Review
### What has been done
1. MoE: different input, localagg
2. Interpret: Lime, hidden space



# 5.4
## TODO
### Interpret
1. what posthoc
2. how to use them
3. how to apply them to me

## Problem
### Interpret
1. 一个batch输入不好
   1. transform the input as standard input






# 5.2
## Interpretability
### Simple
1. hidden 
   1. model
      1. output three layers of hidden output
         1. from mamba
         2. from output
         3. if eval and self.cfg.model..
   2. layer 
      1. output the hidden layer output
         1. if eval and self.cfg.model...
         2. return different stuffs
   3. train
      1. if eval and self.cfg.model...
   4. config
### Tool
### Learn








# 5.1
## Analysis
### Interpret
1. possible methods
   1. Shapley
   2. DeepLiftShap
   3. others in captum
2. module
   1. captum
   2. shap
   3. interpret for numpy
3. implement
   1. batch -> x, index, batch_ptr -> batch
   2. test, 改变接口


# 4.29
## Problem
### Overview
1. performance
   1. fair comparance (transformer)
   2. tune parameter
2. writing
   1. how's the story?
### shuffle?


## Exp
### Brain
1. GPS ok
2. Exphormer ok
2. Mamba different setting
   1. New lr ok
3. Mamba MoE
   1. MoE different input, katz, degree, ok
### Subgraph
4. Mamba Subgraph
### 验证katz bucket的效果
1. Mamba_Hybrid_Noise_KatzCentralityRank_Bucket3 效果很差
### LearnRank
1. gps_layer
   1. choose whether rank or random order.
   2. when random order, output the order score as well.
      1. random order, 先random 后算score
   3. node rank dict. save the best rank ever.
   4. if training and test split
   5. output opt rank score
2. gps_model
   1. after 40 epoch, learn rank
      1. 2 forward, each forward with different para, or modify config
      2. return changes after 40 epochs, have to calculate loss for two
         1. could always keep the 2 output
   2. if loss changes, could be the same
      1. opt rank score ,
      2. training 
3. train
   1. deal with multiple output
4. loss
   1. before 40 epoch average the two losses or only back for one
   2. after, choose the higher to back, make the order match a pre-defined order
   3. input is a batch, target_build
5. config
   1. a global 40 epoch config
s



## Thinking
1. 怎么整理实验？
   1. 跑完一些实验之后，把他们放到excel？
      1. 转移数据消耗时间。
   2. wandb
      1. 实验很乱。跑完之后也不知道自己跑的这个实验是什么，有很多细节调参数。
      2. description?字太多，具体细节改动，实验设计要规范，base parameters，改动，




# 4.28
## Analysis
### Brain
1. parcellation对我们有什么用？功能任务，我们能够给某一个区域一个比较高的分数。方法：分类，attention_weights
   1. multi-modal parcellation是怎么做的？task分类？脑区分类？



# 4.27
## TODO
1. dataset prepare
   1. BrainDataset in dataset
   2. whether random split the dataset with split_generator
      1. need to choose which template


# 4.25
## Problem
### dataset
1. any split?
   
## TODO
1. dataset prepare
   1. BrainDataset in dataset
   2. whether random split the dataset with split_generator
      1. need to choose which template



# 4.23
## Problem
### nan
1. nan is caused during the forward
   1. caused by mamba?
   2. possible reason
      1. overflow
         1. scale the input
### bad performance
1. architecture. 
   1. neighbors aggregation: after Atom.
      1. learnable?
   2. local order? length? random? 
   3. local neighbor layer-wise update?

## Exp
### Different combinations
1. 3 buckets fixed add
2. 3 buckets + heuRank
### Warmup MoE
### Graph domain prior
1. adding edges
2. adding subgraphs/motif/graphlets
### subgraph information incorporation (NAG)
1. neighbor hops 8 [v]
2. repete each neighbor hop
3. dropout 0.5 [v]
4. correct adj [v]
### NAG + MoE
1. no preserve previous hop + local then different rank global
2. no preserve previous hop + local then hop global
3. no preserve 
### how dropout works
1. dropout []



# 4.22

## Revise
### Progress: what useful?
1. double useful for node level
2. MoE might be useful for graph-level (choosing right MoE components)
### what not useful
1. A fixed flatten order
   1. not enough random

## Analysis
### Noise already act as a contrastive learner
### 每一个node作为一个mamba moe，这个时间没法接受
1. 可以试试每一个experts过一个hop的信息
   1. hop和hop之间没有联系
### How to connect neighbor to story?
1. flatten? 不同的flatten方式

## Exp 
### Different combinations
### Warmup MoE
### Graph domain prior
1. adding edges
2. adding subgraphs/motif/graphlets
### subgraph information incorporation (NAG)
1. changable places
   1. adj^n whether residue
   2. whether adj^n*x pass pre_mp
2. design
   1. flip + update neighbor
   2. non-flip
3. specific design with MoE
   1. one local | multiple MoE : different inputs
   2. multiple MoE: one local + different inputs


# 4.18

## Exp
### Single Flatten
1. why need noise?
   1. noise property, 重要nodes放前面，度高的放前面，有一个固有顺序。
   2. 完全噪声，不能利用到这个path的信息。模型的区分能力有限。
   3. 完全没有噪声，某个顺序过拟合。
2. 有没有可能不是noise的原因？而是node-level的原因
   1. 和node的context影响不大，增加context不会让效果变好。
### Noise Flatten
1. 顺序只能是一个很模糊的顺序，目前单纯的noise超过所有的固定顺序。
### Multiple Test
1. 没有什么影响。random的结果都差不多了。不同顺序之间的效果差异变得很小。
### Different combinations
1. single flatten
2. noise flatten
3. bucket flatten
### Weight from Rank Info
1. noise info
### warmup MoE


## Analysis
### 选择一个loss更低的可能只是过拟合
1. 如果在第一个顺序上train，那第二次算loss那也是第一个顺序低
2. contrastive learning
### sinlge flatten combination
1. 样本数太少了，所以用处不大， 固定的顺序没有什么用
2. 加了另外一个random的顺序也没有改，说明side random不会改变overfit的事实
### GCNrank，MoE比随机random效果好，每次会落到一个不同的固定顺序，导致结果不同


# 4.17
## Summarize Idea
### Context Window
1. birectional
2. Double - work for node level task
3. Graph-level Node-level task
### Context Order
1. Fixed context order
2. Random context order (Random type)
3. Heuristic context order
### Combining Context Order with MoE
### Information Block 
1. why context order is important
2. why random doesn't work well
### Graph domain prior
1. adding edges
2. adding subgraphs/motif/graphlets
### Contrasitve learning


## Problems
### Performance not well
1. rank is not important?
   1. the rank has to be very representative of the graph, otherwise become the noise, can't generalize
2. random can't perform well, due to different 
3. can't choose different combinations. They are not diverse enough。
4. 我做了充分的实验吗？（初步试验是层层递进，一开始先试探性跑，然后严谨做控制变量）
   1. rank importance，centrality score 全部先算出来。算一个就行了。
   2. negative的效果要分开表示。bidirectional
5. 如何不学习能够选出最好的参数。rank太heuristic了。
6. 如何从几万个顺序里面选出有用的。（参数空间）
7. 为什么rank比random好。degree是graph固有的性质
   1. 还有什么其他的固有性质吗？
   2. 固定的centrality
   3. learnable why work？ node 学一个分数？x xxx
8. 学习：定长学习。

## Exp
### Order Fix 1h写完 20min跑完 等1。30h
1. 在train/data—-loader中的某个位置加入这个参数，通过cfg控制加入的参数名字
   1. 在全图加入的话，可能比较大，速度很慢，最好一个batch一个batch加入
   2. 所以最好在data——loader出来之后加，单独加一个循环？在dataloader之前加，目的让每一个node都能有一个分数。如果图太大，那就一个图一个图的apply（学学position——e怎么做）
### Order Bucket with Random 30min写完，20min跑完 等1.30h
1. 分3个bucket， 10个bucket，100
2. heteo score也是一个重要的rank
### Then choose the most promising & diverse ranks 30min写完
1. 系统性实验: 最单独的，组合起来最好的，每个跑两次


# 4.15
## Idea
### 同一个mamba，不同顺序。
### learnable based on MLP choosing weights
### 不同mamba，同一个输入。
### 输入每次是一致的吗？

## Analysis
### 什么原因导致dgree好于random？degree好于random吗？




# 4.14

## TODO
### Mix-based methods
1. degree 


# 4.13

## TODO
1. summarize the thoughts from Tuesday
   1. Different routing methods (switch transformer, which token in experts)
   2. Different trick (transformer recipe)
   3. Different flatten methods (GraphCL)
   4. Learning methods. How to learn a rank? 
2. design experiements. read papers.
3. improve performance on other datasets.
4. see why learning ranking doesn't work, and possible solutions. warm up
### Flatten a graph
each run 2 times to prevent randomness
1. importance-based
   1. centrality methods: betweeness centrality, Eigenvector Centrality, Katz Centralit # 输入考虑排序
   2. GCNRank: consider neighbor information
2. similarity-based (similar nodes will be closer)
   1. random walk sample
   2. cluster the graph
3. functionality-based
   1. heterophily score
4. mix-based
   1. calculate a score from the score aggregation
   2. lexsort
5. how to design exp?
   1. each flatten manner with both random / not-random, no MoE
   2. MoE of flatten methods. How to add random? 
      1. random: 阶梯 random
      2. add another random path
   3. MiX of flatten methods, no MoE
   4. need to record very well
### Improving the results on other datasets
1. MoE
### See why permutation has different performances




# 4.7 

## Analysis 

### Why adding more components the results are worse for cluster and malnet-tiny?
1. edges are important? information flow along the edges.
    1. flatten edges as well. adding edges in the front.
### mamba doesn't have inductive bias and need it
1. reason
    1. relation among nodes are not consistent with edges. 
### Why order of flattening matter?
1. degree order
    1. node level task need this more. important nodes should be placed earlier. but those important nodes should have other nodes information as well -> bidirection
2. can we do completely random order?
    1. train loss / test loss
### Whether it's overfit
1. what's possible for overfit
    1. MoE
        1. Training loss and ap is smaller
    2. Order
        1. GCNRank
2. rank: GCNrank > diverserank > localaggranks = 12MoE
3. it might be also due to noise.GCNrank I don't have any noise. -> gcn rank might overfit
    1. bucket noise
4. MalNetTiny is easier to overfit, CLUSTER not overfit. 
5. maybe MoE learner should be simpler -> MLP.
### Is MoE useful?
1. MoE v.s. no MoE (with degrank), training ap √ test ap √
### How to learn rank
1. Bucket rank learning



## Running Exp
1. Mamba_DiverseRank_MLPMulti - to see if bad performance caused by GCN as rank in cluster
2. Mamba_DiverseRank_GCNMulti - now add random, increasing more noise
3. 


# 4.6
## Problem:
### Why the result of GCN rank (not learnt) is better?
1. integrate neighbor information.


## TODO
### 测试多一些MoE会不会效果更好
### 测试不同的flatten方式
1. Mamba_DegRank_GCNMulti
2. Mamba_PCARank_GCNMulti 
3. Mamba_GCNRank_GCNMulti <Good>
4. Mamba_HeteroRank_GCNMulti
5. Mamba_RandomRank_GCNMulti
6. Mamba_Deg2Rank_GCNMulti
7. Mamba_DiverseRank_GCNMulti <Good>
8. Mamba_LocalAgg_GCNMulti <Good>
9. Mamba_12MoE_GCNMulti



## Idea:
1. unsupervised pretraining to let mamba fully capture the graph structure 
2. hierachical mamba
3. 不同GCN过不同mamba
4. 不同Flatten的形式
    1. 异质性分数
    2. degree
    3. PPR
    4. 随机
    5. random walk
5. UNet mamba
6. Multi-hop GCN MOE with Learning Rank


## Find:
### Linear vs MLP vs GCN
1. The latter the better -> complexity
### The result of GMN is good? local aggregation based on mamba?






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


<!-- 1. different combination of moe
    1. each moe with one random walk, each length with 20 <Done>
    2. each moe with several random walk, each length with 10 <Done>
    3. each moe with the same several random walk, each length with 10 <Done>
    3. each moe with a random subgraph, each length with 10 <Done>
    4. each moe with several random walk, each moe the random walk length is different <Done>
    5. above with a global model <Done>
    6. not random walk, only CNN of the subgraph, no moe. <Done>
1. run exp
    1. Mamba_RandomGraph_NodeMulti
    2. Mamba_RandomWalk_NodeMulti
    3. Mamba_RandomWalk_MeanPooling_NodeMulti
    4. Mamba_RandomWalk_Staircase_NodeMulti
    5. Mamba_Several_RandomWalk_Staircase_NodeMulti
    6. Mamba_Several_Same_RandomWalk_NodeMulti
    7. Mamba_Hybrid_Degree_Noise_RandomWalk_NodeMulti
    8. Mamba_NodeGCN
    9. Mamba_Several_RandomWalk_NodeMulti -->