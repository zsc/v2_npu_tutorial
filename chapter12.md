# 第12章：TSP编译器技术

本章深入探讨Groq TSP（Tensor Streaming Processor）的编译器技术，重点关注静态调度、数据流图优化和自动并行化策略。TSP采用软件定义的确定性执行模型，通过编译时完成所有调度决策，消除运行时不确定性，这与传统GPU的动态调度形成鲜明对比。我们将分析如何通过编译器技术充分发挥数据流架构的优势，实现接近理论峰值的计算效率。

TSP编译器的核心理念是"编译时间换运行时间"。通过在编译阶段进行精确的时序分析和资源调度，生成完全确定的执行序列，避免了运行时的调度开销和不确定性。这种方法特别适合批处理推理场景，能够保证稳定的延迟和最大的硬件利用率。

## 12.1 静态调度算法

TSP的静态调度是整个编译器的核心，它将计算图映射到硬件资源上，生成精确到时钟周期的执行计划。这种调度方式要求编译器对硬件的每个细节都有准确的建模，包括计算单元延迟、内存访问时序、片上网络传输时间等。

### 12.1.1 TSP调度模型基础

TSP采用同步数据流（Synchronous Dataflow, SDF）模型作为调度的理论基础。在SDF模型中，每个计算节点的输入输出数据量在编译时已知，这使得编译器可以静态确定所有数据传输和计算的时序。

调度问题可以形式化为一个约束优化问题：

给定计算图 $G = (V, E)$，其中 $V$ 是计算节点集合，$E$ 是数据依赖边集合。对于每个节点 $v_i \in V$：
- 计算时间：$t_{comp}(v_i)$
- 输入数据量：$d_{in}(v_i)$
- 输出数据量：$d_{out}(v_i)$
- 资源需求：$r(v_i) = \{r_{alu}, r_{mem}, r_{noc}\}$

调度目标是找到一个映射函数 $\phi: V \rightarrow (P, T)$，将每个节点映射到处理器 $P$ 和时间槽 $T$，使得：

$$\min \max_{v_i \in V} \{T(v_i) + t_{comp}(v_i)\}$$

约束条件包括：
1. **依赖约束**：$\forall (v_i, v_j) \in E: T(v_j) \geq T(v_i) + t_{comp}(v_i) + t_{comm}(v_i, v_j)$
2. **资源约束**：$\forall t, p: \sum_{v_i: T(v_i) = t, P(v_i) = p} r(v_i) \leq R_{available}(p)$
3. **内存约束**：$\forall t: \sum_{v_i: T(v_i) \leq t < T(v_i) + t_{life}(v_i)} m(v_i) \leq M_{total}$

其中 $t_{comm}(v_i, v_j)$ 是节点间通信时间，取决于数据量和网络拓扑：

$$t_{comm}(v_i, v_j) = \frac{d_{out}(v_i)}{BW_{noc}} \times hop\_distance(P(v_i), P(v_j))$$

### 12.1.2 指令调度与资源分配

TSP的指令调度采用改进的列表调度（List Scheduling）算法，结合模拟退火（Simulated Annealing）进行优化。基本流程如下：

**阶段1：优先级计算**

为每个节点计算调度优先级，考虑关键路径长度和资源使用：

$$priority(v_i) = \alpha \cdot CP(v_i) + \beta \cdot \frac{r(v_i)}{R_{avg}} + \gamma \cdot fanout(v_i)$$

其中：
- $CP(v_i)$ 是从节点 $v_i$ 到输出的最长路径
- $R_{avg}$ 是平均资源使用量
- $fanout(v_i)$ 是节点的扇出度
- $\alpha, \beta, \gamma$ 是权重系数，典型值为 $(0.5, 0.3, 0.2)$

**阶段2：贪心调度**

```
算法：TSP贪心调度
1. 初始化ready_list为所有无前驱的节点
2. while ready_list不为空:
   3. 选择priority最高的节点v
   4. 找到最早可用的时间槽t和处理器p
   5. 检查资源约束：
      - ALU单元：每周期最多16个向量操作
      - 内存带宽：读写总带宽不超过2TB/s
      - NoC带宽：每个链路不超过400GB/s
   6. 分配v到(p, t)
   7. 更新ready_list
```

**阶段3：模拟退火优化**

通过随机交换和移动操作优化初始调度：

$$\Delta E = makespan_{new} - makespan_{old} + \lambda \cdot (utilization_{old} - utilization_{new})$$

接受概率：
$$P_{accept} = \begin{cases}
1 & \text{if } \Delta E < 0 \\
e^{-\Delta E / T} & \text{otherwise}
\end{cases}$$

温度下降策略：$T_{k+1} = 0.95 \cdot T_k$

### 12.1.3 寄存器分配策略

TSP的寄存器分配需要考虑向量寄存器文件的特殊结构。每个流处理器有320个向量寄存器，每个寄存器可存储320个FP16元素。

**活跃区间分析**

首先计算每个变量的活跃区间（Live Range）：

$$LR(v) = [def(v), last\_use(v)]$$

对于跨基本块的变量，需要进行数据流分析：

$$LIVE_{in}(B) = USE(B) \cup (LIVE_{out}(B) - DEF(B))$$
$$LIVE_{out}(B) = \bigcup_{S \in succ(B)} LIVE_{in}(S)$$

**图着色算法**

构建冲突图 $G_{conflict} = (V_{var}, E_{conflict})$，其中：
- 节点是变量
- 边表示两个变量的活跃区间重叠

使用Chaitin-Briggs算法进行着色：

```
算法：寄存器分配
1. 构建冲突图
2. while 图中有节点:
   3. if 存在度数 < K的节点v:
      4. 将v压入栈，从图中删除v
   5. else:
      6. 选择spill代价最小的节点v
      7. spill_cost(v) = (load_cost + store_cost) × freq(v) / degree(v)
      8. 标记v为spill，删除v
3. 弹栈着色
```

**寄存器重命名优化**

为减少false依赖，采用寄存器重命名：

$$rename(v_{old}) \rightarrow v_{new} \text{ if } v_{old} \notin LIVE_{out}(block)$$

### 12.1.4 内存布局优化

TSP的片上SRAM采用分布式设计，共有230MB容量，分为144个bank。内存布局优化的目标是最小化bank冲突和最大化数据局部性。

**Bank分配策略**

采用素数取模哈希减少冲突：

$$bank\_id = (addr / block\_size) \mod P$$

其中 $P$ 是接近bank数量的素数（如139）。

**数据布局变换**

对于矩阵乘法 $C = A \times B$，采用分块布局：

原始布局：$A[M][K], B[K][N], C[M][N]$

优化布局：
$$A_{tiled}[M/T_m][K/T_k][T_m][T_k]$$
$$B_{tiled}[K/T_k][N/T_n][T_k][T_n]$$
$$C_{tiled}[M/T_m][N/T_n][T_m][T_n]$$

块大小选择考虑SRAM容量：
$$T_m \times T_k + T_k \times T_n + T_m \times T_n \leq \frac{SRAM_{size}}{3 \times sizeof(fp16)}$$

典型值：$T_m = T_n = 320, T_k = 640$（对应20KB + 40KB + 20KB = 80KB）

**预取调度**

使用双缓冲隐藏内存延迟：

$$t_{prefetch}(block_{i+1}) = t_{compute}(block_i) - t_{transfer}$$

确保：$t_{transfer} \leq t_{compute}$ 以完全隐藏传输延迟。

## 12.2 数据流图优化

数据流图优化是TSP编译器提升性能的关键环节。通过识别和消除冗余计算、优化数据传输路径、融合相邻操作，可以显著减少计算量和内存访问，提高硬件利用率。

### 12.2.1 数据流图表示与构建

TSP编译器使用静态单赋值（Static Single Assignment, SSA）形式的数据流图作为中间表示。每个节点代表一个操作，边代表数据依赖关系。

**图结构定义**

数据流图 $DFG = (N, E, A)$：
- $N$：节点集合，每个节点 $n_i$ 表示一个操作
- $E$：有向边集合，$(n_i, n_j) \in E$ 表示 $n_j$ 依赖 $n_i$ 的输出
- $A$：属性集合，包括操作类型、数据类型、张量形状等

节点属性：
```
Node = {
  op_type: {GEMM, Conv, Add, ReLU, ...}
  input_shapes: List[TensorShape]
  output_shape: TensorShape
  compute_cost: cycles
  memory_footprint: bytes
  fusion_compatible: List[op_type]
}
```

**从神经网络到DFG的转换**

以Transformer的注意力机制为例：

原始计算：
$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

转换为DFG节点序列：
1. $n_1$: GEMM($Q$, $K^T$) → $S$
2. $n_2$: Scale($S$, $1/\sqrt{d_k}$) → $S'$
3. $n_3$: Softmax($S'$) → $A$
4. $n_4$: GEMM($A$, $V$) → $O$

数据依赖边：
- $(n_1, n_2)$: 传输 $S$ 矩阵
- $(n_2, n_3)$: 传输 $S'$ 矩阵
- $(n_3, n_4)$: 传输 $A$ 矩阵

**关键路径分析**

使用拓扑排序和动态规划计算关键路径：

$$CP(n) = \begin{cases}
0 & \text{if } n \text{ is input} \\
\max_{p \in pred(n)} \{CP(p) + latency(p) + comm(p,n)\} & \text{otherwise}
\end{cases}$$

关键路径长度决定了理论最小执行时间：
$$T_{min} = \max_{n \in N} \{CP(n) + latency(n)\}$$

### 12.2.2 公共子表达式消除

公共子表达式消除（Common Subexpression Elimination, CSE）识别并复用重复计算，减少冗余操作。

**哈希签名算法**

为每个操作生成唯一签名：

$$hash(n) = hash_{combine}(op\_type, hash(inputs), attributes)$$

使用Merkle树结构递归计算：
```
算法：CSE哈希签名
1. for each node n in topological_order:
   2. sig = hash(n.op_type)
   3. for each input i of n:
      4. sig = sig ⊕ (hash(i) << rotation)
   5. sig = sig ⊕ hash(n.attributes)
   6. signature_map[sig].append(n)
```

**等价性验证**

两个节点等价的充要条件：
1. 操作类型相同
2. 输入张量形状相同
3. 数值属性相同（如卷积步长、填充等）
4. 输入来源等价（递归定义）

对于浮点运算，考虑数值稳定性：
$$|result_1 - result_2| < \epsilon \cdot \max(|result_1|, |result_2|)$$

其中 $\epsilon = 2^{-10}$ 对于FP16。

**CSE收益评估**

消除节点 $n$ 的收益：

$$benefit(n) = freq(n) \times (cost_{compute}(n) + cost_{memory}(n)) - cost_{forward}(n)$$

其中：
- $freq(n)$：节点执行频率（批处理中的重复次数）
- $cost_{forward}(n)$：转发结果的通信开销

只有当 $benefit(n) > threshold$ 时才执行CSE。

### 12.2.3 死代码消除

死代码消除（Dead Code Elimination, DCE）移除不影响最终输出的计算。

**活跃性分析**

从输出节点反向标记活跃节点：

```
算法：标记活跃节点
1. worklist = output_nodes
2. while worklist不为空:
   3. n = worklist.pop()
   4. mark n as live
   5. for each input i of n:
      6. if i not marked:
         7. worklist.push(i)
```

**副作用处理**

某些操作即使输出未使用也不能消除：
- 内存写操作（store、scatter）
- 同步操作（barrier、fence）
- 调试输出（print、assert）

标记具有副作用的节点：
$$has\_side\_effect(n) = n.op \in \{store, print, barrier\} \vee \exists_{child} has\_side\_effect(child)$$

**激进死代码消除**

使用值编号（Value Numbering）进行更激进的优化：

$$VN(n) = \begin{cases}
const\_fold(n) & \text{if all inputs are constant} \\
VN(n.input) & \text{if } n \text{ is identity op} \\
\perp & \text{if } n \text{ is dead}
\end{cases}$$

### 12.2.4 循环优化技术

循环是NPU计算的主要模式，优化循环结构对性能至关重要。

**循环展开（Loop Unrolling）**

展开因子选择基于向量宽度和寄存器压力：

$$unroll\_factor = \min\left(\frac{VectorWidth}{DataWidth}, \frac{RegisterFile}{LiveVariables}\right)$$

对于TSP的320宽向量单元处理FP16：
$$unroll\_factor = \min(320, \frac{320}{active\_vars})$$

展开后的循环体：
```
原始：for i in range(N):
        C[i] = A[i] + B[i]

展开4次：for i in range(0, N, 4):
          C[i:i+4] = A[i:i+4] + B[i:i+4]  // 向量化
```

**循环分块（Loop Tiling）**

多维循环的分块策略，以矩阵乘法为例：

$$\begin{aligned}
&\text{for } i_o \in [0, M, T_m]: \\
&\quad \text{for } j_o \in [0, N, T_n]: \\
&\quad\quad \text{for } k_o \in [0, K, T_k]: \\
&\quad\quad\quad \text{// 内部循环完全展开或向量化} \\
&\quad\quad\quad C[i_o:i_o+T_m][j_o:j_o+T_n] += \\
&\quad\quad\quad\quad A[i_o:i_o+T_m][k_o:k_o+T_k] \times B[k_o:k_o+T_k][j_o:j_o+T_n]
\end{aligned}$$

块大小优化目标函数：
$$\min_{T_m,T_n,T_k} \left( \frac{MNK}{T_m T_n T_k} \times t_{tile} + t_{overhead} \right)$$

约束：$T_m \times T_k + T_k \times T_n + T_m \times T_n \leq SRAM_{size}$

**循环融合（Loop Fusion）**

融合条件判断：
1. **依赖兼容**：不存在反向依赖
2. **迭代空间对齐**：循环边界相同或可调整
3. **资源不超限**：融合后资源使用不超过硬件限制

融合收益模型：
$$gain = saved\_memory\_traffic - alignment\_overhead$$

其中：
$$saved\_memory\_traffic = size(intermediate\_tensor) \times 2 \times bandwidth\_cost$$
$$alignment\_overhead = \begin{cases}
0 & \text{if perfectly aligned} \\
padding\_size \times element\_size & \text{otherwise}
\end{cases}$$

**循环交换（Loop Interchange）**

通过交换循环顺序优化内存访问模式：

原始（列优先访问，cache不友好）：
```
for j in range(N):
    for i in range(M):
        process(A[i][j])  // 跨步访问
```

交换后（行优先访问，cache友好）：
```
for i in range(M):
    for j in range(N):
        process(A[i][j])  // 连续访问
```

局部性评分：
$$locality\_score = \frac{consecutive\_accesses}{total\_accesses} \times cache\_hit\_rate$$

选择使 $locality\_score$ 最大的循环顺序。

## 12.3 自动并行化

TSP编译器的自动并行化技术是实现高性能的关键。通过自动识别并行机会、划分计算任务、协调多个处理单元，编译器能够充分利用TSP的大规模并行计算资源。

### 12.3.1 数据并行识别

数据并行是最常见的并行模式，适用于对不同数据执行相同操作的场景。

**并行性分析**

使用多面体模型（Polyhedral Model）分析循环并行性：

对于仿射循环：
$$\vec{i} \in \mathcal{D} = \{\vec{i} | A\vec{i} \geq \vec{b}\}$$

依赖关系表示为：
$$\vec{i} \rightarrow \vec{j} \text{ if } \exists \vec{i}, \vec{j} \in \mathcal{D}: W(\vec{i}) \cap R(\vec{j}) \neq \emptyset \wedge \vec{i} \prec \vec{j}$$

其中 $W(\vec{i})$ 是迭代 $\vec{i}$ 的写集合，$R(\vec{j})$ 是迭代 $\vec{j}$ 的读集合。

**依赖距离向量**

计算依赖距离向量判断并行性：

$$\vec{d} = \vec{j} - \vec{i}$$

如果所有依赖的距离向量在某个维度 $k$ 上满足 $d_k = 0$，则该维度可以并行化。

例如，对于矩阵加法：
```
for i in range(M):
    for j in range(N):
        C[i][j] = A[i][j] + B[i][j]
```

依赖分析显示两个维度都可以并行：$d_i = 0, d_j = 0$。

**批处理维度并行**

神经网络推理的批处理维度天然并行：

$$Y[b][c][h][w] = \sum_k W[c][k] \cdot X[b][k][h'][w']$$

批处理维度 $b$ 无跨样本依赖，可完全并行化：

并行效率：
$$\eta_{batch} = \frac{T_{seq}}{T_{parallel} \times P} = \frac{B \times t_{single}}{(B/P + overhead) \times P}$$

当 $B \gg P$ 时，$\eta_{batch} \approx 1 - \frac{overhead \times P}{B \times t_{single}}$

**空间维度并行**

卷积的空间维度并行化策略：

输出分片：将输出特征图划分为多个tile
$$Output[h_s:h_e][w_s:w_e] = Conv(Input[h_s-p:h_e+p][w_s-p:w_e+p], Weight)$$

其中 $p = \lfloor kernel\_size / 2 \rfloor$ 是padding。

Halo交换开销：
$$overhead_{halo} = 2p \times (tile\_height + tile\_width) \times channels \times sizeof(fp16)$$

最优tile大小：
$$tile\_size_{opt} = \sqrt{\frac{SRAM_{per\_core}}{channels \times sizeof(fp16) \times (1 + \frac{2p}{tile\_size})}}$$

### 12.3.2 流水线并行

流水线并行将计算划分为多个阶段，不同阶段同时处理不同批次的数据。

**阶段划分算法**

目标：均衡各阶段计算时间，最小化流水线延迟。

给定 $L$ 层网络，划分为 $P$ 个阶段，优化目标：

$$\min \max_{p \in [1,P]} \sum_{l \in stage_p} t_l$$

约束：$\sum_{l \in stage_p} m_l \leq M_{per\_stage}$

使用动态规划求解：
$$DP[i][p] = \min_{j<i} \{\max(DP[j][p-1], \sum_{k=j+1}^{i} t_k)\}$$

**流水线调度**

1F1B（One Forward One Backward）调度策略：

时间步 $t$ 时，阶段 $p$ 处理的微批次：
$$microbatch_{p,t} = \begin{cases}
t - p & \text{if } t \geq p \text{ (forward)} \\
t - 2P + p + 1 & \text{if } t \geq 2P - p - 1 \text{ (backward)}
\end{cases}$$

流水线效率：
$$\eta_{pipe} = \frac{n \times \sum_l t_l}{(n + P - 1) \times \max_p \sum_{l \in stage_p} t_l}$$

当 $n \gg P$ 时，$\eta_{pipe} \approx \frac{\text{average stage time}}{\text{max stage time}}$

**气泡优化**

减少流水线气泡的策略：

1. **微批次数量优化**：
   $$n_{opt} = \arg\min_n \left\{\frac{P-1}{n} + \frac{memory\_usage(n)}{memory\_limit}\right\}$$

2. **交错调度**：
   将每个阶段分为两个虚拟阶段，减少气泡：
   $$bubble_{interleaved} = \frac{bubble_{naive}}{2}$$

3. **异步通信隐藏**：
   计算与通信重叠：
   $$t_{effective} = \max(t_{compute}, t_{communicate})$$

### 12.3.3 模型并行策略

模型并行将单个操作分解到多个处理器上执行，适用于超大模型。

**张量并行**

矩阵乘法的列并行分解：

$$Y = XW = X[W_1 | W_2 | ... | W_P] = [XW_1 | XW_2 | ... | XW_P]$$

每个处理器计算：
$$Y_p = XW_p, \quad W_p \in \mathbb{R}^{d_{in} \times (d_{out}/P)}$$

通信模式：
- 前向传播：broadcast $X$，无需通信聚合
- 反向传播：all-reduce梯度

通信量：
$$comm_{tensor} = 2 \times batch \times d_{in} \times sizeof(fp16)$$

**注意力机制并行**

多头注意力的头并行：

$$MultiHead(Q,K,V) = Concat(head_1, ..., head_H)W^O$$

每个处理器计算 $H/P$ 个注意力头：
$$head_{p} = Attention(QW_p^Q, KW_p^K, VW_p^V)$$

优势：
- 头之间完全独立，无需通信
- 只在最后的线性投影需要all-reduce

**2D并行分解**

对于超大矩阵，采用2D分解：

$$C = AB = \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}$$

处理器 $(i,j)$ 计算：
$$C_{ij} = \sum_{k} A_{ik}B_{kj}$$

通信模式：
- 行广播：每个 $A_{ik}$ 在行内广播
- 列广播：每个 $B_{kj}$ 在列内广播
- 结果累加：同位置reduce

总通信量（$P = \sqrt{P_{total}}$）：
$$comm_{2D} = 2 \times \frac{matrix\_size}{\sqrt{P}}$$

相比1D并行减少 $\sqrt{P}$ 倍通信。

### 12.3.4 混合并行优化

实际应用中，通常结合多种并行策略以达到最佳性能。

**并行策略选择**

基于模型特征和硬件参数自动选择：

```
决策树：
if model_size > memory_per_device:
    使用模型并行
    if batch_size > threshold:
        添加数据并行
else:
    if num_devices > num_layers:
        使用流水线并行 + 数据并行
    else:
        纯数据并行
```

**成本模型**

综合考虑计算、通信、内存的成本模型：

$$Cost_{total} = \alpha \cdot T_{compute} + \beta \cdot T_{comm} + \gamma \cdot M_{usage}$$

其中：
- $T_{compute} = \frac{FLOPs}{throughput \times utilization}$
- $T_{comm} = \frac{data_{transfer}}{bandwidth \times efficiency}$
- $M_{usage} = max(M_{activation}, M_{weight}, M_{gradient})$

**自动搜索算法**

使用强化学习搜索最优并行策略：

状态空间：$S = \{data\_parallel, tensor\_parallel, pipeline\_parallel\}$
动作空间：$A = \{increase, decrease, maintain\}$ for each dimension
奖励函数：$R = -Cost_{total}$

搜索算法：
```
算法：并行策略搜索
1. 初始化策略 π
2. for episode in range(max_episodes):
   3. s = random_initial_state()
   4. for step in range(max_steps):
      5. a = π(s) + exploration_noise
      6. s' = apply_action(s, a)
      7. r = evaluate_performance(s')
      8. update_policy(π, s, a, r, s')
      9. s = s'
```

**负载均衡**

动态负载均衡策略：

工作窃取（Work Stealing）：
$$\text{if } queue_{self}.empty() \text{ and } \exists p: |queue_p| > threshold:$$
$$\quad steal(queue_p.pop\_half())$$

负载评估指标：
$$imbalance = \frac{\max_p(workload_p) - \min_p(workload_p)}{avg(workload)}$$

目标：保持 $imbalance < 0.1$。

## 本章小结

本章深入探讨了TSP编译器的核心技术，包括静态调度、数据流图优化和自动并行化三大关键组件。

**关键要点回顾：**

1. **静态调度算法**：TSP采用编译时完全确定的调度策略，通过精确的硬件建模和约束优化，生成时钟级精确的执行计划。核心公式包括：
   - 调度目标函数：$\min \max_{v_i \in V} \{T(v_i) + t_{comp}(v_i)\}$
   - 依赖约束：$T(v_j) \geq T(v_i) + t_{comp}(v_i) + t_{comm}(v_i, v_j)$
   - 内存布局优化：$T_m \times T_k + T_k \times T_n + T_m \times T_n \leq SRAM_{size}$

2. **数据流图优化**：通过CSE、DCE和循环优化减少冗余计算，提高硬件利用率：
   - CSE收益模型：$benefit(n) = freq(n) \times cost(n) - overhead$
   - 循环分块约束：块大小受限于片上SRAM容量
   - 循环融合条件：依赖兼容、迭代空间对齐、资源不超限

3. **自动并行化**：编译器自动识别并利用多种并行机会：
   - 数据并行效率：$\eta_{batch} \approx 1 - \frac{overhead \times P}{B \times t_{single}}$
   - 流水线效率：$\eta_{pipe} \approx \frac{\text{average stage time}}{\text{max stage time}}$
   - 2D并行通信优化：相比1D减少 $\sqrt{P}$ 倍通信量

**与传统GPU编译器的主要区别：**
- 确定性执行 vs 动态调度
- 编译时间换运行时间的设计理念
- 无需运行时调度器，减少控制开销
- 更精确的性能预测和优化

## 练习题

### 基础题（理解概念）

**练习12.1** 给定一个计算图，包含4个节点：A(输入)→B(GEMM, 100 cycles)→C(ReLU, 10 cycles)→D(输出)，B→D也有直接连接(Residual)。假设通信时间为20 cycles，计算该图的关键路径长度。

<details>
<summary>答案</summary>

关键路径是从输入到输出的最长路径。有两条路径：
- 路径1：A → B → C → D，长度 = 100 + 20 + 10 + 20 = 150 cycles
- 路径2：A → B → D，长度 = 100 + 20 = 120 cycles

关键路径长度 = max(150, 120) = 150 cycles
</details>

**练习12.2** 对于矩阵乘法 $C[1024][1024] = A[1024][512] \times B[512][1024]$，使用FP16数据类型，片上SRAM为230MB。计算最优的分块大小 $(T_m, T_n, T_k)$。

<details>
<summary>答案</summary>

约束：$T_m \times T_k + T_k \times T_n + T_m \times T_n \leq \frac{230 \times 10^6}{2}$ bytes

设 $T_m = T_n = T$（对称分块），则：
$2T \times T_k + T^2 \leq 115 \times 10^6$

为最大化块大小，设 $T_k = 2T$：
$4T^2 + T^2 = 5T^2 \leq 115 \times 10^6$
$T \leq \sqrt{23 \times 10^6} \approx 4796$

考虑向量宽度320的倍数：$T_m = T_n = 4800, T_k = 9600$

验证：$(4800 \times 9600 + 9600 \times 4800 + 4800 \times 4800) \times 2 = 230.4MB$ ≈ 230MB ✓
</details>

**练习12.3** 某循环的迭代次数为1000，每次迭代需要5个活跃变量。TSP有320个向量寄存器，向量宽度320。计算最优的循环展开因子。

<details>
<summary>答案</summary>

展开因子 = $\min(\frac{320}{1}, \frac{320}{5}) = \min(320, 64) = 64$

但需要考虑迭代次数1000不能被64整除，实际展开因子选择：
- 50（1000/50 = 20，整除）
- 40（1000/40 = 25，整除）
- 25（1000/25 = 40，整除）

选择40或50都合理，取决于其他优化目标。
</details>

### 挑战题（深入分析）

**练习12.4** 设计一个调度算法，将Transformer的一个encoder block映射到4个TSP核心上。Block包含：Multi-Head Attention (MHA)、LayerNorm1、FFN、LayerNorm2。给出并行策略和通信模式。

<details>
<summary>答案</summary>

策略1：流水线并行
- Core 0: MHA的QKV投影
- Core 1: Attention计算 + 输出投影
- Core 2: LayerNorm1 + FFN第一层
- Core 3: FFN第二层 + LayerNorm2

通信模式：
- Core 0→1: QKV矩阵，$3 \times batch \times seq \times d_{model} \times 2$ bytes
- Core 1→2: Attention输出，$batch \times seq \times d_{model} \times 2$ bytes
- Core 2→3: FFN中间激活，$batch \times seq \times d_{ff} \times 2$ bytes

策略2：张量并行（推荐）
- 每个核心处理 $d_{model}/4$ 维度
- MHA：每核心处理 $num\_heads/4$ 个注意力头
- FFN：列并行分解
- 通信：每层后all-reduce，共4次

比较：张量并行更均衡，通信量更小。
</details>

**练习12.5** 推导2:4稀疏矩阵乘法在TSP上的理论加速比。考虑稀疏索引开销和向量单元利用率。

<details>
<summary>答案</summary>

密集GEMM计算量：$2MNK$ FLOPs
2:4稀疏GEMM计算量：$MNK$ FLOPs（50%稀疏）

但需考虑：
1. 索引开销：每4个元素需2bit索引，开销率 = $\frac{2}{4 \times 16} = 3.125\%$
2. 向量利用率：2:4模式下，向量单元利用率 ≈ 75%（需要mask操作）
3. 内存带宽：减少50%权重读取，但增加索引读取

理论加速比：
$$Speedup = \frac{1}{0.5 \times \frac{1}{0.75} + 0.03125} = \frac{1}{0.667 + 0.031} \approx 1.43$$

实际加速比约1.4-1.5倍。
</details>

**练习12.6** 分析Flash Attention在TSP编译器中的实现策略。给出分块大小选择和SRAM分配方案。

<details>
<summary>答案</summary>

Flash Attention核心思想：分块计算注意力，减少HBM访问。

TSP上的分块策略：
- 序列长度分块：$B_r = B_c = \sqrt{\frac{SRAM_{size}}{4 \times d \times sizeof(fp16)}}$
- 对于230MB SRAM，$d=128$：$B_r = B_c = \sqrt{\frac{230 \times 10^6}{4 \times 128 \times 2}} \approx 1500$

SRAM分配（总计230MB）：
- Q块：$B_r \times d \times 2 = 1500 \times 128 \times 2 = 384KB$
- K块：$B_c \times d \times 2 = 384KB$
- V块：$B_c \times d \times 2 = 384KB$
- S矩阵：$B_r \times B_c \times 2 = 4.5MB$
- 输出O：$B_r \times d \times 2 = 384KB$
- 统计量(row_max, row_sum)：$B_r \times 4 = 6KB$

总计：约6MB per block，可同时处理多个block提高并行度。
</details>

**练习12.7** 设计一个启发式算法，自动选择数据并行、模型并行和流水线并行的组合。输入：模型大小M、批大小B、设备数P、设备内存C。

<details>
<summary>答案</summary>

```
算法：混合并行策略选择
输入：M(模型大小), B(批大小), P(设备数), C(设备内存)

1. 计算模型并行度需求：
   mp = ceil(M / C)  // 最小模型并行度
   
2. 剩余并行度用于数据并行和流水线并行：
   P_remain = P / mp
   
3. 决策树：
   if B >= P_remain * 4:  // 批足够大
      dp = P_remain      // 全部数据并行
      pp = 1
   elif M > 10GB and P_remain >= 4:  // 大模型
      pp = min(4, P_remain)  // 流水线并行
      dp = P_remain / pp      // 剩余数据并行
   else:
      dp = sqrt(P_remain)   // 均衡分配
      pp = P_remain / dp
      
4. 验证内存约束：
   mem_per_device = M/mp + B/dp * activation_memory
   if mem_per_device > C:
      增加mp，返回步骤2
      
5. 返回 (dp, mp, pp)
```

示例：M=20GB, B=256, P=16, C=8GB
- mp = ceil(20/8) = 3 → 实际取4（2的幂）
- P_remain = 16/4 = 4
- B/P_remain = 256/4 = 64 > 16 → dp=4, pp=1
- 结果：(dp=4, mp=4, pp=1)
</details>

**练习12.8** 分析TSP编译器如何优化批量矩阵乘法（BMM）操作：$C[B][M][N] = A[B][M][K] \times B[B][K][N]$。考虑数据重用和并行策略。

<details>
<summary>答案</summary>

优化策略分析：

1. **批维度并行化**：
   - 将B个矩阵乘法分配到不同核心
   - 无数据依赖，完全并行
   - 通信量：0（理想情况）

2. **数据重用优化**：
   如果某些批次共享权重（如attention中的投影矩阵）：
   - 权重驻留：将共享的权重保持在SRAM
   - 重用率：$reuse = B \times \frac{computation}{data\_movement}$

3. **循环顺序优化**：
   ```
   最优顺序（最大化重用）：
   for b_o in range(0, B, Tb):
     for m_o in range(0, M, Tm):
       for n_o in range(0, N, Tn):
         for k_o in range(0, K, Tk):
           // 内层完全展开
           C[b_o:b_o+Tb][m_o:m_o+Tm][n_o:n_o+Tn] += 
             A[b_o:b_o+Tb][m_o:m_o+Tm][k_o:k_o+Tk] @ 
             B[b_o:b_o+Tb][k_o:k_o+Tk][n_o:n_o+Tn]
   ```

4. **分块大小选择**：
   约束：$Tb \times (Tm \times Tk + Tk \times Tn + Tm \times Tn) \times 2 \leq SRAM$
   
   目标：最大化 $Tb$ 以减少批次切换开销
   
   示例：SRAM=230MB, M=N=512, K=512
   - 单个矩阵乘法需要：$(512^2 + 512^2 + 512^2) \times 2 = 1.5MB$
   - $Tb_{max} = 230/1.5 \approx 153$
   - 实际选择 $Tb = 128$（2的幂）

5. **向量化策略**：
   - 最内层循环向量化宽度：320（TSP向量宽度）
   - 展开因子：$unroll = \min(Tn, 320)$

总体加速比（相对于逐个处理）：
$$Speedup = Tb \times \frac{utilization_{vectorized}}{utilization_{scalar}} \approx 128 \times 0.9 / 0.3 \approx 384x$$
</details>

## 常见陷阱与错误

1. **过度优化编译时间**
   - 陷阱：使用过于复杂的优化算法，导致编译时间过长
   - 解决：设置编译时间上限，使用渐进式优化策略

2. **忽视数值精度**
   - 陷阱：激进的融合和重排可能改变浮点运算顺序，影响数值稳定性
   - 解决：保守处理reduction操作，提供精度控制选项

3. **静态调度的局限性**
   - 陷阱：对动态形状或条件分支处理不当
   - 解决：预留动态调度路径，支持有限的运行时调整

4. **内存分配碎片**
   - 陷阱：频繁的小块分配导致SRAM碎片化
   - 解决：使用内存池，预分配大块连续内存

5. **并行策略选择不当**
   - 陷阱：盲目追求高并行度，忽视通信开销
   - 解决：基于实测建立准确的成本模型

6. **依赖分析错误**
   - 陷阱：错误识别数据依赖，导致错误的并行化
   - 解决：保守分析别名，使用形式化验证方法

7. **寄存器溢出**
   - 陷阱：过度展开导致寄存器压力过大
   - 解决：动态调整展开因子，监控寄存器使用

8. **忽略硬件特性**
   - 陷阱：未考虑bank冲突、对齐要求等硬件约束
   - 解决：在成本模型中准确建模硬件行为

## 最佳实践检查清单

### 编译器设计审查

- [ ] **调度算法**
  - □ 是否正确处理所有数据依赖？
  - □ 是否考虑了资源约束（计算、内存、带宽）？
  - □ 是否有死锁可能？
  - □ 关键路径是否已优化？

- [ ] **优化pass顺序**
  - □ Pass之间是否会相互干扰？
  - □ 是否存在优化机会丢失？
  - □ 迭代优化是否会收敛？

- [ ] **内存管理**
  - □ 是否最大化了数据重用？
  - □ Bank冲突是否已minimized？
  - □ 是否支持动态内存分配？
  - □ 内存泄漏检测机制是否完备？

### 性能优化审查

- [ ] **并行化策略**
  - □ 是否识别了所有并行机会？
  - □ 负载是否均衡？
  - □ 通信开销是否已最小化？
  - □ 是否支持动态负载均衡？

- [ ] **向量化效率**
  - □ 向量单元利用率是否 > 80%？
  - □ 是否存在不必要的标量操作？
  - □ 数据对齐是否正确？

- [ ] **循环优化**
  - □ 循环顺序是否最优？
  - □ 分块大小是否合理？
  - □ 是否充分利用了循环不变量？

### 正确性验证

- [ ] **数值准确性**
  - □ 浮点运算顺序改变是否可接受？
  - □ 是否有溢出/下溢风险？
  - □ 量化误差是否在容忍范围内？

- [ ] **功能正确性**
  - □ 是否有完整的单元测试？
  - □ 边界条件是否已覆盖？
  - □ 是否支持所有目标操作？

### 可维护性

- [ ] **代码质量**
  - □ IR设计是否清晰？
  - □ 优化pass是否模块化？
  - □ 是否有充分的文档和注释？

- [ ] **调试支持**
  - □ 是否能生成可读的中间代码？
  - □ 是否支持分步调试？
  - □ 性能分析工具是否完备？

- [ ] **扩展性**
  - □ 是否易于添加新的优化pass？
  - □ 是否支持新的硬件特性？
  - □ 是否有版本兼容性考虑？