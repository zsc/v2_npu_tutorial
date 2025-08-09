# 第7章：TPU编译器与映射

本章深入探讨TPU编译器的核心技术，重点分析XLA（Accelerated Linear Algebra）编译器如何将高层神经网络计算图映射到脉动阵列硬件上。我们将详细剖析编译优化策略、矩阵运算映射方法以及卷积算子的高效实现。通过理解编译器的工作原理，读者将掌握如何充分发挥脉动阵列架构的计算潜力，实现接近理论峰值的性能。

## 7.1 XLA编译流程

### 7.1.1 HLO图表示与优化

XLA编译器的核心是HLO（High-Level Optimizer）中间表示，它将TensorFlow、JAX等框架的计算图转换为统一的IR（Intermediate Representation）。HLO采用静态形状推断和强类型系统，便于进行激进的编译时优化。

HLO图的基本构成包括：
- **计算节点**：表示算子操作，如矩阵乘法、卷积、激活函数等
- **数据边**：表示张量数据流，携带形状和数据类型信息
- **控制边**：表示执行依赖关系，确保正确的计算顺序

```
    Input(X)        Weight(W)
         \            /
          \          /
           MatMul(Y=XW)
               |
           BiasAdd(Z=Y+b)
               |
            ReLU(A=max(0,Z))
               |
            Output
```

HLO优化passes包括：

1. **代数简化**：利用数学恒等式简化表达式
   - 常量折叠：$A \times 1 = A$，$A + 0 = A$
   - 强度削减：将除法转换为乘法 $A/B \rightarrow A \times (1/B)$
   - 交换律优化：重排运算顺序以提高局部性

2. **公共子表达式消除（CSE）**：识别并合并重复计算
   - 例如：多个分支使用相同的矩阵乘法结果
   - 通过哈希表快速识别等价计算

3. **死代码消除（DCE）**：移除未使用的计算节点
   - 基于数据流分析的活跃性分析
   - 递归删除无副作用的死节点

4. **循环优化**：
   - 循环不变量外提（Loop-invariant code motion）
   - 循环展开（Unrolling）以增加并行度
   - 循环融合（Fusion）减少内存访问

### 7.1.2 算子融合策略

算子融合是提升NPU性能的关键技术，通过将多个算子合并为一个融合算子，减少中间结果的内存读写开销。

**垂直融合（Producer-Consumer Fusion）**：
将生产者和消费者算子融合，避免中间张量的存储。

融合条件分析：
- 内存需求：融合后的工作集必须适配片上SRAM
- 计算密度：融合应提高算术强度（Arithmetic Intensity）
- 依赖关系：不能引入循环依赖

常见融合模式：
1. **GEMM + BiasAdd + Activation**
   $$Y = \text{ReLU}(XW + b)$$
   融合后只需一次内存写入

2. **BatchNorm + ReLU**
   $$Y = \text{ReLU}\left(\gamma \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\right)$$
   避免归一化中间结果的存储

3. **多层小算子融合**
   例如：Reshape → Transpose → Reshape可以合并为单个数据重排操作

**水平融合（Horizontal Fusion）**：
将并行的独立算子打包执行，提高硬件利用率。

适用场景：
- 多个小矩阵乘法：批处理提高脉动阵列利用率
- 并行的卷积分支：共享权重加载开销
- 独立的激活函数：向量化执行

### 7.1.3 内存规划与分配

TPU的片上SRAM容量有限（如TPUv4i有144MB HBM，但片上缓存仅32MB），高效的内存管理至关重要。

**静态内存分配**：
编译时确定所有张量的内存地址，避免运行时开销。

内存分配算法：
1. **生命周期分析**：构建张量的生存区间
   - 定义点：张量产生的位置
   - 使用点：所有读取该张量的位置
   - 死亡点：最后一次使用之后

2. **图着色算法**：
   - 构建冲突图：生命周期重叠的张量之间连边
   - 使用贪心着色算法分配内存块
   - 优化目标：最小化峰值内存使用

3. **内存池管理**：
   - 预分配大块连续内存
   - 使用buddy system或slab分配器
   - 支持快速分配和回收

**双缓冲（Double Buffering）**：
重叠计算与数据传输，隐藏内存访问延迟。

```
时刻t:   计算Buffer_A | 加载Buffer_B
时刻t+1: 计算Buffer_B | 加载Buffer_A
```

实现要求：
- 需要2倍的缓冲区空间
- DMA与计算单元并行工作
- 精确的同步机制

### 7.1.4 Tiling策略与参数选择

Tiling将大型张量运算分解为适合硬件资源的小块，是实现高效映射的核心技术。

**Tiling参数空间**：
对于矩阵乘法 $C_{M \times N} = A_{M \times K} \times B_{K \times N}$，tiling参数包括：
- $T_M$：M维度的tile大小
- $T_N$：N维度的tile大小
- $T_K$：K维度的tile大小（累加维度）

约束条件：
1. **硬件约束**：
   $$T_M \times T_N \leq \text{脉动阵列大小}$$
   $$T_M \times T_K + T_K \times T_N + T_M \times T_N \leq \text{片上SRAM容量}$$

2. **对齐约束**：
   - Tile大小应为硬件向量宽度的倍数
   - 考虑内存bank的对齐要求

**自动调优（Auto-tuning）**：
使用机器学习方法搜索最优tiling参数。

搜索策略：
1. **网格搜索**：枚举所有可能的参数组合
2. **随机搜索**：随机采样参数空间
3. **贝叶斯优化**：基于历史性能建模
4. **强化学习**：将tiling决策建模为序列决策问题

性能模型：
$$\text{Cycles} = \frac{M \times N \times K}{T_M \times T_N \times T_K} \times \left( T_K \times \text{Compute\_cycles} + \text{Load\_cycles} \right)$$

其中Load_cycles考虑数据重用：
- Input重用因子：$\frac{N}{T_N}$
- Weight重用因子：$\frac{M}{T_M}$
- Output重用因子：$\frac{K}{T_K}$

## 7.2 矩阵乘法映射

### 7.2.1 大矩阵分块策略

将大规模矩阵乘法高效映射到有限大小的脉动阵列是编译器的核心任务。考虑矩阵乘法 $C = A \times B$，其中 $A \in \mathbb{R}^{M \times K}$，$B \in \mathbb{R}^{K \times N}$。

**分块算法设计**：

基本分块策略将计算分解为三层嵌套循环：
```
for m_tile in range(0, M, T_M):
    for n_tile in range(0, N, T_N):
        for k_tile in range(0, K, T_K):
            # 计算C[m_tile:m_tile+T_M, n_tile:n_tile+T_N]的部分和
            C_tile += A_tile @ B_tile
```

**数据布局优化**：

1. **行主序vs列主序**：
   - A矩阵采用行主序存储，便于按行streaming
   - B矩阵采用列主序存储，便于按列广播
   - 减少数据重排开销

2. **Z字形布局（Z-order）**：
   提高空间局部性，适合2D脉动阵列
   ```
   原始布局:  0 1 2 3     Z字形:  0 1 4 5
              4 5 6 7             2 3 6 7
              8 9 A B             8 9 C D
              C D E F             A B E F
   ```

3. **块内连续存储**：
   将tile内的数据连续存放，提高DMA效率
   - 减少地址生成开销
   - 提高突发传输（burst）效率

**边界处理与padding**：

当矩阵维度不能被tile大小整除时，需要特殊处理：

1. **零填充（Zero Padding）**：
   $$M' = \lceil \frac{M}{T_M} \rceil \times T_M$$
   $$N' = \lceil \frac{N}{T_N} \rceil \times T_N$$
   
   优点：简化控制逻辑
   缺点：浪费计算资源

2. **动态tile大小**：
   最后一个tile使用较小的维度
   - 需要硬件支持可变大小配置
   - 控制逻辑更复杂

3. **预计算掩码（Predication）**：
   使用掩码禁用超出边界的计算
   - 保持规则的tile大小
   - 通过使能信号控制PE

### 7.2.2 性能优化技巧

**工作负载平衡**：
确保所有PE均匀分配计算任务，避免负载不均。

1. **静态调度**：
   编译时确定每个PE的计算任务
   - 优点：无运行时开销
   - 缺点：灵活性差

2. **工作窃取（Work Stealing）**：
   空闲PE从忙碌PE窃取任务
   - 需要硬件支持任务队列
   - 增加控制复杂度

**数据预取优化**：

预取策略设计：
1. **软件流水线**：
   ```
   加载A[i+1] | 加载B[i+1] | 计算C[i] = A[i] × B[i]
   ```
   重叠下一轮的数据加载与当前计算

2. **预取距离计算**：
   $$D_{prefetch} = \lceil \frac{L_{memory}}{T_{compute}} \rceil$$
   其中$L_{memory}$是内存延迟，$T_{compute}$是计算时间

3. **自适应预取**：
   根据运行时访问模式动态调整预取策略

### 7.2.3 批处理维度处理

现代深度学习中，批处理维度（batch dimension）的高效处理至关重要。

**批处理GEMM映射**：
对于批处理矩阵乘法 $C[b] = A[b] \times B[b]$，$b \in [0, B)$：

1. **串行处理**：
   依次处理每个批次
   - 简单但效率低
   - 适合batch size较小的场景

2. **批次并行**：
   将不同批次映射到不同的PE组
   $$\text{PE\_group}[i] \leftarrow \text{Batch}[i \bmod G]$$
   其中G是PE组数量

3. **批次合并**：
   将批次维度展开到M或N维度
   $$A_{BM \times K} = \text{reshape}(A_{B \times M \times K})$$
   提高大矩阵的硬件利用率

**动态批处理优化**：

适应可变batch size的策略：
1. **Padding到2的幂次**：
   $$B' = 2^{\lceil \log_2 B \rceil}$$
   简化地址计算但可能浪费计算

2. **分组处理**：
   将batch分为大小相近的组
   - 组内使用相同的tiling参数
   - 减少重配置开销

3. **在线合并（Online Batching）**：
   动态合并小batch以提高利用率
   - 需要考虑延迟约束
   - 适合推理服务场景

## 7.3 卷积映射优化

### 7.3.1 Im2col变换

Im2col（Image to Column）是将卷积运算转换为矩阵乘法的经典方法。

**变换原理**：
对于卷积运算 $Y = X * W$，其中：
- 输入：$X \in \mathbb{R}^{B \times H \times W \times C_{in}}$
- 卷积核：$W \in \mathbb{R}^{K_h \times K_w \times C_{in} \times C_{out}}$
- 输出：$Y \in \mathbb{R}^{B \times H_{out} \times W_{out} \times C_{out}}$

Im2col变换步骤：
1. 将输入展开为矩阵：$X_{col} \in \mathbb{R}^{(B \cdot H_{out} \cdot W_{out}) \times (K_h \cdot K_w \cdot C_{in})}$
2. 将权重reshape：$W_{col} \in \mathbb{R}^{(K_h \cdot K_w \cdot C_{in}) \times C_{out}}$
3. 执行矩阵乘法：$Y_{col} = X_{col} \times W_{col}$
4. 将结果reshape回原始形状

**内存开销分析**：
Im2col的主要缺点是内存膨胀：
$$\text{膨胀率} = K_h \times K_w$$

对于3×3卷积，数据量增加9倍。优化策略：
1. **分块Im2col**：只展开当前处理的tile
2. **隐式Im2col**：通过地址生成实现虚拟展开
3. **重叠优化**：复用相邻窗口的重叠数据

### 7.3.2 Direct Convolution

直接卷积避免了Im2col的内存开销，直接在脉动阵列上实现卷积运算。

**空间映射策略**：
1. **输出静止（Output Stationary）**：
   - 每个PE负责一个输出像素
   - 输入和权重流经PE阵列
   - 适合大卷积核

2. **权重静止（Weight Stationary）**：
   - 每个PE存储部分权重
   - 输入数据广播到所有PE
   - 适合深度可分离卷积

3. **输入静止（Input Stationary）**：
   - 每个PE缓存部分输入
   - 权重在PE间传递
   - 适合1×1卷积

**数据流优化**：
```
     输入特征图
         ↓
    [Shift Register]  ← 实现滑动窗口
         ↓
    [PE Array] × 权重
         ↓
     输出特征图
```

使用移位寄存器链实现滑动窗口：
- 减少重复的内存访问
- 自然支持stride和dilation

### 7.3.3 Winograd变换

Winograd算法通过数论变换减少乘法次数，特别适合小卷积核。

**F(2,3)变换示例**：
将3×3卷积的2×2输出tile计算从16次乘法减少到4次。

变换矩阵：
$$G = \begin{bmatrix} 1 & 0 & 0 \\ 0.5 & 0.5 & 0.5 \\ 0.5 & -0.5 & 0.5 \\ 0 & 0 & 1 \end{bmatrix}$$

$$B^T = \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 1 & 0 & -1 \end{bmatrix}$$

$$A^T = \begin{bmatrix} 1 & 1 & 1 & 0 \\ 0 & 1 & -1 & -1 \end{bmatrix}$$

计算步骤：
1. 权重变换：$\tilde{W} = G \cdot W \cdot G^T$（离线计算）
2. 输入变换：$\tilde{X} = B^T \cdot X \cdot B$
3. 逐元素乘法：$\tilde{Y} = \tilde{X} \odot \tilde{W}$
4. 输出变换：$Y = A^T \cdot \tilde{Y} \cdot A$

**适用性分析**：
Winograd优势条件：
- 小卷积核（3×3, 5×5）
- 低精度量化（INT8, FP16）
- 计算密集型层

不适用场景：
- 大卷积核（变换开销超过节省）
- 深度可分离卷积（计算已经很少）
- 需要高数值精度（变换引入误差）

### 7.3.4 特殊卷积优化

**深度可分离卷积（Depthwise Separable）**：
分解为depthwise和pointwise两步：
1. Depthwise：$Y_{dw} = X *_{dw} W_{dw}$，每个通道独立卷积
2. Pointwise：$Y = Y_{dw} *_{1×1} W_{pw}$，1×1卷积混合通道

映射策略：
- Depthwise：将通道映射到不同PE
- Pointwise：退化为标准矩阵乘法
- 计算量减少率：$\frac{1}{C_{out}} + \frac{1}{K^2}$

**空洞卷积（Dilated Convolution）**：
通过调整数据访问模式支持dilation：
$$\text{Index}(i,j) = i \times \text{dilation}_h + j \times \text{dilation}_w$$

硬件支持：
- 可配置的地址生成器
- 支持非连续内存访问
- 预取缓冲区管理

**分组卷积（Grouped Convolution）**：
将输入输出通道分组，组间独立计算：
$$Y_g = X_g * W_g, \quad g \in [0, G)$$

优化要点：
- 组间并行执行
- 组内数据局部性优化
- 减少跨组数据移动

## 本章小结

本章详细探讨了TPU编译器的核心技术，从XLA编译流程到具体算子的硬件映射策略。关键要点包括：

1. **编译优化层次**：
   - HLO图级优化：算子融合、内存规划、死代码消除
   - Tiling参数选择：平衡计算效率与内存约束
   - 数据布局优化：提高内存访问效率

2. **矩阵乘法映射核心公式**：
   - 脉动阵列利用率：$U = \frac{\min(T_M, SA_M) \times \min(T_N, SA_N)}{SA_M \times SA_N}$
   - 数据重用度：$R = \frac{2MNK}{MK + KN + MN}$
   - 带宽需求：$BW = \frac{\text{Data\_Movement}}{\text{Compute\_Time}}$

3. **卷积优化策略选择**：
   - Im2col：通用但内存开销大，适合标准卷积
   - Direct：内存高效，需要专门硬件支持
   - Winograd：减少计算量，适合小卷积核
   - 特殊优化：针对depthwise、dilated等变体

4. **性能优化原则**：
   - 最大化数据重用，最小化内存传输
   - 平衡计算与访存，避免成为瓶颈
   - 充分利用硬件特性，如双缓冲、流水线

## 练习题

### 基础题

**练习7.1**：Tiling参数计算
给定脉动阵列大小128×128，片上SRAM容量8MB，要计算矩阵乘法C[1024×1024] = A[1024×768] × B[768×1024]，数据类型为FP16。计算最优的tiling参数(T_M, T_N, T_K)。

*提示*：考虑内存约束：$2(T_M \times T_K + T_K \times T_N + T_M \times T_N) \leq 8MB$

<details>
<summary>答案</summary>

设每个FP16元素占2字节，内存约束为：
$$2(T_M \times T_K + T_K \times T_N + T_M \times T_N) \times 2 \leq 8 \times 2^{20}$$

硬件约束：$T_M, T_N \leq 128$

为最大化重用，选择$T_M = T_N = 128$，计算$T_K$：
$$2(128 \times T_K + T_K \times 128 + 128 \times 128) \times 2 \leq 8 \times 2^{20}$$
$$512T_K + 32768 \leq 4194304$$
$$T_K \leq 8128$$

考虑K维度总大小768，选择$T_K = 768$可以避免K维度的分块。

验证：$2(128 \times 768 + 768 \times 128 + 128 \times 128) \times 2 = 425,984$字节 < 8MB ✓

最优参数：$(T_M, T_N, T_K) = (128, 128, 768)$
</details>

**练习7.2**：算子融合收益分析
考虑融合BatchNorm-ReLU序列，输入张量大小为[64, 224, 224, 128]，数据类型FP16。计算融合前后的内存访问量。

*提示*：BatchNorm需要读写一次中间结果，融合后可以省去这次读写。

<details>
<summary>答案</summary>

张量大小：$64 \times 224 \times 224 \times 128 = 411,041,792$个元素
每个FP16占2字节，总大小：$411,041,792 \times 2 = 822,083,584$字节 ≈ 784MB

未融合：
- BatchNorm：读输入784MB + 写输出784MB = 1568MB
- ReLU：读输入784MB + 写输出784MB = 1568MB
- 总计：3136MB

融合后：
- 读输入784MB + 写输出784MB = 1568MB
- 节省：1568MB（50%）

融合收益：减少50%的内存访问量
</details>

**练习7.3**：Im2col内存膨胀计算
对于输入张量[32, 224, 224, 64]，使用3×3卷积，stride=1，padding=1，输出通道128。计算Im2col变换后的内存需求。

*提示*：Im2col后矩阵大小为$(B \times H_{out} \times W_{out}) \times (K_h \times K_w \times C_{in})$

<details>
<summary>答案</summary>

输出尺寸（padding=1, stride=1）：$H_{out} = W_{out} = 224$

Im2col矩阵维度：
- 行数：$32 \times 224 \times 224 = 1,605,632$
- 列数：$3 \times 3 \times 64 = 576$

矩阵元素总数：$1,605,632 \times 576 = 924,844,032$

FP16内存需求：$924,844,032 \times 2 = 1,849,688,064$字节 ≈ 1.72GB

原始输入大小：$32 \times 224 \times 224 \times 64 \times 2 = 205,520,896$字节 ≈ 196MB

内存膨胀率：$\frac{1.72GB}{196MB} \approx 8.78$倍
</details>

### 挑战题

**练习7.4**：Winograd数值稳定性分析
Winograd F(2,3)变换中，变换矩阵包含0.5和-0.5。如果输入数据范围是[-1, 1]，分析经过变换后的数值范围，并讨论对INT8量化的影响。

*提示*：分析$B^T \cdot X \cdot B$的最大值情况

<details>
<summary>答案</summary>

考虑最坏情况，输入tile $X$所有元素为1或-1。

对于$B^T = \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 1 & 1 & 0 \\ 0 & -1 & 1 & 0 \\ 0 & 1 & 0 & -1 \end{bmatrix}$

第一次变换$B^T \cdot X$：
- 第1行：最大值为$|1-(-1)| = 2$
- 第2行：最大值为$|1+1| = 2$
- 第3行：最大值为$|-1+1| = 0$或$|1-1| = 2$
- 第4行：最大值为$|1-(-1)| = 2$

第二次变换$(B^T \cdot X) \cdot B$：
最大可能值约为4（两次变换的累积）

对INT8量化的影响：
1. 动态范围扩大4倍，需要2bit额外精度
2. INT8范围[-128, 127]，变换后有效范围减少到[-32, 31]
3. 量化误差被变换放大，影响最终精度
4. 建议：Winograd更适合FP16或INT16，INT8需要谨慎使用
</details>

**练习7.5**：多核负载均衡优化
有4个TPU核心，需要处理批大小为100的推理任务。设计一个负载均衡策略，使得：
1. 各核心负载尽可能均匀
2. 最小化同步开销
3. 支持动态批大小

*提示*：考虑静态分配vs动态调度的权衡

<details>
<summary>答案</summary>

策略设计：

1. **基础静态分配**：
   - 核心0-2：各处理25个样本
   - 核心3：处理25个样本
   - 问题：批大小不是4的倍数时不均衡

2. **改进的块循环分配**：
   ```
   块大小 = ceil(100/4) = 25
   分配方案：[25, 25, 25, 25]
   ```
   当批大小变化时，动态调整块大小

3. **两级调度策略**（推荐）：
   - 第一级：粗粒度分配
     * 将100个样本分为10个块，每块10个样本
     * 初始分配：核心0-3各分配2-3个块
   - 第二级：工作窃取
     * 完成快的核心从慢核心窃取剩余块
     * 使用原子操作管理块队列

4. **实现细节**：
   ```
   全局块队列：[Block_0, Block_1, ..., Block_9]
   每个核心：
     while (block = atomic_fetch_block()):
       process(block)
   ```

5. **动态批支持**：
   - 维护块大小查找表：{100: 10, 64: 8, 32: 8, ...}
   - 根据批大小选择合适的块大小
   - 平衡并行度与同步开销

优势：
- 自动负载均衡
- 最小同步（仅在获取块时）
- 灵活适应不同批大小
</details>

**练习7.6**：编译器优化决策树
设计一个决策树，根据网络层的特征（输入大小、卷积核大小、通道数等）自动选择最优的卷积实现策略（Im2col、Direct、Winograd）。

*提示*：考虑计算密度、内存带宽、数值精度等因素

<details>
<summary>答案</summary>

决策树设计：

```
输入层参数：H×W×C_in, 卷积核K×K, 输出通道C_out

1. 卷积核大小检查
   ├─ K > 5：使用Im2col（Winograd变换开销过大）
   └─ K ≤ 5：继续判断
   
2. 特殊卷积检查
   ├─ K = 1：Direct实现（1×1卷积）
   ├─ Depthwise：Direct实现（通道独立）
   └─ 标准卷积：继续判断

3. 精度要求检查
   ├─ FP32/FP64：避免Winograd（数值误差）
   └─ INT8/FP16：继续判断

4. 内存约束检查
   ├─ Im2col内存需求 > 可用SRAM：
   │  └─ 使用Direct或分块Im2col
   └─ Im2col内存需求 ≤ 可用SRAM：继续判断

5. 计算密度分析
   ├─ 计算密度 = (2×K²×C_in×C_out×H×W) / 内存访问量
   ├─ 高密度（>10）：Winograd（K=3）或Im2col
   └─ 低密度（≤10）：Direct（减少内存压力）

6. 批大小考虑
   ├─ Batch ≥ 8：Im2col（批处理效率高）
   └─ Batch < 8：Direct或Winograd

决策函数伪代码：
```python
def select_conv_strategy(H, W, C_in, C_out, K, batch, precision):
    if K == 1:
        return "Direct_1x1"
    if is_depthwise:
        return "Direct_Depthwise"
    if K > 5:
        return "Im2col"
    if precision in ["FP32", "FP64"]:
        return "Im2col" if memory_sufficient else "Direct"
    if K == 3 and precision in ["INT8", "FP16"]:
        compute_density = calculate_density(...)
        if compute_density > 10 and batch >= 4:
            return "Winograd_F(2,3)"
    if im2col_memory_fit():
        return "Im2col"
    return "Direct"
```

性能预测模型：
- Im2col：稳定高性能，内存开销大
- Winograd：计算量减少~2.25×，但有变换开销
- Direct：内存高效，但可能计算效率较低
</details>

**练习7.7**：流水线优化分析
分析一个3级流水线：加载权重→计算→存储结果。如果三个阶段分别需要100、200、50个周期，如何优化流水线以提高吞吐量？

*提示*：找出瓶颈阶段，考虑并行化或分割策略

<details>
<summary>答案</summary>

流水线分析：

1. **瓶颈识别**：
   - 加载：100周期
   - 计算：200周期（瓶颈）
   - 存储：50周期
   - 流水线周期 = max(100, 200, 50) = 200周期

2. **优化策略**：

   **方案1：计算阶段并行化**
   - 将计算分为2个并行单元，每个100周期
   - 新流水线：100 | 100 | 50
   - 瓶颈变为加载阶段（100周期）
   - 吞吐量提升：200/100 = 2×

   **方案2：细粒度流水线**
   - 将计算分为2个子阶段：计算A(120) + 计算B(80)
   - 5级流水线：加载(100) | 计算A(120) | 计算B(80) | 存储(50)
   - 瓶颈：计算A(120周期)
   - 吞吐量提升：200/120 = 1.67×

   **方案3：双缓冲优化**
   ```
   时刻0-200：加载W0(100) | 计算null    | 存储null
   时刻200-400：加载W1(100) | 计算W0(200) | 存储null
   时刻400-600：加载W2(100) | 计算W1(200) | 存储R0(50)
   ```
   稳态吞吐量：每200周期处理一个任务

3. **综合优化方案**：
   - 2个计算单元（各100周期）
   - 2个加载单元交替工作（隐藏延迟）
   - 流水线：[加载A|加载B] | [计算A|计算B] | 存储
   - 实现50周期/任务的吞吐量（4×提升）

4. **资源代价分析**：
   - 方案1：2×计算资源
   - 方案2：额外寄存器和控制逻辑
   - 方案3：2×加载单元 + 2×计算单元
   - 选择依据：面积预算vs性能需求
</details>

## 常见陷阱与错误

### 1. Tiling参数选择错误
- **问题**：选择的tile大小不是硬件向量宽度的倍数
- **后果**：硬件利用率低，产生大量padding开销
- **解决**：确保tile维度对齐到硬件参数（如128的倍数）

### 2. 忽视内存带宽限制
- **问题**：只优化计算，忽略数据传输瓶颈
- **后果**：计算单元空闲等待数据
- **解决**：使用roofline模型分析，平衡计算与访存

### 3. 过度的算子融合
- **问题**：融合过多算子导致寄存器溢出
- **后果**：频繁的spill/reload，性能下降
- **解决**：基于硬件资源约束制定融合策略

### 4. Winograd误用
- **问题**：对所有卷积都使用Winograd
- **后果**：大卷积核变换开销超过收益，数值精度问题
- **解决**：仅对3×3、5×5卷积，且精度要求不高时使用

### 5. 静态调度过于僵化
- **问题**：编译时固定所有参数，无法适应运行时变化
- **后果**：动态batch size时效率低
- **解决**：支持多版本kernel，运行时选择

### 6. 边界处理遗漏
- **问题**：未正确处理非整除的矩阵维度
- **后果**：计算结果错误或访存越界
- **解决**：实现完善的padding和masking机制

## 最佳实践检查清单

### 编译优化检查
- [ ] HLO图是否进行了充分的算子融合？
- [ ] 内存分配是否最小化了峰值使用？
- [ ] 是否识别并优化了所有可并行的计算？
- [ ] 常量折叠和死代码消除是否完成？

### Tiling策略检查
- [ ] Tile大小是否对齐硬件约束？
- [ ] 是否最大化了数据重用？
- [ ] 内存需求是否在SRAM容量内？
- [ ] 是否考虑了不同层的特征选择不同参数？

### 映射效率检查
- [ ] 脉动阵列利用率是否超过80%？
- [ ] 是否实现了计算与数据传输的重叠？
- [ ] 关键路径是否已识别并优化？
- [ ] 是否有不必要的同步点？

### 数值精度检查
- [ ] 量化方案是否保证了精度要求？
- [ ] Winograd变换是否引入了不可接受的误差？
- [ ] 是否实现了必要的饱和处理？
- [ ] 累加器位宽是否足够避免溢出？

### 性能验证检查
- [ ] 是否达到了理论峰值的70%以上？
- [ ] 内存带宽利用率是否合理？
- [ ] 是否存在明显的性能瓶颈？
- [ ] 不同batch size下性能是否稳定？

### 可扩展性检查
- [ ] 是否支持动态shape？
- [ ] 多核扩展时负载是否均衡？
- [ ] 是否预留了新算子的支持接口？
- [ ] 编译时间是否在可接受范围内？