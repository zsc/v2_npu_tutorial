# 第4章：存储系统与数据流

在NPU设计中，存储系统往往是决定整体性能的关键瓶颈。本章深入探讨NPU存储层次设计、数据重用策略、DMA机制以及片上网络基础，帮助读者理解如何通过精心的存储系统设计来最大化NPU的计算效率。我们将以200 TOPS的设计目标为例，定量分析各级存储的带宽需求，并探讨不同数据流模式下的优化策略。

## 4.1 存储层次设计

### 4.1.1 存储器技术对比

NPU的存储层次通常包含三个主要层级，每层在容量、带宽、延迟和功耗方面都有不同的权衡：

**片上SRAM特性**

片上SRAM提供最高的带宽和最低的访问延迟，但容量受限且成本高昂。典型参数：
- 访问延迟：1-2 cycles
- 带宽密度：~10 TB/s/mm²（7nm工艺）
- 功耗：~1 pJ/bit access
- 容量：通常10-100 MB（占芯片面积的30-50%）

SRAM的物理实现通常采用6T或8T单元结构。6T SRAM单元包含6个晶体管，面积约为0.05 μm²（7nm工艺），而8T SRAM提供独立的读写端口，避免读干扰但面积增加约30%。在NPU设计中，通常采用混合策略：权重缓存使用高密度6T SRAM，而需要频繁读写的激活缓存使用8T SRAM。

对于200 TOPS的NPU，假设MAC利用率为80%，nvfp4精度下：
$$\text{所需带宽} = 200 \times 10^{12} \times 0.8 \times 3 \times 4 \text{ bits/s} = 1.92 \text{ Pb/s} = 240 \text{ TB/s}$$

其中因子3来自两个输入操作数和一个输出，因子4是nvfp4的位宽。

这个带宽需求远超单片SRAM所能提供，因此需要采用分布式SRAM架构。假设采用1024个SRAM bank，每个bank需要提供：
$$\text{Bank带宽} = \frac{240 \text{ TB/s}}{1024} = 234 \text{ GB/s}$$

在1 GHz时钟频率下，每个bank需要234位宽的接口，这在物理实现上是可行的。

**HBM技术演进**

HBM (High Bandwidth Memory) 提供了容量和带宽的良好平衡：

```
HBM代际对比：
         HBM2E    HBM3     HBM3E
带宽      460      819      1230    GB/s/stack
容量      16       24       36      GB/stack  
功耗效率   7        5.5      4.5     pJ/bit
I/O宽度   1024     1024     1024    bits
频率      3.6      6.4      9.6     Gbps/pin
电压      1.2      1.1      1.1     V
```

HBM采用2.5D封装技术，通过硅中介层（interposer）实现与主芯片的连接。每个HBM stack包含8-12层DRAM die，通过TSV（Through Silicon Via）垂直互连。对于200 TOPS的NPU设计，典型配置为4个HBM3 stack，提供总计3.2 TB/s的带宽和96 GB的容量。

HBM的关键优势在于其极短的物理距离（<10mm）和宽I/O接口，相比GDDR6降低了约3倍的pJ/bit访问能耗。然而，HBM的成本约为DDR的5-8倍，且需要先进封装技术支持。

**DDR与GDDR权衡**

DDR提供大容量但带宽有限，GDDR则在两者间取得平衡：
- DDR5：~50 GB/s/channel，容量可达128GB/DIMM
- GDDR6X：~80 GB/s/chip，容量2-4GB/chip
- 功耗：DDR ~20 pJ/bit，GDDR ~15 pJ/bit

在自动驾驶场景中，需要存储大量的高精地图、历史轨迹和中间特征图。一个典型的BEV感知模型可能需要：
- 输入缓存：6个相机 × 1920×1080×3 × 4 bytes = 150 MB
- 特征图缓存：多尺度特征 ~500 MB
- 历史帧缓存：10帧 × 200 MB = 2 GB
- 模型权重：~1 GB（INT8量化后）

因此，边缘端NPU通常采用混合存储架构：
1. 片上SRAM（50 MB）：存储当前计算tile和部分权重
2. HBM/GDDR（8-16 GB）：存储完整模型和中间结果
3. DDR（32-64 GB）：存储历史数据和预取缓冲

### 4.1.2 存储带宽需求计算

**算子带宽需求分析**

以矩阵乘法 $C = A \times B$ 为例，计算不同数据流模式的带宽需求：

设矩阵维度为 $M \times K$ 和 $K \times N$，计算量为 $2MNK$ FLOPs。

无重用情况下的理论带宽需求：
$$B_{\text{no-reuse}} = \frac{(MK + KN + MN) \times \text{bits}}{2MNK / \text{FLOPS}} = \frac{(M+N+K) \times \text{bits}}{2K} \times \text{FLOPS}$$

引入重用后，实际带宽需求取决于片上缓存大小 $S$：

- Output Stationary (OS)：缓存输出矩阵块
  $$B_{OS} = \frac{2 \times \text{bits} \times \text{FLOPS}}{K}$$
  
- Weight Stationary (WS)：缓存权重矩阵块
  $$B_{WS} = \frac{2 \times \text{bits} \times \text{FLOPS}}{M}$$

**实际案例：Transformer中的Attention计算**

考虑自动驾驶中的多相机BEV Transformer，序列长度 $L = 10000$（100×100的BEV网格），嵌入维度 $d = 256$：

1. Q、K、V投影：$3 \times L \times d^2$ 计算量
   - 带宽需求：$\frac{3Ld(L+d) \times 4 \text{ bits}}{3Ld^2 \times 2} = \frac{2(L+d)}{d} = 78.5$ bits/FLOP

2. Attention分数计算：$Q \times K^T$，计算量 $2L^2d$
   - 带宽需求：$\frac{2Ld \times 4 + L^2 \times 4}{2L^2d} = \frac{4}{d} + \frac{2}{L} ≈ 0.016$ bits/FLOP

3. 加权求和：$\text{Softmax}(\text{scores}) \times V$，计算量 $2L^2d$
   - 带宽需求：类似于步骤2

可见，Attention的不同阶段对带宽要求差异巨大，需要动态调整数据流策略。

**带宽效率指标**

定义带宽效率 $\eta_B$ 为实际计算吞吐量与理论峰值的比值：
$$\eta_B = \frac{\text{实际FLOPS}}{\min(\text{计算峰值}, \text{带宽} \times \text{算术强度})}$$

其中算术强度（Arithmetic Intensity）定义为：
$$AI = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

**Roofline模型应用**

对于200 TOPS NPU，假设HBM带宽3.2 TB/s，构建Roofline模型：

1. 计算屋顶线：200 TOPS（nvfp4）
2. 内存屋顶线：$3.2 \text{ TB/s} \times AI$
3. 平衡点：$AI_{balance} = \frac{200 \times 10^{12}}{3.2 \times 10^{12}} = 62.5$ FLOPS/byte

常见算子的算术强度：
- 全连接层（1024×1024）：$AI = \frac{2 \times 1024^3}{3 \times 1024^2 \times 4} ≈ 171$ FLOPS/byte
- 1×1卷积：$AI ≈ \frac{2 \times C_{out}}{4}$ （假设输入已缓存）
- 3×3卷积：$AI ≈ \frac{2 \times 9 \times C_{out}}{4 \times 9 + 4}$ 
- Depthwise 3×3：$AI ≈ \frac{18}{40} = 0.45$ FLOPS/byte

可见，depthwise卷积严重受限于内存带宽，而大矩阵乘法则受限于计算能力。

### 4.1.3 Bank冲突与访问模式优化

**Bank组织结构**

典型的多Bank SRAM组织：

```
      ┌─────────────────────────┐
      │    Crossbar Switch      │
      └─────┬───┬───┬───┬───────┘
            │   │   │   │
         ┌──▼─┐ │   │ ┌──▼─┐
         │Bank│ │   │ │Bank│
         │ 0  │ │   │ │ N-1│
         └────┘ │   │ └────┘
              ┌─▼─┐ └─▼─┐
              │Bank│ ... │
              │ 1  │     │
              └────┘     │
```

Bank数量选择需要平衡面积开销和访问灵活性。常见配置：
- Bank数量：16-64
- 每Bank容量：64-256 KB
- 端口数：1R1W或2R1W

**访问模式分析**

以2D卷积为例，不同的数据布局导致不同的访问模式：

1. NCHW布局：连续访问同一通道的空间维度
   - 优点：空间局部性好
   - 缺点：通道间切换开销大

2. NHWC布局：连续访问同一位置的所有通道
   - 优点：适合depthwise操作
   - 缺点：大通道数时缓存利用率低

3. NCHWc布局（通道分块）：将C维度分块为C/c × c
   - 结合两种布局优点
   - c通常选择为SIMD宽度（如16或32）

**实际访问模式优化案例**

以YOLOv8的C2f模块为例（输入256×80×80，输出256×80×80）：

采用NCHW布局时的Bank访问模式：
```
时刻t:   Bank[0]: C0,H0,W0-W15
         Bank[1]: C0,H0,W16-W31
         ...
         Bank[4]: C0,H1,W0-W15
```

问题：当卷积kernel跨行时，会访问Bank[0]和Bank[4]，可能造成冲突。

优化方案：
1. Padding对齐：将宽度从80 padding到84（Bank数的倍数）
2. 交织存储：$\text{Bank}_{id} = (C \times 7 + H \times 13 + W) \mod N_{banks}$
3. 双缓冲：一个buffer用于当前行，另一个预取下一行

这样可将Bank冲突率从15%降低到2%以下。

**冲突消除策略**

1. 地址交织（Interleaving）：
   $$\text{Bank}_{\text{id}} = (\text{addr} >> \text{offset}) \bmod N_{\text{banks}}$$

2. 素数Bank数量：选择素数个Bank（如17、31）减少规律性冲突

3. 双缓冲（Double Buffering）：计算与数据传输重叠

**Bank仲裁器设计**

多端口Bank访问需要仲裁机制，常见策略：

1. 固定优先级：计算单元 > DMA > 调试接口
2. 轮询（Round-Robin）：公平但可能降低关键路径性能
3. 加权轮询：根据请求源的带宽需求分配权重
4. 信用机制（Credit-based）：每个请求源有信用额度

仲裁延迟模型：
$$T_{access} = T_{arbitration} + T_{bank\_access} + T_{routing}$$

典型值（1GHz时钟）：
- $T_{arbitration}$：1 cycle（简单仲裁）或2-3 cycles（复杂仲裁）
- $T_{bank\_access}$：1-2 cycles
- $T_{routing}$：1 cycle（近距离）或2-3 cycles（跨片）

**存储一致性保证**

在多个计算单元共享Bank时，需要保证数据一致性：

1. 写后读（RAW）hazard：通过scoreboard跟踪pending写操作
2. 写后写（WAW）hazard：保证写操作的顺序
3. 原子操作支持：实现atomic_add等操作用于累加

硬件实现通常采用：
- 版本号机制：每个地址关联版本号
- 锁机制：细粒度锁保护关键区域
- 事务内存：支持回滚的投机执行

## 4.2 数据重用模式

数据重用是提高NPU能效的核心策略。通过最大化片上数据的重用次数，可以显著降低对外部存储器的访问需求，从而减少功耗并提高性能。

### 4.2.1 重用分类：Temporal vs Spatial

**时间重用（Temporal Reuse）**

时间重用指同一数据在不同时间被同一计算单元多次使用：

$$\text{时间重用度} = \frac{\text{数据使用次数}}{\text{数据加载次数}}$$

以矩阵乘法为例，当计算 $C_{ij} = \sum_{k} A_{ik} \times B_{kj}$ 时：
- $A_{ik}$ 在计算 $C_{ij}$ 的所有 $j$ 值时被重用
- $B_{kj}$ 在计算 $C_{ij}$ 的所有 $i$ 值时被重用

**空间重用（Spatial Reuse）**

空间重用指同一数据同时被多个计算单元使用：

```
     广播数据
        │
    ┌───▼───┐
    │       │
  ┌─▼─┐   ┌─▼─┐
  │PE0│   │PE1│  ... │PEn│
  └───┘   └───┘      └───┘
```

空间重用的效率取决于：
- PE阵列的组织方式
- 数据广播网络的设计
- 计算的并行维度选择

**重用优化的数学模型**

给定片上存储容量 $S$，优化目标是最小化数据传输量 $D$：

$$\min D = \sum_{i} \frac{s_i \times u_i}{r_i}$$

其中：
- $s_i$：数据 $i$ 的大小
- $u_i$：数据 $i$ 的使用次数
- $r_i$：数据 $i$ 的重用次数

约束条件：$\sum_{i \in \text{on-chip}} s_i \leq S$

**具体示例：MobileNet中的Depthwise Separable卷积**

考虑MobileNetV3中的一个典型块，输入112×112×24，输出112×112×96：

1. Depthwise 3×3：
   - 计算量：$112^2 \times 24 \times 9 \times 2 = 5.4M$ FLOPs
   - 数据量：
     - 输入：$112^2 \times 24 \times 4 = 1.2$ MB
     - 权重：$3^2 \times 24 \times 4 = 0.9$ KB
     - 输出：$112^2 \times 24 \times 4 = 1.2$ MB
   - 算术强度：$\frac{5.4M}{2.4M} = 2.25$ FLOPs/byte

2. Pointwise 1×1：
   - 计算量：$112^2 \times 24 \times 96 \times 2 = 57.8M$ FLOPs
   - 数据量：
     - 输入：$112^2 \times 24 \times 4 = 1.2$ MB（可重用上一步输出）
     - 权重：$24 \times 96 \times 4 = 9.2$ KB
     - 输出：$112^2 \times 96 \times 4 = 4.8$ MB
   - 算术强度：$\frac{57.8M}{6.0M} = 9.6$ FLOPs/byte

优化策略：
1. 融合执行：将depthwise和pointwise融合，避免中间结果写回
2. 空间tiling：将112×112分成7×7个16×16的tile
3. 通道分块：将96个输出通道分成3批，每批32通道

通过这些优化，可将总数据传输量从8.4 MB降低到2.1 MB，提高4倍性能。

### 4.2.2 Loop Tiling与Blocking策略

**基本Tiling原理**

Loop tiling将大的迭代空间分解为小块，使每块的工作集适配片上缓存：

原始循环：
```
for i = 0 to M-1:
    for j = 0 to N-1:
        for k = 0 to K-1:
            C[i][j] += A[i][k] * B[k][j]
```

Tiled版本：
```
for i_outer = 0 to M-1 step T_i:
    for j_outer = 0 to N-1 step T_j:
        for k_outer = 0 to K-1 step T_k:
            // 内层循环处理tile
            for i = i_outer to min(i_outer+T_i, M)-1:
                for j = j_outer to min(j_outer+T_j, N)-1:
                    for k = k_outer to min(k_outer+T_k, K)-1:
                        C[i][j] += A[i][k] * B[k][j]
```

**Tile大小选择**

最优tile大小需要满足：
1. 容量约束：$T_i \times T_k + T_k \times T_j + T_i \times T_j \leq S$
2. 带宽平衡：使计算时间与数据传输时间相匹配

对于计算密度为 $\rho$ FLOPS/cycle 的NPU：
$$T_{\text{opt}} = \sqrt{\frac{S \times \rho \times f_{\text{clk}}}{3 \times BW}}$$

其中 $f_{\text{clk}}$ 是时钟频率，$BW$ 是内存带宽。

**多级Tiling**

针对多级存储层次，采用递归tiling：

```
L3 Cache Tile: 256×256×256
    │
    └─► L2 Cache Tile: 64×64×64
            │
            └─► L1 Cache Tile: 16×16×16
                    │
                    └─► Register Tile: 4×4×4
```

每级的tile大小比例通常遵循：
$$\frac{T_{L_{i+1}}}{T_{L_i}} \approx \sqrt[3]{\frac{S_{L_{i+1}}}{S_{L_i}}}$$

**Tiling搜索空间优化**

对于一个6层嵌套循环的卷积操作（N, C, H, W, K, R, S），tiling参数空间为：
$$|\mathcal{S}| = \prod_{i=1}^{7} d_i$$

其中$d_i$是维度$i$的可能分块大小数。对于典型的224×224×256卷积层，搜索空间可达$10^{15}$。

常用优化方法：
1. 解析模型：基于屋顶线模型预测性能
   $$T_{exec} = \max(T_{compute}, T_{memory})$$
   
2. 机器学习方法：使用XGBoost/LSTM预测最优tiling
   - 训练数据：随机采样的10000个配置
   - 特征：tile大小、重用率、带宽需求
   - 目标：实测执行时间

3. 遗传算法：平衡探索与利用
   - 初始种群：基于启发式规则
   - 适应度函数：$f = \alpha \times \text{FLOPS} - \beta \times \text{Energy}$
   - 变异策略：随机扰动tile大小

实践中，通常结合三种方法：解析模型快速筛选，ML模型精细调优，遗传算法处理边缘情况。

### 4.2.3 Dataflow分类：WS/OS/RS

**Weight Stationary (WS)**

权重驻留数据流中，权重保持在PE本地，输入激活和部分和在PE间流动：

```
特征：
- 每个PE存储部分权重
- 输入激活广播或单播
- 部分和累加并传递

优势：
- 权重重用最大化
- 适合batch size大的场景
- 减少权重读取能耗

劣势：
- 输入/输出数据传输开销大
- 对irregular网络支持差
```

数学表达：
$$\text{数据传输量}_{WS} = \text{Batch} \times (H \times W \times C_{in} + H \times W \times C_{out})$$

**Output Stationary (OS)**

输出驻留数据流中，部分和保持在PE本地直到完成累加：

```
特征：
- 每个PE负责特定输出元素
- 权重和激活流经PE
- 部分和本地累加

优势：
- 部分和不需要传输
- 减少累加器带宽需求
- 适合深度网络

劣势：
- 权重和激活重用受限
- PE利用率可能不均
```

数学表达：
$$\text{数据传输量}_{OS} = K \times H \times W \times (C_{in} + C_{out})$$

**Row Stationary (RS)**

行驻留数据流试图在PE内最大化所有数据类型的重用：

```
特征：
- 1D卷积原语映射到PE行
- 权重、激活、部分和都有重用
- 灵活支持不同层类型

优势：
- 能量效率最优
- 支持多种数据流模式
- 硬件利用率高

劣势：
- 控制复杂度高
- 编译器映射挑战大
```

能量模型比较：
$$E_{total} = E_{RF} \times N_{RF} + E_{NoC} \times N_{NoC} + E_{DRAM} \times N_{DRAM}$$

其中典型能量成本（45nm）：
- $E_{RF}$ ≈ 1 pJ
- $E_{NoC}$ ≈ 2-6 pJ  
- $E_{DRAM}$ ≈ 200 pJ

**实际案例：Eyeriss v2的RS数据流实现**

Eyeriss v2采用分层式RS数据流，针对不同层类型自适应调整：

1. CONV层映射：
   - 每个PE处理1D卷积（1×1×K）
   - 水平PE共享输入激活
   - 垂直PE共享部分和
   - 对角PE共享权重

2. FC层映射：
   - 切换到OS模式
   - 每个PE负责1个输出神经元
   - 权重广播至所有PE

3. Depthwise层映射：
   - 每个PE处理一个通道
   - 无需PE间通信
   - 最大化并行度

能效对比（AlexNet CONV2层）：
- WS：1.4 TOPS/W
- OS：1.6 TOPS/W  
- RS：2.5 TOPS/W

RS通过灵活的数据流切换，在不同层类型上都能达到较高效率。但代价是控制逻辑复杂度增加约40%，面积开销增加15%。

## 4.3 DMA设计与数据预取

DMA（Direct Memory Access）是NPU中实现计算与数据传输重叠的关键组件。通过精心设计的DMA系统和预取策略，可以有效隐藏内存访问延迟，提高整体系统性能。

### 4.3.1 DMA架构设计

**基本DMA架构**

现代NPU的DMA控制器通常包含以下组件：

```
┌─────────────────────────────────────┐
│         DMA Controller              │
├──────────┬──────────┬──────────────┤
│ Command  │ Address  │   Data       │
│  Queue   │Generator │   Buffers    │
├──────────┴──────────┴──────────────┤
│         Arbitration Logic           │
└──────────┬──────────────────────────┘
           │
     ┌─────▼─────┐
     │   Memory  │
     │ Interface │
     └───────────┘
```

关键设计参数：
- 通道数量：典型8-32个独立DMA通道
- 突发长度：64-256字节（匹配内存系统）
- 队列深度：16-64个待处理请求
- 支持模式：线性、2D/3D块传输、scatter-gather

**多通道DMA设计考量**

在自动驾驶NPU中，不同通道通常专门用于特定数据类型：

1. 相机数据通道（6个）：
   - 实时从MIPI/GMSL接口接收
   - 时延要求：< 10ms
   - 带宽：6 × 4MP × 30fps × 12bit = 3.5 GB/s

2. 特征图通道（4个）：
   - 在多个网络层间传输中间特征
   - 支持2D/3D tensor转置
   - 带宽：~20 GB/s

3. 权重加载通道（2个）：
   - 预加载下一层权重
   - 支持2:4稀疏解压
   - 带宽：~5 GB/s

4. 输出通道（2个）：
   - 将结果写回主存
   - 支持NMS后处理
   - 带宽：~2 GB/s

通道优先级设置：
$$P_i = \alpha_i \times \text{latency\_sensitivity} + \beta_i \times \text{bandwidth\_demand}$$

其中相机数据通道的$\alpha$最高，确保实时性。

**带宽分配策略**

多通道DMA的带宽分配可建模为优化问题：

$$\max \sum_{i=1}^{N} w_i \times \min(r_i, b_i)$$

约束条件：$\sum_{i=1}^{N} b_i \leq B_{total}$

其中：
- $w_i$：通道 $i$ 的优先级权重
- $r_i$：通道 $i$ 的请求带宽
- $b_i$：分配给通道 $i$ 的带宽
- $B_{total}$：总可用带宽

**地址生成模式**

支持复杂访问模式的地址生成器：

1. 线性模式：
   $$\text{addr} = \text{base} + \text{offset} \times \text{stride}$$

2. 2D块模式：
   $$\text{addr} = \text{base} + x \times \text{stride}_x + y \times \text{stride}_y$$

3. 循环缓冲模式：
   $$\text{addr} = \text{base} + (\text{offset} \bmod \text{size})$$

4. 3D tensor模式（用于BEV特征）：
   $$\text{addr} = \text{base} + z \times S_z + y \times S_y + x \times S_x$$
   
   其中$S_z = H \times W \times C$, $S_y = W \times C$, $S_x = C$

5. 稀疏索引模式（用于2:4稀疏）：
   $$\text{addr} = \text{base} + \text{index}[i] \times \text{elem\_size}$$

**地址生成器的硬件实现**

```
     ┌─────────────────┐
     │  Base Register  │
     └───────┬─────────┘
              │
     ┌────────┴─────────┐
     │   Counter Bank   │
     │  [X][Y][Z][N]    │
     └───────┬──────────┘
              │
     ┌────────┴─────────┐
     │  Stride Multiply │
     └────────┬─────────┘
              │
     ┌────────┴─────────┐
     │   Accumulator    │
     └────────┬─────────┘
              │
         Final Address
```

关键优化：
- 使用移位代替乘法（当stride为2的幂）
- 多级计数器级联，减少进位链延迟
- 预计算常用stride组合

### 4.3.2 描述符与链表管理

**描述符格式**

典型的DMA描述符包含：

```
struct DMADescriptor {
    uint64_t src_addr;      // 源地址
    uint64_t dst_addr;      // 目标地址
    uint32_t length;        // 传输长度
    uint16_t src_stride;    // 源跨步
    uint16_t dst_stride;    // 目标跨步
    uint8_t  burst_len;     // 突发长度
    uint8_t  flags;         // 控制标志
    uint64_t next_desc;     // 下一描述符指针
};
```

**链表管理机制**

支持多种链表组织方式：

1. 单链表：简单但灵活性有限
2. 循环链表：适合重复的传输模式
3. 树形结构：支持条件分支和复杂控制流

描述符预取优化：
- Shadow寄存器：预加载下一个描述符
- 描述符缓存：减少重复读取开销
- 批量更新：减少描述符写回次数

**描述符压缩技术**

为减少描述符存储开销，可采用压缩编码：

1. 差分编码：存储与前一描述符的差值
   $$\text{desc}_i = \text{desc}_{i-1} + \Delta_i$$

2. 模板编码：预定义常用模板
   - 模板0：连续线性传输
   - 模板1：2D块传输（图像）
   - 模板2：3D tensor传输

3. 位域压缩：减少未使用位
   ```
   原始：64bit地址 + 32bit长度 + 16bit stride = 112 bits
   压缩：32bit基址 + 16bit偏移 + 16bit长度 + 8bit stride = 72 bits
   ```

通过这些技术，可将描述符存储开销减少40-60%。

### 4.3.3 预取策略与隐藏延迟

**静态预取策略**

编译时确定的预取模式：

```
计算时间线：|--Compute Layer N--|--Compute Layer N+1--|
数据传输：      |--Prefetch N+1--|    |--Prefetch N+2--|
```

预取距离计算：
$$D_{prefetch} = \lceil \frac{L_{mem}}{T_{compute}} \rceil$$

其中：
- $L_{mem}$：内存访问延迟
- $T_{compute}$：计算一个tile的时间

**动态预取策略**

运行时自适应的预取机制：

1. 基于历史的预测：
   $$\text{addr}_{next} = \text{addr}_{current} + \alpha \times \Delta_{history}$$

2. 基于stride的检测：
   - 检测连续访问的stride模式
   - 自动触发相应的预取请求

**双缓冲与流水线**

双缓冲机制的时序分析：

```
Buffer A: |--Load--|--Compute--|--Store--|
Buffer B:          |--Load--|--Compute--|--Store--|
时间:     0       T        2T        3T        4T
```

理想情况下的效率：
$$\eta_{pipeline} = \frac{T_{compute}}{\max(T_{compute}, T_{load}, T_{store})}$$

实现200 TOPS的系统，假设：
- 计算密度：1000 GFLOPS/s
- 所需数据量：100 GB/s
- DMA延迟：100 cycles

则需要的预取深度：
$$N_{prefetch} = \lceil \frac{100 \text{ cycles} \times 1 \text{ GHz}}{1000 \text{ GFLOPS} / 100 \text{ GB/s}} \rceil = 10$$

**预取性能建模**

考虑预取准确率和覆盖率：

$$\text{Speedup} = \frac{1}{(1-h) + \frac{h}{1+\alpha \times p}}$$

其中：
- $h$：预取命中率
- $p$：预取覆盖率（隐藏的延迟比例）  
- $\alpha$：预取开销系数

实际测量数据（ResNet-50）：
- 静态预取：$h=0.95$, $p=0.8$, 加速1.7倍
- 动态预取：$h=0.85$, $p=0.9$, 加速1.6倍
- 混合策略：$h=0.92$, $p=0.85$, 加速1.75倍

## 4.4 片上网络(NoC)基础

片上网络是NPU内部各计算单元、存储单元和I/O接口之间的通信基础设施。良好的NoC设计对于实现高效的数据流和低延迟通信至关重要。

### 4.4.1 NoC拓扑结构

**常见拓扑对比**

```
总线(Bus):          环形(Ring):        网格(Mesh):
┌─┬─┬─┬─┬─┐        ┌──┐──┌──┐        ┌──┬──┬──┐
│ │ │ │ │ │        │  └──┘  │        │  │  │  │
└─┴─┴─┴─┴─┘        │        │        ├──┼──┼──┤
                    │  ┌──┐  │        │  │  │  │
                    └──┘  └──┘        ├──┼──┼──┤
                                      │  │  │  │
                                      └──┴──┴──┘
```

性能特征比较：
| 拓扑 | 直径 | 分割带宽 | 成本 | 扩展性 |
|------|------|----------|------|--------|
| 总线 | 1 | O(1) | 低 | 差 |
| 环形 | N/2 | O(2) | 低 | 中 |
| 2D网格 | 2√N | O(√N) | 中 | 好 |
| Torus | √N | O(2√N) | 高 | 好 |
| 胖树 | logN | O(N) | 高 | 优秀 |

对于200 TOPS的NPU，典型选择16×16的2D Mesh，提供：
- 256个节点
- 最大跳数：30
- 分割带宽：16×链路带宽

### 4.4.2 路由算法与流控

**维序路由(Dimension-Order Routing)**

最常用的死锁避免路由算法，先沿X维路由，再沿Y维：

```python
def xy_routing(src_x, src_y, dst_x, dst_y):
    # 先X后Y，避免死锁
    path = []
    # X维路由
    while src_x != dst_x:
        if src_x < dst_x:
            path.append("EAST")
            src_x += 1
        else:
            path.append("WEST") 
            src_x -= 1
    # Y维路由
    while src_y != dst_y:
        if src_y < dst_y:
            path.append("NORTH")
            src_y += 1
        else:
            path.append("SOUTH")
            src_y -= 1
    return path
```

**自适应路由**

根据网络拥塞动态选择路径：

$$P_{route} = \arg\min_{p \in \text{paths}} \sum_{l \in p} C_l$$

其中$C_l$是链路$l$的拥塞度量。

**虚通道(Virtual Channel)**

通过多个虚拟通道共享物理链路，提高利用率并避免死锁：

```
物理链路
┌─────────────────────┐
│  VC0: [████    ]    │
│  VC1: [  ████  ]    │
│  VC2: [      ████]  │
│  VC3: [██      ]    │
└─────────────────────┘
```

虚通道分配策略：
- 静态分配：不同消息类型使用固定VC
- 动态分配：基于信用的VC分配
- 逃逸通道：保留一个VC用于死锁恢复

### 4.4.3 流控机制

**信用流控(Credit-based Flow Control)**

接收方向发送方发放信用，控制数据发送：

```
发送方                     接收方
┌──────┐                 ┌──────┐
│Buffer│ ──data(3 flits)─> │Buffer│
│      │ <──credit(3)──── │      │
└──────┘                 └──────┘
```

信用计算：
$$\text{Credits} = \lceil \frac{\text{RTT} \times BW}{\text{flit\_size}} \rceil + \text{buffer\_depth}$$

**背压机制(Backpressure)**

当下游拥塞时，向上游传播停止信号：

$$\text{Throughput} = \min_{i \in \text{path}}(\text{capacity}_i \times (1 - \text{congestion}_i))$$

### 4.4.4 NoC性能建模

**延迟模型**

端到端延迟包含多个组成部分：

$$L_{total} = L_{header} + L_{router} \times H + L_{link} \times H + L_{contention}$$

其中：
- $L_{header}$：包头处理延迟
- $L_{router}$：单跳路由延迟（2-4 cycles）
- $L_{link}$：链路传输延迟（1 cycle）
- $H$：跳数
- $L_{contention}$：竞争延迟

**带宽模型**

有效带宽受多个因素影响：

$$BW_{effective} = BW_{physical} \times \eta_{protocol} \times \eta_{routing} \times (1 - P_{conflict})$$

典型效率值：
- $\eta_{protocol}$：0.8-0.9（协议开销）
- $\eta_{routing}$：0.7-0.85（路由效率）
- $P_{conflict}$：0.1-0.3（冲突概率）

**实际案例：Google TPU v4的ICI**

TPU v4使用3D Torus拓扑的Inter-Chip Interconnect (ICI)：
- 4096个芯片互连
- 每芯片6个100 Gbps链路
- 总分割带宽：4.8 Tbps
- 平均延迟：< 5μs
- 支持全规约带宽：340 GB/s

优化技术：
1. 自适应路由避免热点
2. 多轨道减少冲突
3. 硬件集合通信原语
4. 拥塞感知的流调度