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

对于200 TOPS的NPU，假设MAC利用率为80%，nvfp4精度下：
$$\text{所需带宽} = 200 \times 10^{12} \times 0.8 \times 3 \times 4 \text{ bits/s} = 1.92 \text{ Pb/s} = 240 \text{ TB/s}$$

其中因子3来自两个输入操作数和一个输出，因子4是nvfp4的位宽。

**HBM技术演进**

HBM (High Bandwidth Memory) 提供了容量和带宽的良好平衡：

```
HBM代际对比：
         HBM2E    HBM3     HBM3E
带宽      460      819      1230    GB/s/stack
容量      16       24       36      GB/stack  
功耗效率   7        5.5      4.5     pJ/bit
```

**DDR与GDDR权衡**

DDR提供大容量但带宽有限，GDDR则在两者间取得平衡：
- DDR5：~50 GB/s/channel，容量可达128GB/DIMM
- GDDR6X：~80 GB/s/chip，容量2-4GB/chip
- 功耗：DDR ~20 pJ/bit，GDDR ~15 pJ/bit

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

**带宽效率指标**

定义带宽效率 $\eta_B$ 为实际计算吞吐量与理论峰值的比值：
$$\eta_B = \frac{\text{实际FLOPS}}{\min(\text{计算峰值}, \text{带宽} \times \text{算术强度})}$$

其中算术强度（Arithmetic Intensity）定义为：
$$AI = \frac{\text{FLOPs}}{\text{Bytes accessed}}$$

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

**冲突消除策略**

1. 地址交织（Interleaving）：
   $$\text{Bank}_{\text{id}} = (\text{addr} >> \text{offset}) \bmod N_{\text{banks}}$$

2. 素数Bank数量：选择素数个Bank（如17、31）减少规律性冲突

3. 双缓冲（Double Buffering）：计算与数据传输重叠

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