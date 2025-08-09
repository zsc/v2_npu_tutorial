# 第10章：TSP微架构设计

本章深入探讨Groq Tensor Streaming Processor (TSP)的微架构设计。TSP作为数据流架构的典型代表，通过编译时确定性调度和大规模片上存储，实现了极低延迟的推理性能。我们将从计算单元、存储系统和片上网络三个核心维度剖析TSP的设计理念，理解其如何通过消除运行时调度开销和外部存储访问来达到确定性的超低延迟。

## 10.1 计算单元设计

TSP的计算核心由三类功能单元组成：向量ALU阵列、矩阵乘法单元(MXU)和特殊函数单元(SFU)。这些单元通过精心设计的数据通路和控制逻辑，实现了高效的张量运算。

### 10.1.1 Vector ALU阵列架构

TSP采用超长指令字(VLIW)架构，每个计算tile包含320个向量ALU，分为20个功能单元组(FUG)，每组16个ALU。这种分层设计既保证了高并行度，又简化了控制逻辑。

向量ALU的关键设计参数：
- 向量宽度：320个元素（对应320个ALU）
- 数据类型：支持INT8/FP16/BF16
- 运算类型：加减乘除、比较、逻辑运算、移位

功能单元组(FUG)的设计考虑：
```
FUG架构：
┌─────────────────────────────────┐
│  Instruction Decoder            │
├─────────────────────────────────┤
│  ALU0  ALU1  ALU2  ... ALU15   │
├─────────────────────────────────┤
│  Local Register File (LRF)     │
├─────────────────────────────────┤
│  Crossbar Switch               │
└─────────────────────────────────┘
```

每个FUG内部的16个ALU共享一个局部寄存器文件(LRF)，减少了全局互连的复杂度。ALU之间通过crossbar实现灵活的数据交换，支持shuffle、broadcast等操作。

向量运算的吞吐量计算：
$$\text{Throughput}_{\text{VALU}} = N_{\text{tiles}} \times N_{\text{ALU}} \times f_{\text{clock}} \times \text{OPS/cycle}$$

对于200 TOPS的设计目标，假设时钟频率为1.25 GHz：
$$N_{\text{tiles}} = \frac{200 \times 10^{12}}{320 \times 1.25 \times 10^9 \times 2} = 250 \text{ tiles}$$

### 10.1.2 Matrix Multiplication Unit设计

MXU是TSP处理深度学习工作负载的核心，采用了独特的向量-矩阵乘法架构，而非传统的矩阵-矩阵乘法。

MXU架构特点：
- 计算模式：向量×矩阵 → 向量
- 矩阵维度：320×320（匹配向量宽度）
- 数值精度：INT8/FP16/BF16/FP32
- 稀疏支持：2:4结构化稀疏

MXU的数据流组织：
```
输入向量 (1×320)
     ↓
┌──────────────────┐
│  Weight Matrix   │ 320×320
│  (Streaming)     │
├──────────────────┤
│  MAC Array       │
│  320×320 PEs     │
├──────────────────┤
│  Accumulator     │
└──────────────────┘
     ↓
输出向量 (1×320)
```

与传统脉动阵列的关键区别在于权重的流式处理。TSP的MXU不存储权重，而是从片上SRAM流式读取，这种设计虽然增加了带宽需求，但提供了更大的灵活性。

MXU性能分析：
- 每周期运算量：$320 \times 320 \times 2 = 204,800$ ops（MAC算2次运算）
- 权重带宽需求：$320 \times 320 \times \text{precision} \times f_{\text{clock}}$
- 对于FP16：$320^2 \times 2 \times 1.25 \text{GHz} = 256 \text{GB/s}$

### 10.1.3 特殊函数单元(SFU)

SFU负责处理非线性激活函数和其他特殊运算，是神经网络推理的必要组件。

SFU支持的函数类型：
1. 激活函数：ReLU、GELU、Sigmoid、Tanh、SiLU
2. 归一化：LayerNorm、RMSNorm、Softmax  
3. 数学函数：exp、log、sqrt、reciprocal
4. 量化/反量化操作

SFU的实现策略：
- 查找表(LUT)：用于复杂函数的快速近似
- 多项式逼近：Taylor级数或Chebyshev多项式
- 迭代算法：Newton-Raphson用于倒数和平方根

以GELU激活函数为例，其数学定义：
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

实际实现采用分段多项式逼近：
$$\text{GELU}_{\text{approx}}(x) = \begin{cases}
0 & x < -3 \\
x \cdot (a_3x^3 + a_2x^2 + a_1x + a_0) & -3 \leq x \leq 3 \\
x & x > 3
\end{cases}$$

SFU的流水线设计需要平衡精度和延迟：
- 查表阶段：1-2周期
- 插值/计算：2-3周期
- 后处理：1周期
- 总延迟：4-6周期

## 10.2 片上存储系统

TSP的革命性创新之一是完全消除了外部DRAM，所有数据都存储在片上SRAM中。这种设计虽然限制了模型大小，但实现了确定性的低延迟。

### 10.2.1 分布式SRAM Banks组织

TSP采用分布式SRAM架构，每个计算tile配备独立的存储块：

存储层次结构：
```
Global Memory Space (逻辑视图)
         ↓
┌────────────────────────────┐
│   Tile 0    │   Tile 1     │
│  ┌──────┐  │  ┌──────┐    │
│  │SRAM  │  │  │SRAM  │    │
│  │Banks │  │  │Banks │    │
│  └──────┘  │  └──────┘    │
│             │              │
│  ┌──────┐  │  ┌──────┐    │
│  │Comp  │  │  │Comp  │    │
│  │Units │  │  │Units │    │
│  └──────┘  │  └──────┘    │
└────────────────────────────┘
```

每个tile的存储配置：
- SRAM容量：1.5 MB
- Bank数量：16个banks
- Bank宽度：512 bits
- 访问延迟：2周期（单端口）

总片上存储计算：
$$\text{Total SRAM} = N_{\text{tiles}} \times \text{SRAM}_{\text{per tile}} = 250 \times 1.5 \text{MB} = 375 \text{MB}$$

这足以存储中等规模的神经网络模型（如BERT-Base的参数约340MB）。

### 10.2.2 地址生成与仲裁机制

分布式存储需要复杂的地址生成和仲裁机制来协调多个计算单元的访问请求。

地址生成单元(AGU)设计：
- 支持stride访问模式
- 2D/3D张量索引计算
- 循环buffer管理
- 地址交织(interleaving)

地址计算公式（以3D张量为例）：
$$\text{Addr} = \text{base} + i \times \text{stride}_i + j \times \text{stride}_j + k \times \text{stride}_k$$

仲裁器设计采用两级结构：
1. **Tile内仲裁**：处理本地计算单元的请求
   - Round-robin或优先级调度
   - Bank冲突检测与处理
   
2. **Tile间仲裁**：处理跨tile的数据访问
   - 基于credit的流控
   - 虚通道防止死锁

Bank冲突分析：
假设16个banks，访问模式为stride-s：
$$P_{\text{conflict}} = \begin{cases}
0 & \gcd(s, 16) = 1 \\
1 & s = 16k, k \in \mathbb{Z} \\
\frac{\gcd(s, 16)}{16} & \text{otherwise}
\end{cases}$$

### 10.2.3 Multi-casting机制

Multi-casting是TSP优化数据重用的关键技术，允许一次读取服务多个消费者。

Multi-cast树的构建：
```
Source Tile
    │
    ├─→ Consumer Group A
    │     ├─→ Tile i
    │     └─→ Tile j
    │
    └─→ Consumer Group B
          ├─→ Tile k
          └─→ Tile l
```

Multi-cast效率分析：
- 单播带宽：$B_{\text{unicast}} = N_{\text{consumers}} \times \text{data\_size} \times f$
- 多播带宽：$B_{\text{multicast}} = \text{data\_size} \times f$
- 带宽节省比：$\eta = 1 - \frac{1}{N_{\text{consumers}}}$

对于权重广播场景（如batch处理），multi-cast可以显著降低带宽需求。以batch=32为例：
$$\text{Bandwidth\_reduction} = 1 - \frac{1}{32} = 96.875\%$$

## 10.3 片上网络设计

TSP的片上网络(NoC)负责连接所有计算和存储资源，是实现高效数据流的关键基础设施。

### 10.3.1 2D Mesh拓扑结构

TSP采用2D mesh拓扑，每个节点通过4个方向（东西南北）与邻居连接：

```
┌───┬───┬───┬───┐
│ R │ R │ R │ R │
├───┼───┼───┼───┤
│ R │ R │ R │ R │
├───┼───┼───┼───┤
│ R │ R │ R │ R │
├───┼───┼───┼───┤
│ R │ R │ R │ R │
└───┴───┴───┴───┘
R = Router + Compute Tile
```

Mesh拓扑的特性分析：
- 节点度：4（边界节点为2-3）
- 网络直径：$D = 2(\sqrt{N} - 1)$，其中N为节点数
- 平均跳数：$H_{avg} = \frac{2\sqrt{N}}{3}$
- 二分带宽：$B_{bisection} = 2\sqrt{N} \times B_{link}$

对于16×16的mesh（256个节点）：
- 网络直径：30跳
- 平均跳数：10.67跳
- 二分带宽：32×链路带宽

### 10.3.2 路由算法与死锁避免

TSP采用维序路由(Dimension-Order Routing, DOR)算法，先沿X轴路由，再沿Y轴路由。

路由决策逻辑：
```python
def route_decision(current, destination):
    if current.x < destination.x:
        return EAST
    elif current.x > destination.x:
        return WEST
    elif current.y < destination.y:
        return NORTH
    elif current.y > destination.y:
        return SOUTH
    else:
        return LOCAL
```

死锁避免策略：
1. **转向模型限制**：XY路由天然避免环形依赖
2. **虚通道分离**：请求/响应使用不同虚通道
3. **Escape通道**：预留紧急逃生路径

通道依赖图(CDG)分析：
$$\text{CDG}_{XY} = \{(E,N), (E,S), (W,N), (W,S)\}$$

由于没有形成环，XY路由保证无死锁。

### 10.3.3 虚通道与流控机制

虚通道(Virtual Channel, VC)技术通过逻辑上复用物理链路，提高网络利用率并避免头阻塞。

虚通道配置：
- VC数量：4个（请求、响应、多播、控制）
- 每VC缓冲深度：8 flits
- Flit宽度：512 bits

流控采用credit-based机制：
```
发送方                     接收方
┌─────────┐              ┌─────────┐
│ Credits │<─────────────│ Buffer  │
│ Counter │              │ Space   │
├─────────┤              ├─────────┤
│ Send    │─────Data────>│ Receive │
│ Logic   │<────Credit───│ Logic   │
└─────────┘              └─────────┘
```

Credit更新公式：
$$\text{Credits}_{t+1} = \text{Credits}_t - \text{Sent}_t + \text{Returned}_t$$

虚通道仲裁策略比较：
- Round-Robin：公平但可能低效
- Age-based：优先老请求，减少延迟方差
- Priority-based：区分QoS等级

网络吞吐量建模：
$$T_{network} = \min\left(\frac{N_{links} \times B_{link}}{H_{avg}}, T_{injection}\right)$$

其中$T_{injection}$是注入率上限，通常是理论吞吐量的60-80%。

## 本章小结

本章深入探讨了TSP微架构的三个核心组成部分：计算单元、片上存储系统和片上网络。

**关键概念总结：**

1. **计算单元设计**
   - Vector ALU阵列：320个ALU分组织，VLIW架构实现高并行度
   - MXU独特性：向量-矩阵乘法模式，权重流式处理
   - SFU优化：多种实现策略平衡精度与延迟

2. **片上存储系统**
   - 全片上存储：消除外部DRAM，375MB总容量
   - 分布式组织：每tile 1.5MB，16个banks
   - Multi-casting：显著降低广播带宽需求

3. **片上网络设计**
   - 2D Mesh拓扑：良好的可扩展性和规则性
   - XY路由：简单高效，天然无死锁
   - 虚通道技术：提高利用率，支持QoS

**关键公式回顾：**

向量运算吞吐量：
$$\text{Throughput}_{\text{VALU}} = N_{\text{tiles}} \times N_{\text{ALU}} \times f_{\text{clock}} \times \text{OPS/cycle}$$

MXU带宽需求：
$$B_{\text{MXU}} = \text{Matrix\_size} \times \text{precision} \times f_{\text{clock}}$$

Bank冲突概率：
$$P_{\text{conflict}} = \frac{\gcd(\text{stride}, N_{\text{banks}})}{N_{\text{banks}}}$$

网络平均跳数：
$$H_{avg} = \frac{2\sqrt{N}}{3}$$

**设计权衡分析：**

TSP架构通过以下设计选择实现了确定性低延迟：
- 牺牲模型大小灵活性，换取零外存访问延迟
- 牺牲部分硬件利用率，换取编译时确定性调度
- 牺牲通用性，专注于推理工作负载优化

这些权衡使TSP在特定场景下（中等规模模型的低延迟推理）具有显著优势，特别适合实时性要求高的应用场景。

## 练习题

### 基础题

**练习10.1** 计算题：假设TSP芯片有256个tiles，每个tile包含320个INT8 ALU，时钟频率1.5GHz。计算：
a) 理论峰值INT8运算性能（TOPS）
b) 若实际利用率为75%，实际性能是多少？
c) 要达到200 TOPS实际性能，需要多少利用率？

*Hint*：记住MAC运算算作2次操作。

<details>
<summary>参考答案</summary>

a) 理论峰值性能：
$$\text{Peak} = 256 \times 320 \times 1.5 \times 10^9 \times 2 = 245.76 \text{ TOPS}$$

b) 实际性能（75%利用率）：
$$\text{Actual} = 245.76 \times 0.75 = 184.32 \text{ TOPS}$$

c) 所需利用率：
$$\text{Utilization} = \frac{200}{245.76} = 81.4\%$$

</details>

**练习10.2** MXU带宽分析：一个320×320的MXU处理FP16数据，时钟频率1.25GHz。计算：
a) 权重流式读取的带宽需求
b) 输入向量和输出向量的带宽需求
c) 总带宽需求及与HBM3（819GB/s）的比较

*Hint*：考虑权重只读一次，输入输出可能有重用。

<details>
<summary>参考答案</summary>

a) 权重带宽：
$$B_{weight} = 320 \times 320 \times 2 \text{ bytes} \times 1.25 \text{ GHz} = 256 \text{ GB/s}$$

b) 输入输出带宽：
- 输入：$320 \times 2 \text{ bytes} \times 1.25 \text{ GHz} = 0.8 \text{ GB/s}$
- 输出：$320 \times 2 \text{ bytes} \times 1.25 \text{ GHz} = 0.8 \text{ GB/s}$

c) 总带宽：$256 + 0.8 + 0.8 = 257.6 \text{ GB/s}$
相当于HBM3带宽的31.5%，片上SRAM可以满足。

</details>

**练习10.3** 存储容量规划：某Transformer模型包含：
- Embedding: 50K×768 
- 12层Transformer，每层包含：
  - Self-attention: 4个768×768矩阵
  - FFN: 768×3072 + 3072×768
- 全部使用FP16存储

问：这个模型能否完全装入375MB的片上SRAM？

*Hint*：逐层计算参数量，FP16占2字节。

<details>
<summary>参考答案</summary>

参数量计算：
- Embedding: $50000 \times 768 = 38.4M$ 参数
- 每层Self-attention: $4 \times 768 \times 768 = 2.36M$ 参数  
- 每层FFN: $768 \times 3072 + 3072 \times 768 = 4.72M$ 参数
- 每层总计: $2.36M + 4.72M = 7.08M$ 参数
- 12层总计: $12 \times 7.08M = 84.96M$ 参数
- 模型总参数: $38.4M + 84.96M = 123.36M$ 参数

存储需求：$123.36M \times 2 \text{ bytes} = 246.72 \text{ MB}$

结论：可以装入375MB的SRAM，还有128MB余量用于激活值。

</details>

### 挑战题

**练习10.4** NoC性能建模：16×16 mesh网络，每条链路带宽10GB/s，平均数据包大小2KB。在均匀随机流量模式下：
a) 计算理论二分带宽
b) 估算网络饱和注入率
c) 若采用4个虚通道，每个8-flit深，需要多少缓冲存储？

*Hint*：饱和注入率约为二分带宽的60%。

<details>
<summary>参考答案</summary>

a) 理论二分带宽：
$$B_{bisection} = 2\sqrt{256} \times 10 \text{ GB/s} = 32 \times 10 = 320 \text{ GB/s}$$

b) 饱和注入率：
- 每节点注入率：$\frac{320 \times 0.6}{256} = 0.75 \text{ GB/s}$
- 包/秒：$\frac{0.75 \text{ GB/s}}{2 \text{ KB}} = 375K \text{ packets/s}$

c) 缓冲存储（假设flit=64B）：
- 每路由器：$5 \text{ ports} \times 4 \text{ VCs} \times 8 \text{ flits} \times 64B = 10KB$
- 全网：$256 \times 10KB = 2.56MB$

</details>

**练习10.5** 编译时调度优化：考虑以下计算图，需要在TSP上调度：
```
A(320×320) × B(320×1) → C(320×1)
D(320×320) × C(320×1) → E(320×1)  
E + F(320×1) → G(320×1)
```
假设MXU延迟10周期，VALU延迟2周期，数据传输1跳需1周期。设计一个调度方案最小化总延迟。

*Hint*：考虑操作之间的依赖关系和可能的并行机会。

<details>
<summary>参考答案</summary>

调度方案：
1. T0-T9: 执行A×B→C（MXU，10周期）
2. T10-T19: 执行D×C→E（MXU，10周期）
3. T20-T21: 执行E+F→G（VALU，2周期）

总延迟：22周期

优化考虑：
- A×B和D的加载可以并行
- C的结果可以直接转发给第二个MXU，避免存储往返
- F可以在E计算期间预取

进一步优化（如果D独立于A）：
- 可以预先开始加载D，隐藏部分延迟
- 使用double buffering减少等待

</details>

**练习10.6** 稀疏优化分析：对于2:4结构化稀疏，MXU需要额外的索引处理逻辑。假设：
- 稀疏索引编码：每4个元素用2-bit表示非零位置
- 索引解码延迟：1周期
- 稀疏MAC单元：面积减少40%

问：在面积受限情况下，稀疏MXU相比密集MXU的有效算力提升是多少？

*Hint*：考虑可以部署更多稀疏单元。

<details>
<summary>参考答案</summary>

分析：
1. 稀疏压缩率：50%（2:4表示4个中2个非零）
2. 单元面积比：稀疏/密集 = 0.6
3. 相同面积下稀疏单元数量：$\frac{1}{0.6} = 1.67$倍

有效算力计算：
- 密集MXU：1.0×（参考）
- 稀疏MXU：$1.67 \times 0.5 = 0.835$×

结论：纯算力下降16.5%，但考虑到：
- 权重存储减少50%，可以装下更大模型
- 带宽需求减少50%
- 某些层稀疏度>50%时更有优势

实际收益取决于模型稀疏度分布。

</details>

**练习10.7** 功耗优化策略：TSP的功耗分布大致为：
- 计算单元：40%
- 片上SRAM：35%  
- NoC：20%
- 控制逻辑：5%

设计一个动态功耗管理策略，在保证性能前提下降低功耗。考虑：
a) 哪些组件适合clock gating？
b) 如何实现细粒度的power gating？
c) DVFS的应用场景？

*Hint*：考虑不同工作负载的资源利用特征。

<details>
<summary>参考答案</summary>

动态功耗管理策略：

a) Clock gating适用组件：
- 空闲的ALU组（检测连续N周期无指令）
- 未使用的SRAM banks（根据地址访问模式）
- NoC中的空闲虚通道
- 预计节能：15-25%

b) Power gating实现：
- Tile级：根据模型映射，关闭未使用tiles
- FUG级：在向量宽度<320时关闭部分FUG
- 实现要求：
  - 状态保存/恢复机制
  - 唤醒延迟<100周期
  - 预计节能：20-30%（部分负载时）

c) DVFS应用：
- 内存受限kernel：降低计算频率，维持内存频率
- 计算受限kernel：提高计算频率，降低NoC频率
- 批处理大小自适应：
  - Small batch：降频以提高能效
  - Large batch：满频以提高吞吐
- 预计节能：10-15%

综合策略可实现25-40%的功耗降低。

</details>

**练习10.8** 开放思考题：如果要将TSP架构扩展支持训练（不仅是推理），需要哪些关键架构修改？分析每项修改的成本和收益。

*Hint*：考虑训练特有的需求如梯度计算、参数更新、高精度累加等。

<details>
<summary>参考答案</summary>

支持训练的关键架构修改：

1. **增加FP32累加器**
   - 原因：防止梯度下溢和累积误差
   - 成本：MXU面积增加约30%
   - 收益：支持混合精度训练

2. **双向数据流支持**
   - 原因：反向传播需要逆向数据流
   - 修改：NoC支持双向调度，增加缓冲
   - 成本：NoC复杂度增加，缓冲翻倍
   - 收益：高效的梯度传播

3. **扩展片上存储**
   - 原因：需要保存激活值用于反向传播
   - 修改：SRAM从375MB扩至1GB+
   - 成本：芯片面积增加40%+
   - 收益：支持更大batch size

4. **优化器单元**
   - 原因：Adam/SGD等参数更新
   - 新增：专用的参数更新单元
   - 成本：额外10%计算面积
   - 收益：参数更新不占用主计算资源

5. **高带宽外存接口**
   - 原因：大模型无法全部片上存储
   - 新增：HBM接口（1TB/s+）
   - 成本：功耗增加30%，需要额外IO
   - 收益：支持大模型训练

6. **AllReduce硬件**
   - 原因：分布式训练的梯度同步
   - 新增：专用reduction tree
   - 成本：额外网络硬件
   - 收益：高效多卡扩展

7. **动态调度能力**
   - 原因：训练时的动态批大小、梯度累积
   - 修改：增加运行时调度器
   - 成本：失去完全确定性
   - 收益：灵活性和利用率提升

综合评估：
- 最小可行修改：1+2+3（单卡小模型训练）
- 实用训练系统：1+2+3+4+5（单卡大模型）
- 生产级系统：全部修改（多卡大模型）

架构哲学转变：从"编译时确定"到"运行时自适应"，这是TSP推理优化和训练通用性之间的根本权衡。

</details>

## 常见陷阱与错误

### 1. 计算单元设计陷阱

**陷阱1.1**：过度追求峰值算力而忽视实际利用率
- 错误：设计超大规模MAC阵列，实际利用率仅30%
- 正确：平衡计算能力与数据供给能力

**陷阱1.2**：SFU精度不足导致精度损失累积
- 错误：过度简化激活函数实现
- 正确：关键路径使用高精度实现，非关键路径可以近似

**陷阱1.3**：忽视稀疏索引开销
- 错误：只计算稀疏计算节省，忽视索引处理开销
- 正确：综合评估索引存储、解码延迟和计算节省

### 2. 存储系统陷阱

**陷阱2.1**：Bank冲突导致性能骤降
- 错误：未考虑访问模式导致严重bank冲突
- 正确：设计时分析典型访问模式，优化bank数量和交织方式

**陷阱2.2**：Multi-cast树构建开销
- 错误：动态构建multi-cast树，延迟过高
- 正确：编译时预计算常用multi-cast模式

**陷阱2.3**：片上存储碎片化
- 错误：简单的首次适应分配导致碎片
- 正确：使用伙伴系统或预规划的内存池

### 3. 片上网络陷阱

**陷阱3.1**：局部热点导致网络拥塞
- 错误：未考虑流量不均匀性
- 正确：支持自适应路由或流量整形

**陷阱3.2**：虚通道Head-of-Line阻塞
- 错误：虚通道过少或分配不当
- 正确：根据流量类型合理分配虚通道

**陷阱3.3**：路由死锁
- 错误：自定义路由算法引入循环依赖
- 正确：严格验证CDG无环或使用成熟算法

### 调试技巧

1. **性能调试**：
   - 添加硬件性能计数器
   - 记录关键路径延迟分解
   - 可视化资源利用率热图

2. **功能调试**：
   - 实现cycle-accurate软件模拟器
   - 添加trace buffer记录关键事件
   - 支持确定性重放

3. **功耗调试**：
   - 细粒度功耗监控
   - 识别异常功耗模式
   - 动态功耗策略调优

## 最佳实践检查清单

### 架构设计审查

- [ ] **计算单元设计**
  - [ ] 算力与带宽是否平衡？
  - [ ] 是否支持主流算子高效映射？
  - [ ] 数值精度是否满足应用需求？
  - [ ] 特殊函数实现是否经过验证？

- [ ] **存储系统设计**
  - [ ] 容量是否满足目标模型需求？
  - [ ] Bank组织是否避免常见冲突模式？
  - [ ] 是否实现了有效的数据重用机制？
  - [ ] 地址生成是否支持复杂访问模式？

- [ ] **片上网络设计**
  - [ ] 拓扑选择是否适合通信模式？
  - [ ] 路由算法是否证明无死锁？
  - [ ] 带宽是否满足最坏情况需求？
  - [ ] 是否有拥塞控制机制？

### 实现验证要点

- [ ] **功能正确性**
  - [ ] 所有算子是否有bit-accurate参考模型？
  - [ ] 边界条件是否充分测试？
  - [ ] 并发访问是否正确处理？

- [ ] **性能验证**
  - [ ] 是否达到目标TOPS？
  - [ ] 关键kernel的利用率是否达标？
  - [ ] 延迟是否满足实时性要求？

- [ ] **功耗评估**
  - [ ] 是否在功耗预算内？
  - [ ] 动态功耗管理是否有效？
  - [ ] 热设计是否充分？

### 编译器协同

- [ ] **映射效率**
  - [ ] 编译器是否能高效映射目标模型？
  - [ ] 是否提供了必要的硬件原语？
  - [ ] 调度开销是否可接受？

- [ ] **调试支持**
  - [ ] 是否提供了充分的可观测性？
  - [ ] 是否支持性能profiling？
  - [ ] 错误报告是否完善？

### 系统集成

- [ ] **可扩展性**
  - [ ] 是否支持多芯片扩展？
  - [ ] 扩展效率如何？
  - [ ] 是否有标准接口？

- [ ] **软件栈**
  - [ ] 是否有完整的软件工具链？
  - [ ] 是否支持主流框架？
  - [ ] 部署流程是否简化？
