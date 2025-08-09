# 第6章：脉动阵列RTL实现

脉动阵列作为现代NPU的核心计算引擎，其RTL实现直接决定了芯片的性能、功耗和面积(PPA)指标。本章深入探讨脉动阵列从架构到RTL的实现细节，包括处理单元(PE)设计、阵列互连、控制器设计以及时序优化策略。我们将以200 TOPS的设计目标为例，支持nvfp4量化和2:4稀疏，详细分析各个模块的设计权衡和实现技巧。

## 6.1 PE (Processing Element) 设计

### 6.1.1 MAC单元架构

脉动阵列的基本计算单元是乘累加器(MAC)，每个PE包含一个MAC单元用于执行矩阵乘法的基本运算。对于支持nvfp4 (E2M1)格式的设计，MAC单元需要特殊的处理逻辑。

nvfp4数值表示范围：
$$V = (-1)^s \times 2^{e-bias} \times (1 + \frac{m}{2})$$

其中 $s$ 是符号位，$e \in \{0,1,2,3\}$ 是2位指数，$m \in \{0,1\}$ 是1位尾数。

```
        PE单元结构
    ┌─────────────────┐
    │   Weight Reg    │
    │   ┌─────────┐   │
    │   │  W_reg  │   │
    │   └────┬────┘   │
    │        │        │
    │   ┌────▼────┐   │
A_in──►─┤  Multiply │   │
    │   └────┬────┘   │
    │        │        │
    │   ┌────▼────┐   │
    │   │   Add    │◄──┼── Psum_in
    │   └────┬────┘   │
    │        │        │
    │   ┌────▼────┐   │
    │   │ Accumulator │
    │   └────┬────┘   │
    └────────┼────────┘
             │
         Psum_out
```

### 6.1.2 累加器设计

累加器需要支持更宽的位宽以防止溢出。对于nvfp4输入，累加器通常采用fp16或fp32格式：

累加器位宽计算：
- nvfp4乘法结果：最多需要5位指数，3位尾数
- N次累加后：需要额外 $\lceil \log_2(N) \rceil$ 位防止溢出
- 典型配置：32位累加器支持 $2^{16}$ 次累加

### 6.1.3 权重寄存器与预加载

权重固定(Weight-stationary)是TPU采用的核心设计理念，每个PE包含一个权重寄存器：

```
权重加载时序：
Clock:  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐
        ┘ └─┘ └─┘ └─┘ └─┘ └─┘ └─
W_load: ───┐     ┌─────────────
        ───┘     └─────────────
W_data: ─X─┬─X─┬─X─┬───────────
         W0│ W1│ W2│
W_reg:  ───┬───┬───┬───────────
         W0│ W1│ W2│
```

### 6.1.4 2:4稀疏支持

对于2:4结构化稀疏，每4个权重中恰好2个非零。PE需要额外的逻辑来处理稀疏索引：

稀疏索引编码（2位表示4选2）：
- 00: 位置[0,1]非零
- 01: 位置[0,2]非零  
- 10: 位置[0,3]非零
- 11: 位置[1,2]非零
- ...（共6种组合）

实际稀疏MAC运算量减少：
$$\text{Effective\_MACs} = \text{Dense\_MACs} \times \frac{2}{4} = 0.5 \times \text{Dense\_MACs}$$

### 6.1.5 时序约束与流水线

典型的PE时序路径包括：
1. 输入寄存：1个周期
2. 乘法运算：1-2个周期（取决于位宽）
3. 加法累加：1个周期
4. 输出寄存：1个周期

关键路径延迟估算：
$$T_{critical} = T_{reg} + T_{mult} + T_{add} + T_{mux} + T_{setup}$$

对于28nm工艺，1GHz目标频率：
- $T_{reg}$: ~50ps
- $T_{mult}$ (nvfp4): ~400ps  
- $T_{add}$ (fp32): ~300ps
- $T_{mux}$: ~100ps
- $T_{setup}$: ~50ps
- 余量: ~100ps

## 6.2 阵列级互连

### 6.2.1 数据广播网络

脉动阵列需要三种数据流：输入激活、权重和部分和。数据广播网络负责将输入分发到各个PE。

```
     16x16 脉动阵列数据流
     
     Input Activations (水平传播)
     A0 → A1 → A2 → ... → A15
     ↓    ↓    ↓         ↓
    PE00-PE01-PE02-...-PE0F → Psum
     ↓    ↓    ↓         ↓
    PE10-PE11-PE12-...-PE1F → Psum
     ↓    ↓    ↓         ↓
     :    :    :         :
     ↓    ↓    ↓         ↓
    PEF0-PEF1-PEF2-...-PEFF → Psum
     
     Weights (预加载到PE)
     Partial Sums (垂直传播)
```

### 6.2.2 Skew Buffer设计

为了实现脉动执行，输入数据需要斜向(skewed)进入阵列：

```
Skew Buffer结构：
        T=0   T=1   T=2   T=3
Row 0:  A00   A01   A02   A03
Row 1:  ---   A10   A11   A12  
Row 2:  ---   ---   A20   A21
Row 3:  ---   ---   ---   A30
```

Skew buffer深度计算：
$$D_{skew} = N_{rows} - 1$$

对于16x16阵列，需要15级缓冲器。

### 6.2.3 部分和累积链

部分和在垂直方向传递，每个PE将自己的MAC结果加到上一行传来的部分和上：

```
部分和传递时序：
       PE_00         PE_10         PE_20
T=0:   W00×A00      ---           ---
T=1:   W01×A01      W10×A00      ---  
T=2:   W02×A02      W11×A01      W20×A00
       +Psum_01     +Psum_00     
```

### 6.2.4 边界处理与Padding

矩阵维度不是阵列大小整数倍时需要padding：

实际利用率计算：
$$\eta = \frac{M \times N \times K}{⌈\frac{M}{S_m}⌉ \times S_m \times ⌈\frac{N}{S_n}⌉ \times S_n \times ⌈\frac{K}{S_k}⌉ \times S_k}$$

其中 $S_m, S_n, S_k$ 是阵列维度。

### 6.2.5 双缓冲机制

为了隐藏数据加载延迟，采用乒乓缓冲：

```
双缓冲时序：
        Buffer A        Buffer B
T0-T99: Computing      Loading W1
T100-199: Loading W2   Computing
T200-299: Computing    Loading W3
```

有效带宽需求：
$$BW_{required} = \frac{\text{Weight\_Size}}{\text{Compute\_Time}} = \frac{S_m \times S_n \times b_{weight}}{S_m \times S_n \times S_k / f_{clock}}$$

## 6.3 控制器设计

### 6.3.1 有限状态机设计

脉动阵列控制器采用分层FSM架构：

```
主状态机：
        ┌─────┐
        │IDLE │
        └──┬──┘
           │ start
        ┌──▼──┐
        │LOAD │──────┐
        └──┬──┘      │
           │ done    │ abort
        ┌──▼──┐      │
        │COMP │      │
        └──┬──┘      │
           │ done    │
        ┌──▼──┐      │
        │DRAIN│◄─────┘
        └──┬──┘
           │ done
        ┌──▼──┐
        │DONE │
        └─────┘
```

各状态持续时间：
- LOAD: $S_m$ 周期（权重加载）
- COMP: $S_k$ 周期（主计算）
- DRAIN: $S_n - 1$ 周期（结果输出）

### 6.3.2 计数器链设计

多级嵌套计数器用于生成地址：

```
计数器层次：
Level 0: PE内部计数 (0 to K-1)
Level 1: Tile行计数 (0 to M/Sm-1)
Level 2: Tile列计数 (0 to N/Sn-1)
Level 3: Batch计数  (0 to B-1)
```

地址生成公式：
$$Addr = Base + i \times Stride_i + j \times Stride_j + k \times Stride_k$$

### 6.3.3 依赖管理

控制器需要处理三种依赖：
1. RAW (Read After Write)：等待前序计算完成
2. WAR (Write After Read)：确保数据已被消费
3. WAW (Write After Write)：保持写入顺序

依赖检查逻辑：
```
if (dst_addr == pending_write_addr) {
    stall_pipeline();
} else if (src_addr == pending_write_addr) {
    wait_for_write_complete();
}
```

### 6.3.4 异常处理

需要处理的异常情况：
- 数值溢出/下溢
- 非法指令
- 内存访问越界
- ECC错误

异常优先级编码：
1. 硬错误（ECC不可纠正）
2. 访问违例
3. 数值异常
4. 软错误（ECC可纠正）

### 6.3.5 功耗管理

细粒度时钟门控：
```
Clock Gating条件：
- PE空闲：weight == 0 或 activation == 0
- 行空闲：整行PE未使用
- 列空闲：整列PE未使用
```

功耗节省估算：
$$P_{saved} = P_{dynamic} \times (1 - \eta_{utilization}) \times \alpha_{gating\_efficiency}$$

其中 $\alpha_{gating\_efficiency} \approx 0.9$。

## 本章小结

脉动阵列RTL实现的关键要点：

1. **PE设计权衡**：
   - MAC单元位宽vs面积/功耗
   - 累加器精度vs溢出风险
   - 流水线级数vs频率目标

2. **互连优化**：
   - Skew buffer深度最小化
   - 部分和链路延迟优化
   - 双缓冲隐藏加载延迟

3. **控制器复杂度**：
   - FSM状态数vs控制灵活性
   - 计数器链深度vs地址生成延迟
   - 异常处理完备性vs面积开销

4. **时序收敛策略**：
   - 关键路径识别：MAC > Add > Mux
   - 插入流水线寄存器
   - 时钟域交叉(CDC)处理

5. **验证重点**：
   - 边界条件：矩阵维度非对齐
   - 数值精度：累加误差分析
   - 性能瓶颈：带宽vs计算

关键性能公式汇总：

峰值算力：
$$TOPS = 2 \times S_m \times S_n \times f_{clock} \times 10^{-12}$$

实际算力：
$$TOPS_{effective} = TOPS_{peak} \times \eta_{utilization} \times \alpha_{sparsity}$$

能效比：
$$TOPS/W = \frac{TOPS_{effective}}{P_{dynamic} + P_{static}}$$

## 练习题

### 基础题

**6.1** 对于一个16×16的脉动阵列，工作频率1GHz，计算：
- a) 理论峰值算力（TOPS）
- b) 当执行8×24×32的矩阵乘法时的利用率
- c) 完成该矩阵乘法需要的周期数

<details>
<summary>提示</summary>
考虑矩阵分块和padding的影响，利用率 = 实际计算/总计算槽位
</details>

<details>
<summary>答案</summary>

a) 峰值算力：
$$TOPS = 2 \times 16 \times 16 \times 1 \times 10^{-12} = 0.512 \text{ TOPS}$$

b) 利用率计算：
- M=8需要1个tile，利用率: 8/16 = 50%
- N=24需要2个tiles，利用率: 24/32 = 75%  
- K=32需要2个tiles
- 总体利用率: (8×24×32)/(16×32×32) = 37.5%

c) 周期数：
- 加载时间: 16周期
- 计算时间: 32周期
- 输出时间: 15周期
- 总计: 2个tile × (16+32+15) = 126周期
</details>

**6.2** nvfp4格式乘法后，累加N次需要多少位的累加器才能保证不溢出？假设输入数据均匀分布在[-1, 1]范围内。

<details>
<summary>提示</summary>
考虑最坏情况：所有乘积同号且达到最大值
</details>

<details>
<summary>答案</summary>

nvfp4最大值约为3.5，乘积最大值约为12.25。
N次累加最坏情况：$12.25 \times N$

需要的指数位：
$$E_{bits} = \lceil \log_2(\log_2(12.25 \times N)) \rceil$$

对于N=256：需要约11位指数
对于N=4096：需要约15位指数

实践中使用fp32（8位指数）可支持约$10^{37}$的动态范围，足够大部分应用。
</details>

**6.3** 设计一个4×4脉动阵列的skew buffer，输入数据宽度为8位，画出其结构并计算所需的寄存器数量。

<details>
<summary>提示</summary>
每行需要不同的延迟，延迟量与行号成正比
</details>

<details>
<summary>答案</summary>

```
Skew Buffer结构：
Row 0: 直通 (0个寄存器)
Row 1: D─┐ (1个寄存器)
Row 2: D─D─┐ (2个寄存器)  
Row 3: D─D─D─┐ (3个寄存器)

总寄存器数 = 0+1+2+3 = 6个
每个寄存器8位
总位数 = 6 × 8 = 48位
```
</details>

### 挑战题

**6.4** 某脉动阵列支持2:4稀疏，设计一个高效的稀疏索引编码方案，使得：
- 可以用最少的位数表示4选2的所有组合
- 解码逻辑简单
- 支持快速的稀疏矩阵乘法

<details>
<summary>提示</summary>
4选2共有C(4,2)=6种组合，理论最少需要3位
</details>

<details>
<summary>答案</summary>

最优编码方案（3位）：
```
000: [1,1,0,0] - 位置0,1非零
001: [1,0,1,0] - 位置0,2非零
010: [1,0,0,1] - 位置0,3非零
011: [0,1,1,0] - 位置1,2非零
100: [0,1,0,1] - 位置1,3非零
101: [0,0,1,1] - 位置2,3非零
```

解码逻辑（组合逻辑）：
```
pos[0] = ~code[2] & ~code[1] | ~code[2] & code[0]
pos[1] = ~code[2] & ~code[0] | code[2] & ~code[1]
pos[2] = code[1] & ~code[0] | code[2] & code[1]
pos[3] = code[1] & code[0] | code[2] & ~code[0]
```

稀疏MAC实现只需2个乘法器而非4个，节省50%乘法器面积。
</details>

**6.5** 设计一个脉动阵列控制器的指令格式，支持：
- 矩阵乘法
- 逐元素运算
- 激活函数
要求指令长度不超过32位。

<details>
<summary>提示</summary>
考虑操作码、地址、大小参数的位分配
</details>

<details>
<summary>答案</summary>

32位指令格式：
```
[31:28] Opcode (4位)
  0000: GEMM
  0001: Element-wise Add
  0010: Element-wise Mul
  0100: ReLU
  0101: Sigmoid lookup
  
[27:24] Precision (4位)
  0000: FP32
  0001: FP16
  0010: nvfp4
  0100: INT8
  
[23:16] M dimension (8位, 最大256)
[15:8]  N dimension (8位, 最大256)
[7:0]   K dimension (8位, 最大256)

扩展指令（第二个32位字）：
[31:24] Base_addr_A[31:24]
[23:16] Base_addr_B[31:24]
[15:8]  Base_addr_C[31:24]
[7:0]   Stride/flags
```

这种设计支持最常见操作，复杂操作通过指令序列实现。
</details>

**6.6** 分析一个32×32脉动阵列在不同批大小(batch size)下的能效比。假设：
- 静态功耗：2W
- 动态功耗：8W (满载)
- 频率：1GHz
如何选择最优批大小？

<details>
<summary>提示</summary>
考虑利用率和功耗的关系，存在一个最优点
</details>

<details>
<summary>答案</summary>

峰值性能：$2 \times 32 \times 32 \times 1 = 2.048$ TOPS

不同batch size分析：

Batch=1, 矩阵32×32×32:
- 利用率: 100%
- 有效性能: 2.048 TOPS
- 功耗: 2+8=10W
- 能效: 0.2048 TOPS/W

Batch=4, 矩阵128×128×128:
- 需要分块: 4×4×4=64个tiles
- 利用率: 100%
- 有效性能: 2.048 TOPS
- 功耗: 10W
- 能效: 0.2048 TOPS/W

Batch=1, 矩阵16×16×16:
- 利用率: 12.5%
- 有效性能: 0.256 TOPS
- 动态功耗: 8×0.125=1W
- 总功耗: 2+1=3W
- 能效: 0.085 TOPS/W

结论：大矩阵(≥阵列大小)能效最优。小矩阵时，批处理合并可提升利用率。最优批大小使得合并后矩阵维度略大于阵列维度。
</details>

**6.7** 设计一个流水线深度为3的PE，分析其对阵列性能的影响。考虑：
- 数据依赖
- 控制复杂度
- 面积开销

<details>
<summary>提示</summary>
流水线会引入延迟，需要调整控制时序
</details>

<details>
<summary>答案</summary>

3级流水线PE设计：
```
Stage 1: 输入寄存 + 乘法前半
Stage 2: 乘法后半 + 加法
Stage 3: 累加 + 输出寄存
```

对阵列的影响：

1. 初始延迟增加：
   - 第一个结果需要3个周期
   - 整个阵列填充时间：3×(M+N-1)周期

2. 吞吐量不变：
   - 稳态后每周期still产生M×N个MAC

3. 控制复杂度：
   - 需要额外的valid信号传播
   - Skew buffer深度增加到N+2

4. 面积开销：
   - 每个PE增加2组流水线寄存器
   - 约增加20%的PE面积

5. 频率提升：
   - 关键路径从900ps降到300ps
   - 理论可达3GHz
   - 实际性能提升：3×0.8=2.4倍（考虑利用率下降）

权衡：当目标频率>1.5GHz时，3级流水线设计更优。
</details>

**6.8** 优化脉动阵列的数据复用，给定片上SRAM容量64KB，如何分配给Input、Weight和Output buffer以最大化复用？考虑AlexNet的Conv2层(27×27×256→13×13×384, 5×5卷积)。

<details>
<summary>提示</summary>
分析三种数据的复用机会，使用roofline模型
</details>

<details>
<summary>答案</summary>

Conv2参数分析：
- Input: 27×27×256 = 186KB
- Weight: 5×5×256×384 = 2.4MB
- Output: 13×13×384 = 65KB

复用分析：
1. Weight复用：每个权重使用13×13=169次
2. Input复用：每个输入使用5×5×384/256=37.5次
3. Output复用：每个输出累加5×5×256=6400次

最优buffer分配策略：
```
Output buffer: 32KB (存储部分输出通道)
Weight buffer: 24KB (存储多个卷积核)
Input buffer: 8KB (存储滑窗需要的行)
```

执行策略：
1. 循环顺序：Output_Channel → Input_Y → Input_X → Ky → Kx → Input_Channel
2. Tiling大小：
   - Output channels: 192 (一半)
   - Input channels: 64 (1/4)
   - 空间维度：逐行处理

带宽需求：
- Weight读取：5×5×64×192 = 307KB per tile
- Input读取：27×64 = 1.7KB per row
- Output读写：13×192×2 = 5KB per row

总带宽：~250GB/s @1GHz
实际片外带宽需求：~25GB/s (复用率10×)
</details>

## 常见陷阱与错误 (Gotchas)

### 1. 时序违例陷阱

**问题**：简单串联MAC单元导致组合逻辑路径过长

**错误示例**：
```
assign mac_out = a * w + psum_in; // 组合逻辑太长
```

**正确做法**：
```
always @(posedge clk) begin
    mult_reg <= a * w;
    mac_out <= mult_reg + psum_in;
end
```

### 2. 数值精度损失

**问题**：过早截断导致精度损失

**常见错误**：每次MAC后都截断到输入精度

**最佳实践**：使用更宽的累加器，只在最后截断

### 3. 死锁情况

**问题**：控制信号依赖形成环路

**典型场景**：
- Input buffer等待PE空闲
- PE等待output buffer
- Output buffer等待外部读取
- 外部等待input buffer

**解决方案**：设计清晰的优先级和超时机制

### 4. 资源冲突

**问题**：多个master同时访问同一SRAM bank

**症状**：仿真正确但综合后时序违例

**预防**：
- 采用多bank设计
- 实现仲裁器
- 使用双端口SRAM

### 5. 边界条件处理

**问题**：矩阵维度非2的幂次时地址计算错误

**容易出错的地方**：
- 最后一个tile的padding
- stride计算
- 循环边界

**调试技巧**：先测试对齐的情况，再测试各种非对齐组合

### 6. 功耗优化误区

**误区**：只关注计算单元功耗

**事实**：数据搬移功耗often超过计算
- 28nm工艺：32位加法~1pJ，32位SRAM读取~5pJ，DRAM读取~640pJ

**优化重点**：减少数据搬移，特别是片外访问

## 最佳实践检查清单

### RTL设计审查

- [ ] **时序收敛**
  - [ ] 识别并优化关键路径
  - [ ] 合理插入流水线寄存器
  - [ ] 避免过长的组合逻辑

- [ ] **功能正确性**
  - [ ] 所有控制状态都有退出条件
  - [ ] 异常情况都有处理
  - [ ] 边界条件测试完备

- [ ] **资源优化**
  - [ ] 共享乘法器资源
  - [ ] 复用存储器带宽
  - [ ] 最小化寄存器使用

- [ ] **可测试性**
  - [ ] 提供调试接口
  - [ ] 支持扫描链插入
  - [ ] 性能计数器设计

### 验证策略

- [ ] **功能验证**
  - [ ] 单元测试每个PE
  - [ ] 集成测试整个阵列
  - [ ] 系统测试实际workload

- [ ] **性能验证**
  - [ ] 测量实际利用率
  - [ ] 验证带宽需求
  - [ ] 确认功耗预算

- [ ] **边界测试**
  - [ ] 各种矩阵维度组合
  - [ ] 数值范围极限
  - [ ] 并发访问冲突

### 综合与实现

- [ ] **综合前检查**
  - [ ] 代码符合综合规范
  - [ ] 没有锁存器推断
  - [ ] 时钟域清晰定义

- [ ] **物理设计考虑**
  - [ ] 布局规划合理
  - [ ] 电源网络充足
  - [ ] 时钟树平衡

- [ ] **后仿真验证**
  - [ ] 门级仿真通过
  - [ ] 时序违例修复
  - [ ] 功耗分析达标