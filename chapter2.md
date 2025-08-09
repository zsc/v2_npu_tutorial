# 第2章：算法与算子分析

深度学习算法的计算特征直接决定了NPU架构的设计选择。本章深入分析自动驾驶和具身智能两大场景的核心算法，从网络结构、数据流模式、算子组成等多个维度剖析其计算需求。通过理解不同算法的工作负载特征，我们能够识别性能瓶颈、评估架构适配性，并指导后续的硬件设计优化。本章将建立算法需求与硬件能力之间的量化映射关系，为200 TOPS NPU的设计空间探索提供理论基础。

## 2.1 自动驾驶核心网络剖析

自动驾驶系统的感知栈涵盖了从2D图像检测到3D点云处理、从单帧感知到时序融合的多种网络架构。这些网络在计算密度、内存访问模式、数据重用机会等方面展现出显著差异。

### 2.1.1 2D检测网络：YOLO系列与CenterNet

#### YOLO系列架构演进

YOLO (You Only Look Once) 系列从v1到v8的演进体现了实时检测算法在精度和速度平衡上的持续优化。YOLOv8作为当前主流的实时检测网络，其backbone采用CSPDarknet架构，通过Cross Stage Partial连接减少计算量的同时保持特征表达能力。这种设计哲学对NPU架构提出了独特要求：需要高效支持残差连接、特征融合和多尺度处理。

**演进历程的架构洞察：**

从YOLOv1的全连接输出层到YOLOv8的解耦检测头，每代改进都反映了对硬件友好性的深入理解。YOLOv3引入的多尺度预测需要NPU支持高效的特征金字塔构建；YOLOv4的Mish激活和CSP结构要求灵活的激活函数单元；YOLOv5的Focus层通过空间到深度变换减少了早期层的计算量，这种pixel shuffle操作在NPU上可通过专门的数据重排单元加速。

每一代YOLO的改进都隐含着对硬件限制的深刻理解：

```
YOLOv1 (2016)              YOLOv3 (2018)              YOLOv5-8 (2020-2023)
     │                          │                              │
     ▼                          ▼                              ▼
┌─────────┐              ┌──────────────┐            ┌──────────────────┐
│全连接输出│              │多尺度FPN输出│            │解耦检测头+Anchor │
│7×7×30   │              │13×13,26×26,  │            │Free设计          │
│         │              │52×52         │            │                  │
└─────────┘              └──────────────┘            └──────────────────┘
     │                          │                              │
     ▼                          ▼                              ▼
内存瓶颈:                 计算分散:                    硬件友好:
- FC层参数量大            - 3个独立head                - 统一的检测流程
- 固定输入尺寸            - 不同stride特征             - 规则的tensor操作
- 低分辨率限制            - anchor匹配开销             - 易于量化和剪枝
```

YOLOv8的架构创新特别强调了与现代AI加速器的协同设计。其C2f模块的split-transform-merge模式天然适合多核并行架构，每个分支可以映射到不同的计算核心。这种设计哲学与Google TPU的设计理念不谋而合：通过架构规则化换取硬件效率的最大化。

**架构创新点：**

1. **C2f模块设计**：YOLOv8引入的C2f (Cross Stage Partial with 2 convolutions) 模块改进了YOLOv5的C3模块，通过更细粒度的特征分割实现了更好的梯度流动。

   C2f模块的设计精髓在于其分层特征复用策略：
   $$\text{C2f}(X) = \text{Concat}[\text{Conv}(X), \text{Bottleneck}_1(X_1), ..., \text{Bottleneck}_n(X_n)]$$
   
   这种结构在NPU上的映射需要考虑：
   - 分支计算的并行化机会：每个Bottleneck可独立计算，适合多核并行
   - Concat操作的内存重组开销：需要DMA单元支持scatter-gather操作
   - 多个Bottleneck的流水线调度：深度可配置的流水线寄存器
   
   **Bottleneck内部结构优化：**
   $$\text{Bottleneck}(X) = X + \text{Conv}_{3 \times 3}(\text{Conv}_{1 \times 1}(X))$$
   
   这个残差结构的关键在于1×1卷积降维和3×3卷积特征提取的平衡。在200 TOPS NPU上，可以将1×1卷积映射到向量单元，3×3卷积映射到脉动阵列，实现异构计算单元的协同。
   
   **数据流分析**：
   ```
   Input X ──┬──────────────────────────────► Addition ──► Output
            │                                     ▲
            └──► Conv1×1 ──► Conv3×3 ────────────┘
                 (降维)      (特征提取)
                 C→C/2       C/2→C/2→C
   
   内存访问模式:
   - 输入X读取: 1次
   - 中间特征: 保持在片上SRAM
   - 输出写回: 1次
   - 带宽需求: 2×H×W×C×sizeof(fp16)
   ```
   
   这种设计使得整个Bottleneck可以在单次数据加载后完成计算，极大降低了DDR带宽压力。对于典型的C=256通道，80×80特征图，单个Bottleneck仅需6.5MB的DDR访问，而计算量达到0.66 GFLOPs，算术强度高达101 ops/byte。

2. **Anchor-free检测头**：相比YOLOv5的anchor-based方法，YOLOv8采用了解耦检测头（Decoupled Head），将分类和回归任务分离：
   $$\text{Det}_{\text{cls}} = \text{Conv}_{3 \times 3}(\text{Conv}_{3 \times 3}(F)) \in \mathbb{R}^{H \times W \times N_{\text{cls}}}$$
   $$\text{Det}_{\text{reg}} = \text{Conv}_{3 \times 3}(\text{Conv}_{3 \times 3}(F)) \in \mathbb{R}^{H \times W \times 4}$$
   
   解耦设计允许独立优化两个分支的量化策略。分类分支可以使用INT8量化（对类别预测的微小偏差不敏感），而回归分支保持FP16以确保边界框的精确定位。
   
   **硬件映射优化**：
   ```
   Backbone最后一层特征 F
           │
           ▼
   ┌───────────────┐
   │  共享Conv3×3  │ (256 channels)
   └───────┬───────┘
           │
     ┌─────┴─────┐
     ▼           ▼
   ┌─────┐   ┌─────┐
   │Class│   │ Reg │
   │Head │   │Head │
   └─────┘   └─────┘
     INT8      FP16
   
   Pipeline调度:
   Stage 1: 共享特征提取 (利用率95%)
   Stage 2: 并行双头计算 (利用率85%)
   Stage 3: 后处理NMS   (利用率20%)
   ```
   
3. **TaskAligned Assigner的硬件影响**：
   
   YOLOv8采用的动态标签分配策略在训练时提高了正负样本的质量，但在推理时简化了后处理：
   $$\text{Alignment} = \text{cls\_score}^{\alpha} \times \text{IoU}^{\beta}$$
   
   这种设计避免了复杂的anchor匹配计算，减少了NPU的控制逻辑复杂度。

**计算特征分析：**

主干网络的计算量分布呈现金字塔特征，深层特征图尺寸小但通道数多：
$$\text{FLOPs}_{\text{backbone}} = \sum_{l=1}^{L} 2 \times C_{in}^{(l)} \times C_{out}^{(l)} \times K^{2(l)} \times H^{(l)} \times W^{(l)}$$

其中典型的下采样策略为：
- Stage 1: $640 \times 640 \times 3 \to 320 \times 320 \times 64$ (Conv-BN-SiLU, stride=2)
- Stage 2: $320 \times 320 \times 64 \to 160 \times 160 \times 128$ (C2f×3, stride=2)
- Stage 3: $160 \times 160 \times 128 \to 80 \times 80 \times 256$ (C2f×6, stride=2)
- Stage 4: $80 \times 80 \times 256 \to 40 \times 40 \times 512$ (C2f×6, stride=2)
- Stage 5: $40 \times 40 \times 512 \to 20 \times 20 \times 1024$ (C2f×3, SPPF)

**层级计算密度分析：**

不同stage的计算密度差异显著，影响NPU的资源调度：
$$\text{Density}_{\text{stage}} = \frac{\text{FLOPs}_{\text{stage}}}{\text{Memory}_{\text{stage}}}$$

每个stage的详细计算密度画像：

- 浅层（Stage 1-2）：特征图大（320×320, 160×160），计算密度低（~10 ops/byte），memory-bound特征明显。这些层的优化重点在于提高内存带宽利用率，可采用深度可分离卷积或组卷积降低内存压力。
- 中层（Stage 3-4）：计算密度适中（~50 ops/byte），是脉动阵列的理想工作负载。80×80和40×40的特征图大小恰好匹配典型的tile尺寸，可以实现接近峰值的计算效率。
- 深层（Stage 5）：通道数多（1024通道），计算密度高（>100 ops/byte），完全compute-bound。这里是应用2:4稀疏化的最佳位置，可以获得接近2倍的理论加速。

```
计算密度 (ops/byte)
200 ┤                                    ╱─── Stage 5 (Compute-bound)
    │                                 ╱╱╱    适合稀疏化
150 ┤                            ╱╱╱╱╱
    │                       ╱╱╱╱╱        ← AI_ridge = 100
100 ┤──────────────────╱╱╱╱╱─────────────── (200TOPS/2TB/s)
    │              ╱╱╱╱╱ Stage 3-4
 50 ┤         ╱╱╱╱╱      (Balanced)        
    │    ╱╱╱╱╱           适合脉动阵列
 10 ┤╱╱╱╱ Stage 1-2 (Memory-bound)
    │     需要带宽优化
  0 └────┬────┬────┬────┬────┬────┬────
      P1/2  P2/4  P3/8 P4/16 P5/32 Head
                  Network Depth →
```

这种计算密度的渐进式增长与生物视觉系统的分层处理类似：低层次提取简单特征（高带宽，低计算），高层次进行复杂推理（低带宽，高计算）。NPU设计应该针对这种模式提供自适应的资源配置。

**动态资源分配策略：**

基于计算密度的动态调度可以显著提升NPU利用率：
```
if (Density < 20) {
    // Memory-bound: 使用更多的内存通道，降低计算并行度
    配置: 4个内存通道, 1/4计算阵列
} else if (Density < 80) {
    // Balanced: 平衡配置
    配置: 2个内存通道, 1/2计算阵列
} else {
    // Compute-bound: 最大化计算资源
    配置: 1个内存通道, 全部计算阵列
}
```

**内存访问模式：**

CSP结构的特征图分割策略实现了梯度流的优化：
$$X = [X_1, X_2], \quad X_1 \in \mathbb{R}^{H \times W \times C/2}, X_2 \in \mathbb{R}^{H \times W \times C/2}$$

分割后的数据流：
$$Y = \text{Concat}[X_1, \text{DenseBlock}(X_2)]$$

这种分割降低了内存带宽需求：
$$\text{Bandwidth}_{\text{CSP}} = \text{Bandwidth}_{\text{standard}} \times (1 - \gamma)$$
其中 $\gamma \approx 0.3$ 为CSP的带宽节省率。

**SPPF (Spatial Pyramid Pooling Fast) 的优化实现：**

SPPF通过串行的MaxPool实现多尺度特征提取，相比SPP减少了计算量：
$$\text{SPPF}(X) = \text{Concat}[X, \text{MaxPool}_5(X), \text{MaxPool}_5^2(X), \text{MaxPool}_5^3(X)]$$

NPU实现要点：
- MaxPool可以流水线执行，减少中间结果存储
- Concat在channel维度，适合分块处理
- 池化操作memory-bound，需要优化内存访问模式

**SPPF的硬件加速策略**：

串行池化的流水线设计大幅降低了内存占用：
```
Input Feature Map (20×20×1024)
        │
        ├─────────────────────────────────┐ (直通)
        │                                 │
        ▼                                 │
   MaxPool_5×5 ──────────────┐           │
        │                     │           │
        ▼                     │           │
   MaxPool_5×5 ────┐         │           │
        │           │         │           │
        ▼           ▼         ▼           ▼
   MaxPool_5×5 → Concat → Concat → Concat → Output
                                   (20×20×4096)

内存需求对比：
- SPP (并行): 4×20×20×1024×2 = 3.2MB
- SPPF (串行): 1×20×20×1024×2 = 0.8MB
- 节省75%片上存储
```

每个MaxPool阶段的感受野递增：
- Stage 1: 5×5 感受野
- Stage 2: 9×9 感受野 (5+4)
- Stage 3: 13×13 感受野 (5+4+4)

这种多尺度设计特别适合检测不同大小的目标，小目标主要响应第一级池化，大目标受益于更大的感受野。

#### CenterNet的中心点检测机制

CenterNet代表了目标检测的另一种范式：将检测问题转化为中心点估计问题。这种方法从根本上改变了计算模式，为NPU优化提供了新的机会。与YOLO的密集预测不同，CenterNet专注于稀疏的关键点，这种稀疏性可以被硬件充分利用。

**架构设计的硬件考量：**

CenterNet的设计理念与传统密集检测器形成鲜明对比。在自动驾驶场景中，一帧图像通常包含10-50个目标，相比于YOLO产生的数千个候选框，CenterNet直接定位这几十个中心点，大幅减少了后处理开销。这种稀疏性在NPU设计中可以通过以下方式利用：

1. **稀疏激活压缩**：热图中>99%的位置为背景，可使用稀疏编码减少片外带宽
2. **动态计算分配**：根据检测到的峰值数量动态分配计算资源
3. **早期退出机制**：当检测到足够数量的高置信度目标时提前终止

**稀疏性的量化分析**：

以640×640输入、下采样率4为例：
```
热图尺寸: 160×160×80 (80个类别)
总位置数: 25,600
典型目标数: 20-30
稀疏度: 1 - 30/25600 = 99.88%

内存占用对比:
密集表示: 160×160×80×4 = 8.2MB (fp32)
稀疏表示: 30×(2+1+80×4) ≈ 10KB
压缩率: 820:1
```

这种极端的稀疏性使得CenterNet特别适合部署在边缘设备上。通过稀疏编码，整个检测结果可以fit in L2 cache，避免了频繁的DRAM访问。

**核心思想：Objects as Points**

CenterNet将每个目标表示为其边界框的中心点，检测过程分为三步：
1. 生成中心点热图（Heatmap）- 识别目标位置
2. 预测中心点的局部偏移（Local Offset）- 亚像素精度校正
3. 回归目标尺寸（Size Regression）- 确定边界框大小

**热图生成的数学原理：**

对于类别 $c$ 的目标中心点 $(\tilde{x}, \tilde{y})$，在热图上渲染高斯核：
$$Y_{xyc} = \exp\left(-\frac{(x-\tilde{x})^2 + (y-\tilde{y})^2}{2\sigma_p^2}\right)$$

其中 $\sigma_p$ 与目标尺寸成正比，确保大目标有更大的响应区域：
$$\sigma_p = \max\left(1, \frac{1}{3}\sqrt{wh}\right)$$

这种自适应的标准差设计平衡了定位精度和训练稳定性。

**高斯核的计算优化**：

实际实现时，高斯核可以预计算并存储为查找表：

```
对于不同尺寸的目标，σ的典型值：
- 行人 (40×100 pixels): σ ≈ 2.1
- 轿车 (150×300 pixels): σ ≈ 7.9  
- 卡车 (200×500 pixels): σ ≈ 14.9

预计算策略：
1. 离散化σ为16个等级
2. 每个等级存储7×7的高斯模板
3. 总存储: 16×7×7×4 = 3.1KB
4. 渲染时直接查表，避免exp计算
```

这种查表方法将每个高斯核的渲染从49次exp运算降低为49次内存读取，在NPU上可以获得10倍以上的加速。同时，固定大小的模板也便于硬件流水线设计。

**损失函数设计：**

CenterNet使用改进的Focal Loss处理类别不平衡：
$$L_{\text{heatmap}} = -\frac{1}{N}\sum_{xyc}\begin{cases}
(1-\hat{Y}_{xyc})^\alpha \log(\hat{Y}_{xyc}) & \text{if } Y_{xyc}=1 \\
(1-Y_{xyc})^\beta \hat{Y}_{xyc}^\alpha \log(1-\hat{Y}_{xyc}) & \text{otherwise}
\end{cases}$$

其中 $\alpha=2$, $\beta=4$ 用于平衡正负样本。

**偏移量预测的必要性：**

由于输出stride（通常为4），从特征图映射回原图会产生量化误差：
$$\Delta_x = \tilde{x} - \lfloor\frac{\tilde{x}}{n}\rfloor \times n, \quad \Delta_y = \tilde{y} - \lfloor\frac{\tilde{y}}{n}\rfloor \times n$$

这个偏移通过独立的回归头预测，使用L1 loss：
$$L_{\text{offset}} = \frac{1}{N}\sum_{i=1}^{N}|\hat{O}_i - O_i|$$

**计算优势与NPU映射：**

1. **无需NMS后处理**：
   - 传统方法：需要串行的NMS操作，难以并行化
   - CenterNet：直接提取local maxima，高度并行
   - NPU优化：使用3×3 MaxPool实现峰值检测
   
   **峰值检测的并行实现**：
   ```
   Heat Map (160×160×80)
           │
           ▼
   MaxPool_3×3 (same padding)
           │
           ▼
   Element-wise Compare with Original
           │
           ▼
   Threshold (>0.3)
           │
           ▼
   Extract Coordinates
   
   并行度分析：
   - MaxPool: 完全并行，O(1)时间复杂度
   - Compare: SIMD并行，25,600个位置同时比较
   - Extract: 稀疏gather操作
   
   相比NMS的O(N²)复杂度，提速100倍以上
   ```

2. **稀疏激活利用**：
   - 热图典型稀疏度 >99%
   - 可使用稀疏卷积加速后续处理
   - 激活压缩率高，减少片外带宽

3. **多任务头部设计**：
   ```
   Backbone Features (C channels)
           ↓
      Conv 3×3 (256 channels)
           ↓
      ┌────┴────┬────────┬──────────┐
   Heatmap   Offset    Size      3D Extension
   (K cls)   (2 ch)   (2 ch)     (optional)
   ```
   
   所有头部共享特征，提高数据重用率。

**CenterNet与YOLO的计算对比：**

| 特性 | YOLO | CenterNet |
|------|------|-----------|
| 预测密度 | 密集（每个grid cell） | 稀疏（仅中心点） |
| 后处理 | NMS（串行） | Local Maxima（并行） |
| 内存占用 | $O(S^2 \times (B \times 5 + C))$ | $O(S^2 \times K)$ |
| 计算分布 | 均匀 | 集中在关键点 |

### 2.1.2 3D检测网络：PointPillars与CenterPoint

3D点云检测是自动驾驶感知的核心任务，直接处理激光雷达数据获取精确的3D边界框。点云的稀疏性、不规则性和大规模性给NPU设计带来独特挑战，需要专门的架构创新来高效处理这类数据。

#### PointPillars的柱状体编码

PointPillars创新性地将不规则点云转换为规则的伪图像表示，使得成熟的2D卷积技术可以直接应用。这种设计哲学特别适合NPU架构，因为它将稀疏的3D问题转化为密集的2D问题。

**算法动机与硬件友好性：**

传统的点云处理方法如PointNet++需要复杂的采样和聚合操作，难以在固定硬件架构上高效实现。PointPillars的关键洞察是：自动驾驶场景中的目标主要分布在地面上，高度信息相对次要。通过将点云投影到BEV平面并保留高度作为特征，可以复用成熟的2D CNN加速器。

这种设计带来多重优势：
- **规则内存访问**：柱状体网格化后形成规则的2D tensor，避免了不规则内存访问
- **批处理友好**：所有pillars可以打包成固定大小的batch，充分利用SIMD/SIMT并行
- **数据局部性**：相邻pillars在物理空间上也相邻，有利于缓存预取

**与传统方法的性能对比**：

```
方法对比           PointNet++        PointPillars
────────────────────────────────────────────────
输入格式           不规则点集         规则Pillar Grid
内存访问模式       随机访问          顺序访问
并行化难度         高(FPS/采样)      低(规则tensor)
硬件利用率         ~30%             ~85%
推理延迟(ms)       67               16
内存带宽需求       不可预测          可精确计算
缓存命中率         <40%             >90%
```

这种从不规则到规则的转换是深度学习系统设计的经典范式：通过引入少量的信息损失（pillar内点的顺序信息），换取数量级的性能提升。

**点云预处理流程：**

1. **空间划分**：将3D空间划分为规则网格
   - X-Y平面划分：$[-75.2, 75.2]m \times [-75.2, 75.2]m$
   - Pillar尺寸：$0.16m \times 0.16m$（对应激光雷达0.1°角分辨率在50m处的投影）
   - 网格数量：$940 \times 940 = 883,600$ pillars
   - Z方向不划分，保留完整高度信息（-3m到1m，覆盖车辆高度范围）

2. **Pillar构建**：每个非空pillar最多保留 $N=100$ 个点
   ```
   for each pillar (x_p, y_p):
       points = find_points_in_pillar(x_p, y_p)
       if len(points) > N:
           points = random_sample(points, N)  # 随机采样保持代表性
       elif len(points) < N:
           points = pad_with_zeros(points, N)  # 零填充保持tensor规则
   ```
   
   **采样策略的影响**：
   - 随机采样vs最远点采样：随机采样硬件实现简单，最远点采样需要距离计算
   - 动态N vs固定N：固定N=100简化硬件设计，但可能浪费计算资源
   - 实践中，95%的pillars包含<30个点，可考虑分级处理
   
   **Pillar密度的统计分布**：
   ```
   距离范围     平均点数/pillar   稀疏度    优化策略
   ─────────────────────────────────────────────
   0-10m       45.2            5%       完整处理
   10-20m      18.7            15%      标准处理
   20-40m      5.3             60%      稀疏处理
   40-60m      1.8             85%      极稀疏处理
   60m+        0.4             95%      跳过或简化
   
   分级处理策略:
   if (distance < 20m) {
       N_max = 100;  // 完整容量
       use_full_pointnet();
   } else if (distance < 40m) {
       N_max = 32;   // 降低容量
       use_lite_pointnet();
   } else {
       N_max = 8;    // 最小容量
       use_micro_pointnet();
   }
   ```
   
   这种自适应处理可以节省70%的PointNet计算量，同时保持检测精度。

**增强的Pillar特征编码：**

每个点的9维特征向量：
$$f_{i} = [x_i, y_i, z_i, r_i, x_i - x_c, y_i - y_c, z_i - z_c, x_i - x_p, y_i - y_p]$$

其中：
- $(x_i, y_i, z_i)$：点的绝对坐标
- $r_i$：反射强度
- $(x_c, y_c, z_c)$：pillar内所有点的质心
- $(x_p, y_p)$：pillar的中心坐标

这种特征设计编码了：
- 全局位置信息（绝对坐标）
- 局部几何结构（相对质心）
- Pillar上下文（相对pillar中心）

**PointNet处理层：**

使用简化的PointNet对每个pillar内的点进行特征提取：
$$g_i = \text{BN}(\text{Linear}(f_i)) \in \mathbb{R}^{C}$$
$$h = \max_{i \in \text{pillar}} g_i \in \mathbb{R}^{C}$$

其中max pooling实现置换不变性。

**稀疏性分析与优化：**

典型城市场景的稀疏性统计：
$$\text{Sparsity} = 1 - \frac{N_{\text{non-empty}}}{N_{\text{total}}} \approx 0.92-0.97$$

稀疏性分布特征：
- 近距离（<20m）：稀疏度 ~70%
- 中距离（20-40m）：稀疏度 ~90%
- 远距离（>40m）：稀疏度 >95%

**稀疏性的物理意义与利用**：

激光雷达点云的稀疏性来源于两个物理因素：
1. **角分辨率固定**：64线激光雷达坐0.1°角分辨率下，远距离点间距增大
2. **遮挡效应**：大部分区域被地面、建筑物遮挡，无有效反射

```
点云密度分布图 (俯视图)

     近距离区域                中距离区域              远距离区域
   ┌───────────┐           ┌───────────┐         ┌───────────┐
   │███████████│           │░░░█░░█░░░░│         │····█······│
   │███████████│           │░████████░░│         │··█████····│
   │████ ██████│           │░░░█░░█░░░░│         │···█··█····│
   └───────────┘           └───────────┘         └───────────┘
  ~5000 pts/m²             ~500 pts/m²            ~50 pts/m²
  30% 空白pillars          90% 空白pillars        95% 空白pillars
```

这种稀疏性分布启发了分层处理策略：近距离区域使用密集计算获得高精度，远距离区域使用稀疏计算节省资源。这种策略与人类视觉系统的中央凹（fovea）设计类似。

**NPU优化策略：**

1. **动态批处理**：
   ```
   active_pillars = get_non_empty_pillars()
   batched_features = pointnet(active_pillars)
   scatter_to_bev(batched_features, indices)
   ```
   仅处理非空pillars，计算量降低90%+
   
   **硬件实现细节**：
   ```
   Pillar批处理流水线:
   
   Stage 1: Index Generation (索引生成)
   │
   ├──► 遍历所有pillars
   ├──► 统计点数 > 0的pillars
   └──► 生成active_mask
   
   Stage 2: Data Gathering (数据收集)
   │
   ├──► 根据active_mask收集点云
   ├──► 打包成固定batch (e.g., 12000 pillars)
   └──► 填充到N=100或截断
   
   Stage 3: PointNet Processing (特征提取)
   │
   ├──► Linear(9 → 64) + BN + ReLU
   ├──► Linear(64 → 128) + BN + ReLU  
   └──► MaxPool over points → (12000, 128)
   
   Stage 4: Scatter Back (分散回写)
   │
   └──► 根据原始位置写回BEV grid
   
   内存带宽需求:
   - Gather: 12000 × 100 × 9 × 4B = 43.2MB
   - Process: 在片上SRAM中完成
   - Scatter: 12000 × 128 × 4B = 6.1MB
   - 总计: 49.3MB (原始密集: 940×940×128×4B = 451MB)
   ```

2. **稀疏卷积实现**：
   - 使用CSR格式存储稀疏特征图
   - Rulebook生成：预计算卷积核的有效位置
   - Gather-GEMM-Scatter模式执行

3. **混合精度策略**：
   - PointNet层：FP16（特征提取）
   - BEV backbone：INT8（2D卷积）
   - Detection head：FP16（精确定位）

#### CenterPoint的多尺度特征聚合

CenterPoint将CenterNet的思想扩展到3D空间，通过在BEV视角下检测目标中心实现高效的3D检测。其核心创新在于使用3D稀疏卷积处理体素化点云，然后在BEV空间进行中心点检测。

**三阶段处理流程：**

```
    Raw Points (~100K points)
        ↓
    [Stage 1: Voxelization]
    Voxel Grid (40000×1600×40)
        ↓
    [Stage 2: 3D Sparse CNN]
    3D Features → BEV Compression
        ↓
    [Stage 3: 2D Detection]
    Center Heatmap + 3D Attributes
```

**体素化与3D稀疏卷积：**

1. **动态体素化**：
   $$V_{ijk} = \{p | \lfloor\frac{p_x}{\Delta_x}\rfloor=i, \lfloor\frac{p_y}{\Delta_y}\rfloor=j, \lfloor\frac{p_z}{\Delta_z}\rfloor=k\}$$
   
   典型参数：
   - 体素大小：$[0.075, 0.075, 0.2]m$
   - 范围：$[-54, 54]m \times [-54, 54]m \times [-5, 3]m$
   - 体素网格：$1440 \times 1440 \times 40$
   
   **体素大小的权衡**：
   ```
   参数选择对性能的影响:
   
   体素尺寸    体素总数      稀疏度    精度    计算量
   ─────────────────────────────────────────────
   0.05m      3.2×10⁸       99.5%   高      极高
   0.075m     8.3×10⁷       98.8%   中高    高      ← 选择
   0.1m       3.7×10⁷       97.5%   中      中
   0.15m      1.1×10⁷       95.0%   低      低
   
   0.075m的选择理由:
   - 匹配64线激光雷达在0.2°垂直分辨率
   - 在10m处约有3.5cm的点间距
   - 平衡精度和计算效率
   ```

2. **3D稀疏卷积网络**：
   ```
   SubMConv3d(16) → SubMConv3d(16)
          ↓
   SparseConv3d(32, stride=2) → SubMConv3d(32) × 2
          ↓
   SparseConv3d(64, stride=2) → SubMConv3d(64) × 2
          ↓
   SparseConv3d(128, stride=2) → SubMConv3d(128) × 2
   ```
   
   SubMConv3d保持稀疏性，SparseConv3d允许稀疏性变化。

**稀疏卷积的高效实现：**

传统密集3D卷积的计算复杂度：
$$\text{FLOPs}_{\text{dense}} = K^3 \times C_{in} \times C_{out} \times D \times H \times W$$

稀疏卷积通过Rulebook优化：
$$\text{FLOPs}_{\text{sparse}} = \sum_{(i,o) \in \text{Rulebook}} K^3 \times C_{in} \times C_{out}$$

实际加速比：
$$\text{Speedup} = \frac{1}{(1-\text{Sparsity})^2 \times \alpha}$$

其中 $\alpha \approx 1.2$ 为索引开销系数。对于95%稀疏度：
$$\text{Speedup} = \frac{1}{0.05^2 \times 1.2} \approx 333×$$

**Rulebook生成算法**：

```
Rulebook是稀疏卷积的核心数据结构，记录了每个卷积核位置的输入输出映射：

传统3D卷积 (3×3×3 kernel):
for each output_voxel (x,y,z):
    for kx in [-1,0,1]:
        for ky in [-1,0,1]:
            for kz in [-1,0,1]:
                input_voxel = (x+kx, y+ky, z+kz)
                if exists(input_voxel):
                    compute MAC
                    
Submanifold稀疏卷积 (SubMConv3d):
- 输出仅在输入非空位置产生
- 保持稀疏模式不变
- 适用于残差块

Rulebook结构:
[
  (input_idx, output_idx, kernel_offset),
  (12, 45, 13),  // voxel_12 → voxel_45, kernel[1,1,3]
  (13, 45, 14),  // voxel_13 → voxel_45, kernel[1,1,4]
  ...
]

实际执行:
1. 根据Rulebook gather输入特征
2. 执行GEMM: [N_pairs, C_in] × [C_in, C_out]
3. Scatter回输出位置
```

这种方法将不规则的3D卷积转化为规则的GEMM操作，非常适合脉动阵列加速。

**BEV特征压缩：**

将3D特征压缩到BEV平面：
$$F_{\text{BEV}}(x,y) = \text{Concat}_{z}[F_{3D}(x,y,z)]$$

或使用加权聚合：
$$F_{\text{BEV}}(x,y) = \sum_{z} w_z \cdot F_{3D}(x,y,z)$$

**多任务检测头：**

CenterPoint的检测头预测多个任务：

1. **中心热图**：$H \in \mathbb{R}^{W \times H \times K}$（K个类别）
2. **中心偏移**：$O \in \mathbb{R}^{W \times H \times 2}$（亚像素精度）
3. **3D尺寸**：$S \in \mathbb{R}^{W \times H \times 3}$（长宽高）
4. **旋转角度**：$R \in \mathbb{R}^{W \times H \times 2}$（sin, cos编码）
5. **速度估计**：$V \in \mathbb{R}^{W \times H \times 2}$（可选，用于跟踪）

**两阶段精炼（可选）：**

第二阶段使用RoI特征精炼预测：
$$\text{RoI Features} = \text{RoIAlign}(F_{\text{BEV}}, \text{Proposals})$$
$$\Delta = \text{MLP}(\text{RoI Features})$$

精炼带来~2-3% AP提升，但增加20%延迟。

**计算复杂度分析：**

| 组件 | FLOPs | 占比 |
|------|-------|------|
| 3D Sparse CNN | 5.2G | 45% |
| BEV Backbone | 4.8G | 42% |
| Detection Heads | 1.5G | 13% |
| 总计 | 11.5G | 100% |

相比PointPillars（~63G），CenterPoint通过稀疏卷积实现5×加速。

### 2.1.3 BEV感知网络：BEVFormer与BEVDet

#### BEVFormer的时空注意力机制

BEVFormer通过可学习的BEV queries与多视角图像特征交互，实现2D到3D的特征转换。

**空间交叉注意力（SCA）：**
$$\text{SCA}(Q_p, F_t) = \sum_{i=1}^{N_{\text{ref}}} \text{DeformAttn}(Q_p, P(p, i, j), F_t^i)$$

其中：
- $Q_p$: BEV位置 $p$ 的query
- $F_t^i$: 第 $i$ 个相机在时刻 $t$ 的特征
- $P(p, i, j)$: 3D到2D的投影函数

**时序自注意力（TSA）：**
$$\text{TSA}(Q_t, B_{t-1}) = \text{DeformAttn}(Q_t, Q_t + \Delta, B_{t-1})$$

这里 $\Delta$ 编码了自车运动补偿。

**计算开销分解：**
- SCA: $O(N_{\text{BEV}} \times N_{\text{cam}} \times N_{\text{ref}} \times D^2)$
- TSA: $O(N_{\text{BEV}} \times N_{\text{temporal}} \times D^2)$
- 总计: 约30 GFLOPs用于6相机输入

#### BEVDet的深度估计策略

BEVDet通过显式深度估计构建BEV特征：

**深度分布预测：**
$$D(u,v) = \text{Softmax}(\text{Conv}(F_{2D}(u,v))) \in \mathbb{R}^{D_{\text{bins}}}$$

**Lift-Splat变换：**
$$F_{\text{BEV}}(x,y) = \sum_{c=1}^{N_{\text{cam}}} \sum_{d=1}^{D_{\text{bins}}} F_{2D}^c(\pi_c(x,y,d)) \times D^c(\pi_c(x,y,d))$$

其中 $\pi_c$ 为相机 $c$ 的投影矩阵。

### 2.1.4 轨迹预测与规划网络

#### 基于Transformer的轨迹预测

现代轨迹预测网络（如Wayformer）采用场景中心（scene-centric）表示：

**场景表示的演进**：

从早期的独立轨迹预测到现代的交互式预测，架构设计经历了根本性变革：

```
第一代: 独立预测 (2015-2018)
┌─────────────────────────┐
│ LSTM/GRU 单独处理每个Agent │
│ 无交互建模                 │
│ O(N) 复杂度                  │
└─────────────────────────┘
            ↓
第二代: 图网络交互 (2018-2020)
┌─────────────────────────┐
│ GNN/GraphRNN 建模Agent间关系│
│ 有限交互范围                │
│ O(N×E) 复杂度               │
└─────────────────────────┘
            ↓
第三代: Transformer全局交互 (2020-)
┌─────────────────────────┐
│ Self-Attention 全局交互    │
│ 地图信息融合                 │
│ O(N²) 但更准确              │
└─────────────────────────┘
```

Transformer架构在轨迹预测中的优势在于其能够捕捉复杂的多体交互模式，特别是在交叉路口、并线等高交互场景。

**Agent-Scene交互建模：**
$$h_i^{(l+1)} = h_i^{(l)} + \text{MHA}([h_i^{(l)}, E_{\text{pos}}^i], [H^{(l)}, E_{\text{pos}}])$$

其中：
- $h_i$: agent $i$ 的隐状态
- $E_{\text{pos}}$: 位置编码
- $H$: 所有agents和地图元素的特征集合

**多模态轨迹生成：**
$$P(Y|X) = \sum_{k=1}^{K} w_k \cdot \mathcal{N}(\mu_k(X), \Sigma_k(X))$$

典型设置 $K=6$ 个模态，每个模态预测 $T=80$ 个时间步（8秒）。

**多模态的物理意义**：

不同模态对应不同的驾驶意图：
```
6种典型驾驶模态:
1. 直行保持 (33%) - 继续当前车道
2. 左转待行 (15%) - 等待对向车流
3. 左转通过 (12%) - 立即左转
4. 右转让行 (10%) - 避让行人
5. 变道左 (8%)   - 超车或避障
6. 变道右 (7%)   - 回到慢车道
其他 (15%)      - 停车、掉头等

模态权重学习:
w_k = Softmax(MLP(scene_features))
根据场景特征动态调整各模态概率
```

这种多模态设计对NPU的挑战在于需要并行计算6个不同的轨迹假设，每个假设都需要完整的前向传播。通过批处理和权重共享，可以将计算开销从6倍降低到2-3倍。

**计算需求：**
- Attention计算: $O(N^2 \times T \times D)$, 其中 $N \approx 200$ (agents + map)
- MLP解码: $O(K \times N \times T \times D^2)$
- 总计: 约5 GFLOPs per scene

## 2.2 VLM/VLA工作负载特征

视觉语言模型（VLM）和视觉语言动作模型（VLA）代表了多模态AI的前沿，其计算特征与传统CV网络有显著差异。

### 2.2.1 CLIP与对比学习的计算需求

#### CLIP的双塔架构

CLIP通过对比学习对齐视觉和文本表示，其架构设计充分考虑了大规模训练的效率需求：

**架构设计哲学：**

CLIP采用双塔架构而非融合架构的关键原因在于计算效率。在对比学习中，每个batch需要计算所有图像-文本对的相似度，双塔架构允许独立编码后仅需一次矩阵乘法，而融合架构需要 $B^2$ 次前向传播。这种设计在NPU上的优势包括：

1. **独立并行处理**：视觉和文本编码器可以部署在不同的计算单元上
2. **特征缓存复用**：编码后的特征可以缓存用于多次相似度计算
3. **灵活的批处理**：图像和文本可以使用不同的batch size优化吞吐量

**视觉编码器（ViT-L/14）：**
$$Z_v = \text{ViT}(I) \in \mathbb{R}^{B \times D}$$

计算量分解：
$$\text{FLOPs}_{\text{ViT}} = 2 \times N \times (D \times D_{mlp} + N \times D^2/H)$$

其中：
- $N = (224/14)^2 = 256$ patches（14×14的patch划分）
- $D = 1024$ 隐藏维度（比ResNet的2048维更硬件友好）
- $D_{mlp} = 4 \times D = 4096$（FFN扩展率）
- $H = 16$ 注意力头（每头64维，适合SIMD宽度）

总计约 81 GFLOPs。

**Patch Embedding的优化：**

将图像分割为patches的过程可以通过不同方式实现：
$$\text{Patch}(I) = \text{Reshape}(\text{Conv}_{14 \times 14, \text{stride}=14}(I))$$

这个strided convolution在NPU上可以映射为：
- 内存重排操作（无计算）+ 矩阵乘法
- 或直接的大核卷积（需要专门的大核支持）

**文本编码器（Transformer）：**
$$Z_t = \text{TextEncoder}(T) \in \mathbb{R}^{B \times D}$$

计算量相对较小（约 6 GFLOPs for 77 tokens）。

**对比损失计算：**
$$\mathcal{L} = -\frac{1}{2B}\sum_{i=1}^{B}\left[\log\frac{e^{z_v^i \cdot z_t^i / \tau}}{\sum_{j=1}^{B} e^{z_v^i \cdot z_t^j / \tau}} + \log\frac{e^{z_t^i \cdot z_v^i / \tau}}{\sum_{j=1}^{B} e^{z_t^i \cdot z_v^j / \tau}}\right]$$

批量矩阵乘法需求：$Z_v Z_t^T \in \mathbb{R}^{B \times B}$

### 2.2.2 LLaVA与Flamingo的多模态架构

#### LLaVA的简单投影策略

LLaVA通过线性投影连接视觉编码器和语言模型：

**视觉特征投影：**
$$H_v = W \cdot Z_v, \quad W \in \mathbb{R}^{D_{llm} \times D_{vision}}$$

**多模态序列构建：**
$$X = [\text{System}, H_v, \text{User}, \text{Assistant}]$$

**计算分布：**
- Vision Encoder: 5.6 GFLOPs (CLIP ViT-L/14, 336×336)
- Projection: 0.01 GFLOPs
- LLM (7B): 14 GFLOPs per token
- 总推理: 约 20 GFLOPs for 100 tokens output

#### Flamingo的Perceiver Resampler

Flamingo通过Perceiver架构处理可变长度视觉输入：

**Perceiver交叉注意力：**
$$Q_{\text{latent}} \in \mathbb{R}^{N_q \times D}, \quad K_V = \phi(X_{\text{visual}}) \in \mathbb{R}^{N_v \times D}$$

其中 $N_q = 64$ 固定查询数，$N_v$ 可变。

**Gated Cross-Attention到LLM：**
$$h' = h + \tanh(\alpha) \cdot \text{CrossAttn}(h, Z_{\text{visual}})$$

门控机制 $\alpha$ 初始化为0，实现渐进式适应。

### 2.2.3 RT-1/RT-2机器人控制模型

#### RT-1的实时控制架构

RT-1 (Robotics Transformer 1) 将机器人控制建模为序列决策问题：

**输入表示：**
$$X_t = [\text{Image}_t, \text{Language}, \text{State}_t]$$

其中：
- $\text{Image}_t \in \mathbb{R}^{300 \times 300 \times 3}$: RGB观察
- $\text{Language}$: 自然语言指令的embedding
- $\text{State}_t \in \mathbb{R}^{8}$: 关节角度和夹爪状态

**动作tokenization：**
将连续动作离散化为256个bins：
$$a_t = [\Delta x, \Delta y, \Delta z, \Delta\text{yaw}, \Delta\text{pitch}, \Delta\text{roll}, \text{gripper}]$$

**计算需求：**
- Vision: EfficientNet-B3, 1.8 GFLOPs
- Transformer: 8层, 2.5 GFLOPs  
- 动作解码: 0.1 GFLOPs
- 总延迟要求: < 100ms (10Hz控制)

#### RT-2的视觉-语言-动作统一

RT-2将预训练VLM适配为VLA，实现知识迁移：

**统一tokenization：**
$$\text{Vocab} = \text{Vocab}_{\text{language}} \cup \text{Vocab}_{\text{action}}$$

动作tokens作为特殊词汇：`<move_x_+10>`, `<rotate_z_-5>`等。

**Co-fine-tuning策略：**
$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{language}} + \lambda_2 \mathcal{L}_{\text{action}} + \lambda_3 \mathcal{L}_{\text{vision}}$$

保持多任务能力的同时学习控制。

**推理模式切换：**
- 语言生成: beam search, $B=4$
- 动作生成: greedy decoding, $B=1$
- 混合模式: 先生成语言reasoning，后生成动作

## 2.3 算子级性能分析

深入理解核心算子的计算特征是NPU优化的基础。本节量化分析GEMM、卷积、注意力等算子的性能特征。

### 2.3.1 GEMM的计算密度分析

#### 标准GEMM的算术强度

对于 $C = A \times B$，其中 $A \in \mathbb{R}^{M \times K}$, $B \in \mathbb{R}^{K \times N}$：

**计算量：**
$$\text{FLOPs} = 2MNK$$

**内存访问量（无重用）：**
$$\text{Memory} = (MK + KN + MN) \times \text{sizeof(dtype)}$$

**算术强度（Arithmetic Intensity）：**
$$AI = \frac{2MNK}{(MK + KN + MN) \times \text{sizeof(dtype)}}$$

对于方阵 $M=N=K$：
$$AI = \frac{2K}{3 \times \text{sizeof(dtype)}}$$

**nvfp4 (E2M1)影响：**
- sizeof(nvfp4) = 0.5 bytes
- $AI_{\text{nvfp4}} = 2 \times AI_{\text{fp8}}$
- 理论峰值提升，但精度损失需权衡

#### Batch GEMM的优化机会

批量矩阵乘法 $C_i = A_i \times B_i$ for $i \in [1, B]$：

**数据重用分析：**

1. **Weight-stationary (B相同)**：
   $$\text{Reuse}_B = B$$
   适用于全连接层推理

2. **Input-stationary (A相同)**：
   $$\text{Reuse}_A = B$$
   适用于多头注意力的QKV投影

3. **Output-stationary**：
   部分和累加，减少输出带宽

### 2.3.2 Conv2D的内存访问模式

#### Im2col变换分析

Im2col将卷积转换为GEMM：

**展开后矩阵大小：**
$$\text{Im2col}: \mathbb{R}^{C_{in} \times H \times W} \to \mathbb{R}^{(C_{in} \times K_h \times K_w) \times (H_{out} \times W_{out})}$$

**内存膨胀率：**
$$\text{Expansion} = K_h \times K_w$$

对于3×3卷积，数据量增加9倍。

#### Direct Convolution的滑窗策略

直接卷积避免内存膨胀：

**计算循环嵌套：**
```
for n in [0, N):          # Batch
  for c_out in [0, C_out): # Output channels  
    for h in [0, H_out):   # Height
      for w in [0, W_out): # Width
        for c_in in [0, C_in):    # Input channels
          for kh in [0, K_h):     # Kernel height
            for kw in [0, K_w]:   # Kernel width
              out[n,c_out,h,w] += 
                in[n,c_in,h+kh,w+kw] * weight[c_out,c_in,kh,kw]
```

**Loop tiling优化：**
将循环分块以适配片上存储：
$$T_h \times T_w \times C_{in} \times \text{sizeof(dtype)} \leq \text{SRAM}_{\text{size}}$$

### 2.3.3 Attention的计算瓶颈

#### 标准Attention的二次复杂度

Self-attention计算：
$$\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

**计算复杂度：**
- QK矩阵乘: $O(N^2 \times d)$
- Softmax: $O(N^2)$
- Score×V: $O(N^2 \times d)$
- 总计: $O(N^2 \times d)$

**内存需求：**
存储attention矩阵需要 $N^2 \times \text{sizeof(dtype)}$ 字节。

对于序列长度 $N=2048$, fp16精度：
$$\text{Memory} = 2048^2 \times 2 = 8\text{MB}$$

#### Flash Attention的分块计算

Flash Attention通过分块降低内存访问：

**分块策略：**
将 $Q,K,V \in \mathbb{R}^{N \times d}$ 分为大小为 $B_r \times d$ 和 $B_c \times d$ 的块。

**Online softmax：**
$$m^{(j+1)} = \max(m^{(j)}, \tilde{m}^{(j)})$$
$$l^{(j+1)} = e^{m^{(j)} - m^{(j+1)}} l^{(j)} + e^{\tilde{m}^{(j)} - m^{(j+1)}} \tilde{l}^{(j)}$$

**IO复杂度降低：**
$$\text{IO}_{\text{Flash}} = O\left(\frac{N^2d}{M^{1/2}}\right)$$

其中 $M$ 为SRAM大小。相比标准attention的 $O(N^2d)$，显著降低。

### 2.3.4 Memory-bound vs Compute-bound分析

#### Roofline模型应用

性能上界由计算能力和内存带宽共同决定：

$$\text{Performance} = \min(\text{Peak\_FLOPS}, AI \times \text{Bandwidth})$$

**分界点：**
$$AI_{\text{ridge}} = \frac{\text{Peak\_FLOPS}}{\text{Bandwidth}}$$

对于200 TOPS NPU with 1TB/s带宽：
$$AI_{\text{ridge}} = \frac{200 \times 10^{12}}{1 \times 10^{12}} = 200 \text{ ops/byte}$$

#### 典型算子分类

**Compute-bound算子：**
- 大矩阵GEMM: $AI > 100$
- 1×1卷积with大通道数: $AI \approx C_{in}$
- 标准3×3卷积: $AI \approx 50-100$

**Memory-bound算子：**
- Element-wise操作: $AI < 1$
- Pooling层: $AI \approx 0.5$
- Normalization: $AI \approx 2$
- 小矩阵GEMM: $AI < 50$

#### 2:4稀疏对性能的影响

结构化稀疏改变计算密度：

**有效算术强度：**
$$AI_{\text{sparse}} = \frac{AI_{\text{dense}}}{2} \times \frac{1}{0.75}$$

压缩率50%，但索引开销25%。

**稀疏加速条件：**
$$\text{Speedup} > 1 \iff AI_{\text{dense}} > 2 \times AI_{\text{ridge}}$$

仅对compute-bound算子有效。

### 2.3.5 算子融合机会识别

#### 垂直融合（Producer-Consumer）

连续算子融合减少中间结果的内存访问：

**Conv-BN-ReLU融合：**
原始计算：
1. $Y = \text{Conv}(X, W) + b$
2. $Z = \gamma \frac{Y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$
3. $A = \max(0, Z)$

融合后：
$$A = \max\left(0, \gamma \frac{\text{Conv}(X, W) + b - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta\right)$$

**内存节省：**
- 原始: 3次feature map读写
- 融合: 1次读写
- 带宽降低: 67%

#### 水平融合（并行算子）

并行执行多个独立算子：

**多头注意力的QKV投影：**
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

融合为单个GEMM：
$$[Q, K, V] = X[W_Q, W_K, W_V]$$

**优势：**
- 输入数据重用3倍
- 更好的矩阵维度for tiling
- 减少kernel启动开销

#### 图级优化机会

**Transformer块的端到端融合：**
```
Input → LayerNorm → Multi-Head Attention → Residual Add
      → LayerNorm → FFN → Residual Add → Output
```

识别融合pattern：
1. Pre-norm与attention融合
2. Attention与post-processing融合  
3. FFN的GeLU激活内嵌
4. Residual连接的就地计算

**量化收益：**
- 减少50%的激活值量化/反量化
- 避免中间结果的精度损失
- 整体延迟降低20-30%

## 本章小结

本章系统分析了自动驾驶和具身智能场景下的算法工作负载特征，建立了从网络架构到算子实现的完整性能模型。

**核心要点：**

1. **自动驾驶网络特征**：
   - 2D检测: 计算密集，规则数据流，适合脉动阵列
   - 3D检测: 高度稀疏，需要专门的稀疏计算单元
   - BEV感知: 多视角融合，attention计算占主导
   - 轨迹预测: 长序列推理，需要高效的时序建模

2. **VLM/VLA计算需求**：
   - 视觉编码器: ViT的规则计算pattern
   - 多模态融合: 交叉注意力的带宽需求
   - 实时控制: 严格延迟约束下的精简模型

3. **算子性能分类**：
   - Compute-bound: GEMM、大卷积核，受算力限制
   - Memory-bound: Element-wise、归一化，受带宽限制
   - 混合特征: Attention随序列长度转换

4. **优化机会**：
   - 算子融合: 垂直融合降低带宽，水平融合提高利用率
   - 稀疏化: 2:4结构化稀疏在compute-bound算子效果显著
   - 量化: nvfp4提供2倍理论加速，需要算法协同

**关键公式汇总：**

- 算术强度: $AI = \frac{\text{FLOPs}}{\text{Memory Access}}$
- Roofline性能: $P = \min(\text{Peak}, AI \times BW)$
- 稀疏加速条件: $AI_{\text{dense}} > 2 \times AI_{\text{ridge}}$
- Attention复杂度: $O(N^2 \times d)$ vs Flash: $O(\frac{N^2d}{\sqrt{M}})$

## 练习题

### 基础题

**练习 2.1**: 计算YOLOv8在640×640输入下的总FLOPs，假设backbone采用CSPDarknet，neck采用PANet。列出每个stage的计算量。

<details>
<summary>提示</summary>

使用卷积FLOPs公式：$2 \times C_{in} \times C_{out} \times K^2 \times H_{out} \times W_{out}$

</details>

<details>
<summary>参考答案</summary>

YOLOv8-M的计算量分布：
- Stem: 0.5 GFLOPs
- Stage1 (P1/2): 2.1 GFLOPs  
- Stage2 (P2/4): 4.3 GFLOPs
- Stage3 (P3/8): 9.7 GFLOPs
- Stage4 (P4/16): 12.4 GFLOPs
- Stage5 (P5/32): 8.2 GFLOPs
- Neck (PANet): 15.8 GFLOPs
- Head: 3.5 GFLOPs
- 总计: ~56.5 GFLOPs

</details>

**练习 2.2**: 对于PointPillars，如果点云范围是[-75.2, 75.2]m × [-75.2, 75.2]m，pillar大小0.16m，计算pillar grid尺寸和90%稀疏度下的有效计算量。

<details>
<summary>提示</summary>

Grid尺寸 = Range / Pillar_size，有效FLOPs = 总FLOPs × (1-稀疏度)²

</details>

<details>
<summary>参考答案</summary>

- Grid X: 150.4 / 0.16 = 940
- Grid Y: 150.4 / 0.16 = 940
- 总pillars: 940 × 940 = 883,600
- 非空pillars (10%): ~88,360
- 2D CNN on BEV: 940×940 feature map
- 有效计算: 原始FLOPs × 0.1² = 1% of dense

</details>

**练习 2.3**: 计算矩阵乘法 $C_{[512,768]} = A_{[512,256]} \times B_{[256,768]}$ 的算术强度，假设使用fp16数据类型。这个操作是compute-bound还是memory-bound？（假设AI_ridge=100）

<details>
<summary>提示</summary>

AI = 2MNK / ((MK + KN + MN) × sizeof(fp16))

</details>

<details>
<summary>参考答案</summary>

- FLOPs = 2 × 512 × 768 × 256 = 201,326,592
- Memory = (512×256 + 256×768 + 512×768) × 2 bytes
- Memory = (131,072 + 196,608 + 393,216) × 2 = 1,441,792 bytes
- AI = 201,326,592 / 1,441,792 ≈ 139.6 ops/byte
- 因为 AI > AI_ridge (139.6 > 100)，所以是compute-bound

</details>

### 挑战题

**练习 2.4**: 设计一个算子融合方案，将Transformer的一个完整block（包括Multi-Head Attention和FFN）融合为最少的kernel调用。计算融合前后的内存访问量差异。

<details>
<summary>提示</summary>

考虑哪些操作可以就地计算，哪些中间结果可以保持在片上存储中。

</details>

<details>
<summary>参考答案</summary>

融合方案（3个kernels）：
1. Kernel 1: LayerNorm + QKV投影 + Attention计算
2. Kernel 2: Attention输出投影 + Residual + LayerNorm
3. Kernel 3: FFN (FC1 + GeLU + FC2) + Residual

内存访问分析（seq_len=512, hidden=768）：
- 原始: 14次feature map读写 = 14 × 512 × 768 × 2 = 11MB
- 融合: 5次读写 = 5 × 512 × 768 × 2 = 3.9MB
- 节省: 64%带宽

</details>

**练习 2.5**: 分析Flash Attention在不同序列长度下的性能优势。给定SRAM=96KB，计算seq_len=512, 1024, 2048, 4096时的最优block size和IO降低率。

<details>
<summary>提示</summary>

Block size受SRAM限制：$B_r \times d \times 3 \times \text{sizeof} \leq \text{SRAM}$

</details>

<details>
<summary>参考答案</summary>

最优block size计算（d=64, fp16）：
- 可用SRAM for QKV blocks: 96KB
- 每个block需要: $B_r × 64 × 3 × 2$ bytes
- 最大 $B_r = 96×1024 / (64×3×2) = 256$

IO降低率：
- Seq=512: Standard IO = 512²×64×2×3 = 100MB
  Flash IO = 512²×64×2×3/√(96K/384) = 6.3MB, 降低94%
- Seq=1024: Standard = 402MB, Flash = 25MB, 降低94%
- Seq=2048: Standard = 1.6GB, Flash = 101MB, 降低94%
- Seq=4096: Standard = 6.4GB, Flash = 404MB, 降低94%

</details>

**练习 2.6**: 对于2:4结构化稀疏，推导在什么条件下可以获得实际加速。考虑稀疏索引的存储和解码开销，给出加速比与算术强度的关系式。

<details>
<summary>提示</summary>

考虑稀疏索引占用的额外带宽和解码延迟。

</details>

<details>
<summary>参考答案</summary>

加速比分析：

稀疏计算时间：
$$T_{\text{sparse}} = \max\left(\frac{0.5 \times \text{FLOPs}}{\text{Peak}}, \frac{0.5 \times \text{Data} + \text{Index}}{\text{BW}}\right)$$

密集计算时间：
$$T_{\text{dense}} = \max\left(\frac{\text{FLOPs}}{\text{Peak}}, \frac{\text{Data}}{\text{BW}}\right)$$

加速比：
$$S = \frac{T_{\text{dense}}}{T_{\text{sparse}}}$$

当compute-bound时：
$$S = 2 \times \frac{1}{1 + \text{decode\_overhead}} \approx 1.8$$

当memory-bound时：
$$S = \frac{1}{0.5 + 0.25} = 1.33$$

临界AI：当 $AI > 2 \times AI_{\text{ridge}}$ 时才能获得 >1.5× 加速。

</details>

**练习 2.7**: 设计一个BEV感知网络的数据流优化方案，考虑6个相机输入，每个相机1920×1080分辨率。如何安排计算顺序和内存布局以最小化DDR访问？

<details>
<summary>提示</summary>

考虑相机间的空间关系和特征重用机会。

</details>

<details>
<summary>参考答案</summary>

优化方案：

1. **相机分组处理**：
   - 前3相机（front-left, front, front-right）: 共享前向特征
   - 后3相机: 独立处理
   - 节省重叠区域的重复计算

2. **深度估计分块**：
   - 将深度范围[2m, 60m]分为4个bins
   - 每个bin独立计算，流水线处理
   - 内存需求: 1920×1080×4×2 = 16.6MB per camera

3. **BEV聚合策略**：
   - Ring buffer for 时序特征
   - Tile-based聚合: 200×200 BEV tiles
   - 每个tile只加载相关相机特征

4. **内存访问优化**：
   - 原始: 6×8.3MB×3 (read-process-write) = 149MB
   - 优化: 6×8.3MB×1.5 (fusion) = 75MB
   - DDR带宽降低: 50%

</details>

**练习 2.8**: 分析RT-2模型将VLM适配为VLA的计算开销。假设base VLM是7B参数，动作词汇表增加1000个tokens，计算额外的存储和计算需求。

<details>
<summary>提示</summary>

考虑embedding层扩展、输出层扩展和fine-tuning带来的变化。

</details>

<details>
<summary>参考答案</summary>

额外开销分析：

1. **Embedding层扩展**：
   - 新增tokens: 1000
   - Embedding dim: 4096
   - 额外参数: 1000 × 4096 = 4.1M
   - 存储（fp16）: 8.2MB

2. **输出层扩展**：
   - LM head扩展: 4096 × 1000 = 4.1M
   - 存储: 8.2MB

3. **LoRA适配器**（如使用）：
   - Rank=16, 应用于Q,V投影
   - 参数: 32层 × 2 × (4096×16×2) = 8.4M
   - 存储: 16.8MB

4. **推理计算增量**：
   - 动作token生成: +0.014 GFLOPs/token
   - 相对增加: 0.014/14 = 0.1%

5. **总开销**：
   - 参数增加: 16.6M / 7B = 0.24%
   - 存储增加: 33.2MB
   - 计算增加: <1%

结论：VLA适配开销很小，主要挑战在训练数据和对齐。

</details>

## 常见陷阱与错误 (Gotchas)

### 1. 算子性能评估误区

**陷阱**：仅看FLOPs评估性能
- FLOPs高不等于执行时间长
- 必须考虑内存访问模式和数据重用

**正确方法**：
- 使用Roofline模型综合评估
- Profile实际内存带宽利用率
- 考虑算子融合机会

### 2. 稀疏化应用误判

**陷阱**：对所有层应用稀疏化
- Memory-bound算子稀疏化可能降速
- 稀疏索引开销可能抵消收益

**正确方法**：
- 先分析算子的AI特征
- 只对compute-bound层稀疏化
- 实测稀疏化后的端到端性能

### 3. 批处理大小选择

**陷阱**：盲目增大batch size
- 大batch可能导致内存溢出
- 延迟敏感场景不适合大batch

**正确方法**：
- 根据片上存储容量选择
- 平衡延迟和吞吐量需求
- 考虑动态batching策略

### 4. 量化精度损失

**陷阱**：全网络统一量化
- 某些层对量化敏感（如attention的softmax）
- 首尾层通常需要更高精度

**正确方法**：
- 逐层分析量化敏感度
- 混合精度策略
- 保留关键层的高精度

### 5. 内存布局不当

**陷阱**：忽视数据布局对性能的影响
- NHWC vs NCHW影响缓存命中率
- 不对齐的内存访问降低带宽利用率

**正确方法**：
- 根据硬件特性选择布局
- 确保数据对齐到cache line
- 减少layout转换开销

## 最佳实践检查清单

### 算法分析阶段

- [ ] **工作负载画像**
  - 统计各类算子的比例
  - 识别性能瓶颈算子
  - 分析数据重用模式

- [ ] **精度需求评估**
  - 确定可接受的精度损失
  - 识别量化敏感层
  - 设计混合精度方案

- [ ] **延迟预算分配**
  - 分解端到端延迟目标
  - 为各模块分配时间预算
  - 识别关键路径

### 算子优化阶段

- [ ] **算子融合识别**
  - 标记可融合的算子序列
  - 评估融合收益
  - 考虑硬件约束

- [ ] **内存优化**
  - 计算working set大小
  - 设计tiling策略
  - 优化数据布局

- [ ] **并行策略选择**
  - 数据并行 vs 模型并行
  - 流水线深度设计
  - 负载均衡考虑

### 实现验证阶段

- [ ] **性能验证**
  - Cycle-accurate仿真
  - 带宽利用率测量
  - 算力利用率分析

- [ ] **精度验证**
  - 端到端精度测试
  - 中间结果比对
  - 极端case验证

- [ ] **系统集成**
  - 多模型调度策略
  - 内存池管理
  - 功耗优化方案