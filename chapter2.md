# 第2章：算法与算子分析

深度学习算法的计算特征直接决定了NPU架构的设计选择。本章深入分析自动驾驶和具身智能两大场景的核心算法，从网络结构、数据流模式、算子组成等多个维度剖析其计算需求。通过理解不同算法的工作负载特征，我们能够识别性能瓶颈、评估架构适配性，并指导后续的硬件设计优化。本章将建立算法需求与硬件能力之间的量化映射关系，为200 TOPS NPU的设计空间探索提供理论基础。

## 2.1 自动驾驶核心网络剖析

自动驾驶系统的感知栈涵盖了从2D图像检测到3D点云处理、从单帧感知到时序融合的多种网络架构。这些网络在计算密度、内存访问模式、数据重用机会等方面展现出显著差异。

### 2.1.1 2D检测网络：YOLO系列与CenterNet

#### YOLO系列架构演进

YOLOv8作为当前主流的实时检测网络，其backbone采用CSPDarknet架构，通过Cross Stage Partial连接减少计算量的同时保持特征表达能力。

**计算特征分析：**

主干网络的计算量分布：
$$\text{FLOPs}_{\text{backbone}} = \sum_{l=1}^{L} 2 \times C_{in}^{(l)} \times C_{out}^{(l)} \times K^{2(l)} \times H^{(l)} \times W^{(l)}$$

其中典型的下采样策略为：
- Stage 1: $640 \times 640 \times 3 \to 320 \times 320 \times 64$
- Stage 2: $320 \times 320 \times 64 \to 160 \times 160 \times 128$
- Stage 3: $160 \times 160 \times 128 \to 80 \times 80 \times 256$
- Stage 4: $80 \times 80 \times 256 \to 40 \times 40 \times 512$
- Stage 5: $40 \times 40 \times 512 \to 20 \times 20 \times 1024$

**内存访问模式：**

CSP结构的特征图分割策略：
$$X = [X_1, X_2], \quad X_1 \in \mathbb{R}^{H \times W \times C/2}, X_2 \in \mathbb{R}^{H \times W \times C/2}$$

这种分割降低了内存带宽需求：
$$\text{Bandwidth}_{\text{CSP}} = \text{Bandwidth}_{\text{standard}} \times (1 + \gamma)$$
其中 $\gamma \approx 0.5$ 为CSP的带宽节省率。

#### CenterNet的中心点检测机制

CenterNet将目标检测转化为关键点检测问题，通过高斯核生成中心点热图：
$$Y_{xyc} = \exp\left(-\frac{(x-\tilde{x})^2 + (y-\tilde{y})^2}{2\sigma_p^2}\right)$$

其中 $\sigma_p$ 与目标尺寸成正比：$\sigma_p = \max(1, \frac{1}{3}\sqrt{wh})$

**计算优势：**
1. 无需NMS后处理，减少串行计算瓶颈
2. 热图生成可通过可分离卷积加速
3. 单阶段推理，避免RPN的额外开销

### 2.1.2 3D检测网络：PointPillars与CenterPoint

#### PointPillars的柱状体编码

点云数据的稀疏性为NPU设计带来独特挑战。PointPillars通过将点云组织为柱状体（pillars）来规则化数据结构。

**Pillar特征编码：**
$$f_{i} = [x_i, y_i, z_i, r_i, x_i - x_c, y_i - y_c, z_i - z_c]$$

其中 $(x_c, y_c, z_c)$ 为柱内点的质心。

**稀疏性分析：**
典型场景下，仅约10%的pillars包含点：
$$\text{Sparsity} = 1 - \frac{N_{\text{non-empty}}}{N_{\text{total}}} \approx 0.9$$

这种稀疏性可通过以下策略优化：
1. 动态批处理非空pillars
2. 使用稀疏卷积减少无效计算
3. 自适应pooling聚合pillar特征

#### CenterPoint的多尺度特征聚合

CenterPoint在BEV空间进行3D检测，关键创新在于其多尺度中心特征提取：

```
    3D Points
        ↓
    Voxelization
        ↓
    3D Sparse Conv
        ↓
    BEV Feature Map
        ↓
    Center Heatmap + 3D Box Regression
```

**计算复杂度分析：**

3D稀疏卷积的实际计算量：
$$\text{FLOPs}_{\text{sparse}} = \text{FLOPs}_{\text{dense}} \times (1 - \text{Sparsity})^2$$

对于90%稀疏度，实际计算量仅为密集卷积的1%。

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

**Agent-Scene交互建模：**
$$h_i^{(l+1)} = h_i^{(l)} + \text{MHA}([h_i^{(l)}, E_{\text{pos}}^i], [H^{(l)}, E_{\text{pos}}])$$

其中：
- $h_i$: agent $i$ 的隐状态
- $E_{\text{pos}}$: 位置编码
- $H$: 所有agents和地图元素的特征集合

**多模态轨迹生成：**
$$P(Y|X) = \sum_{k=1}^{K} w_k \cdot \mathcal{N}(\mu_k(X), \Sigma_k(X))$$

典型设置 $K=6$ 个模态，每个模态预测 $T=80$ 个时间步（8秒）。

**计算需求：**
- Attention计算: $O(N^2 \times T \times D)$, 其中 $N \approx 200$ (agents + map)
- MLP解码: $O(K \times N \times T \times D^2)$
- 总计: 约5 GFLOPs per scene

## 2.2 VLM/VLA工作负载特征

视觉语言模型（VLM）和视觉语言动作模型（VLA）代表了多模态AI的前沿，其计算特征与传统CV网络有显著差异。

### 2.2.1 CLIP与对比学习的计算需求

#### CLIP的双塔架构

CLIP通过对比学习对齐视觉和文本表示：

**视觉编码器（ViT-L/14）：**
$$Z_v = \text{ViT}(I) \in \mathbb{R}^{B \times D}$$

计算量：
$$\text{FLOPs}_{\text{ViT}} = 2 \times N \times (D \times D_{mlp} + N \times D^2/H)$$

其中：
- $N = (224/14)^2 = 256$ patches
- $D = 1024$ 维度
- $D_{mlp} = 4 \times D = 4096$
- $H = 16$ 注意力头

总计约 81 GFLOPs。

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