# NPU设计全流程教程：从算法到RTL实现

## 课程概述

本教程深入探讨神经网络处理器（NPU）的设计全流程，从上游算法映射到RTL实现、验证与仿真。课程聚焦自动驾驶和具身智能两大前沿应用场景，以200 TOPS推理性能为设计目标，支持2:4结构化稀疏和nvfp4量化。通过对比脉动阵列（Systolic Array）和数据流（Dataflow）两种主流架构，结合TPU和Groq TSP的实际案例，帮助读者掌握NPU设计的核心技术。

## 目标读者

- 具有深度学习算法背景的工程师
- 数字IC设计与验证工程师  
- 计算机体系结构研究人员
- AI编译器与系统软件开发者

## 预备知识

- 深度学习基础（CNN、Transformer、Attention机制）
- 数字电路设计（Verilog/SystemVerilog）
- 计算机体系结构（流水线、存储层次、并行计算）
- 线性代数与矩阵运算

---

## 第一部分：基础篇

### [第1章：NPU设计导论](chapter1.md)
1. NPU vs GPU vs CPU：架构特征对比
2. 推理加速的关键指标：延迟、吞吐量、能效
3. 自动驾驶场景：感知、预测、规划算法需求
4. 具身智能场景：VLM/VLA模型特征
5. 200 TOPS设计空间探索
6. **练习重点**：屋顶线模型(Roofline Model)分析

### [第2章：算法与算子分析](chapter2.md)
1. 自动驾驶核心网络剖析
   - 2D检测：YOLO系列、CenterNet
   - 3D检测：PointPillars、CenterPoint
   - BEV感知：BEVFormer、BEVDet
   - 轨迹预测与规划网络
2. VLM/VLA工作负载特征
   - CLIP与对比学习
   - LLaVA、Flamingo多模态架构
   - RT-1/RT-2机器人控制模型
3. 算子级性能分析
   - GEMM、Conv2D、Attention计算密度
   - Memory-bound vs Compute-bound分析
4. **练习重点**：算子融合优化策略

### [第3章：量化与稀疏化技术](chapter3.md)
1. nvfp4 (E2M1)数值系统
   - 动态范围与精度权衡
   - 指数偏置选择策略
   - Gradual underflow处理
2. 2:4结构化稀疏
   - 稀疏模式约束与压缩率
   - 稀疏索引编码方案
   - 与非结构化稀疏对比
3. 量化感知训练(QAT)与后训练量化(PTQ)
4. 混合精度策略与敏感层识别
5. **练习重点**：量化误差传播分析

### [第4章：存储系统与数据流](chapter4.md)
1. 存储层次设计
   - 片上SRAM vs HBM vs DDR
   - 存储带宽需求计算
   - Bank冲突与访问模式优化
2. 数据重用模式
   - Temporal vs Spatial重用
   - Loop tiling与blocking策略
   - Dataflow分类：WS/OS/RS
3. DMA设计与数据预取
4. 片上网络(NoC)基础
5. **练习重点**：带宽利用率计算

---

## 第二部分：脉动阵列架构（以TPU为例）

### [第5章：脉动阵列原理与设计](chapter5.md)
1. 脉动阵列基本概念
   - 数据流动与计算同步
   - Weight-stationary设计选择
   - 阵列维度与利用率分析
2. TPU v1-v4i架构演进
   - MXU (Matrix Multiply Unit)设计
   - 向量单元与标量单元
   - 双缓冲与流水线设计
3. 控制流与指令集
   - VLIW vs MIMD vs SIMD
   - 指令调度与依赖管理
4. **练习重点**：脉动阵列利用率优化

### [第6章：脉动阵列RTL实现](chapter6.md)
1. PE (Processing Element)设计
   - MAC单元与累加器
   - 权重寄存器与控制逻辑
   - 时序收敛策略
2. 阵列级互连
   - 数据广播网络
   - 部分和传递链
   - Skew buffer设计
3. 控制器设计
   - FSM状态机设计
   - Counter chain与地址生成
   - 异常处理与边界情况
4. **练习重点**：RTL时序分析与优化

### [第7章：TPU编译器与映射](chapter7.md)
1. XLA编译流程
   - HLO (High-Level Optimizer)图优化
   - 算子融合与内存规划
   - Tiling策略与参数选择
2. 矩阵乘法映射
   - 大矩阵分块策略
   - Padding与对齐要求
   - 批处理维度处理
3. 卷积映射优化
   - Im2col vs Direct convolution
   - Winograd变换适用性
   - Depthwise/Pointwise分解
4. **练习重点**：映射效率分析

### [第8章：脉动阵列验证方法](chapter8.md)
1. 功能验证策略
   - 单元测试vs系统测试
   - 定向测试vs随机测试
   - 覆盖率驱动验证
2. 性能验证
   - Cycle-accurate模拟器
   - 性能计数器设计
   - 瓶颈分析方法
3. 数值验证
   - Bit-accurate参考模型
   - 误差累积分析
   - Corner case测试
4. **练习重点**：验证计划制定

---

## 第三部分：数据流架构（以Groq TSP为例）

### [第9章：数据流架构原理](chapter9.md)
1. 数据流计算模型
   - 静态vs动态数据流
   - Token-based执行
   - 确定性执行优势
2. Groq TSP架构特征
   - 无外部DRAM设计理念
   - 编译时调度
   - 确定性延迟保证
3. 与脉动阵列对比
   - 灵活性vs效率权衡
   - 编程模型差异
   - 适用场景分析
4. **练习重点**：数据流图构建与分析

### [第10章：TSP微架构设计](chapter10.md)
1. 计算单元设计
   - Vector ALU阵列
   - Matrix multiplication unit
   - 特殊函数单元(SFU)
2. 片上存储系统
   - 分布式SRAM banks
   - 地址生成与仲裁
   - Multi-casting机制
3. 片上网络设计
   - 2D mesh拓扑
   - 路由算法与死锁避免
   - 虚通道与流控
4. **练习重点**：NoC带宽与延迟计算

### [第11章：数据流RTL实现](chapter11.md)
1. 流处理器设计
   - 指令解码与发射
   - 操作数收集与转发
   - 背压处理机制
2. 同步与调度
   - 全局同步机制
   - Credit-based流控
   - Stall处理策略
3. 功耗优化技术
   - Clock gating
   - Power gating
   - DVFS支持
4. **练习重点**：流水线hazard分析

### [第12章：TSP编译器技术](chapter12.md)
1. 静态调度算法
   - 指令调度与资源分配
   - 寄存器分配策略
   - 内存布局优化
2. 数据流图优化
   - 公共子表达式消除
   - 死代码消除
   - 循环优化技术
3. 自动并行化
   - 数据并行识别
   - 流水线并行
   - 模型并行策略
4. **练习重点**：调度算法复杂度分析

---

## 第四部分：系统集成与优化

### [第13章：多核扩展与互连](chapter13.md)
1. Scale-up架构
   - 多芯片互连技术
   - Cache一致性协议
   - NUMA效应与优化
2. Scale-out架构  
   - 分布式训练vs推理
   - 参数服务器vs AllReduce
   - 梯度压缩与量化
3. 芯片间互连
   - NVLink/CXL/UCIe对比
   - 拓扑选择：Ring/Mesh/Torus
   - 集合通信优化
4. **练习重点**：多核性能建模

### [第14章：软硬件协同设计](chapter14.md)
1. 硬件抽象层(HAL)设计
   - 驱动接口定义
   - 内存管理接口
   - 同步原语实现
2. 运行时系统
   - 任务调度器设计
   - 内存分配器
   - 性能profiling接口
3. 调试与诊断
   - Hardware trace机制
   - Performance counter
   - Error reporting系统
4. **练习重点**：API设计最佳实践

### [第15章：性能分析与优化](chapter15.md)
1. 性能建模方法
   - Analytical model
   - Cycle-level simulation
   - ML-based预测
2. 瓶颈识别技术
   - Roofline分析
   - Critical path分析
   - 资源利用率分析
3. 优化案例研究
   - Attention优化：Flash Attention
   - 卷积优化：Implicit GEMM
   - 激活函数融合
4. **练习重点**：性能优化决策树

### [第16章：工程实践与部署](chapter16.md)
1. 芯片bring-up流程
   - 功能验证检查清单
   - 性能验证与校准
   - 可靠性测试
2. 量产考虑
   - 良率分析与binning
   - 功耗分级
   - 热设计与散热
3. 实际部署案例
   - 数据中心部署
   - 边缘端部署
   - 车载部署要求
4. **练习重点**：部署检查清单

---

## 附录

### [附录A：数学基础回顾](appendix_a.md)
- 线性代数要点
- 数值分析基础
- 概率论与信息论

### [附录B：常用算子参考](appendix_b.md)
- GEMM变体详解
- 卷积算子变体
- 注意力机制变体

### [附录C：工具链使用指南](appendix_c.md)
- RTL仿真工具
- 性能分析工具
- 编译器工具链

### [附录D：术语表](appendix_d.md)
- 中英文术语对照
- 缩略语列表

---

## 学习路径建议

### 快速路径（4周）
- 第1、3、5、9章：核心概念
- 第6、11章（选读）：RTL要点
- 第15章：性能优化

### 标准路径（8周）
- 第一部分：基础篇（2周）
- 第二部分或第三部分（3周，二选一）
- 第四部分：系统集成（2周）
- 实践项目（1周）

### 深入路径（12周）
- 完整阅读所有章节
- 完成所有练习题
- 实现mini项目
- 参与开源项目

## 配套资源

- 代码仓库：示例RTL代码、验证环境
- 在线仿真平台：免费FPGA验证环境
- 论坛社区：技术交流与答疑
- 视频课程：重点章节配套讲解

---

*本教程持续更新中，欢迎反馈与贡献*
