> 全上下文 科学论点验证
> 

# 1 Intro

---

[arxiv.org](https://arxiv.org/pdf/2112.01640)

https://github.com/dwadden/multivers

## 1.1 关于 Scientific clain verification 科学声明验证

---

任务涉及通过对候选摘要进行标签来验证科学声明，标签为 

- `SUPPORTS`（支持）
- `REFUTES`（驳斥）
- `NEI`（信息不足）

对于标记为 `SUPPORTS` 或 `REFUTES` 的摘要，系统还必须识别出具体句子（理由），这些句子证明了标签 (给定的声明claim) 的合理性

## 1.2 关于checkpoints

---

1. **longformer_large_science.ckpt**
    - 在科学文献语料库上预训练。该模型没有在任何事实检查数据上进行训练；它是所有其他模型的起点。
2. **fever.ckpt**
    - 在 FEVER 数据集上进行预训练的模型，用于通用领域的事实检查。
3. **fever_sci.ckpt**
    - 直接使用在 FEVER 和两个弱监督科学数据集（PubMedQA 和 Evidence Inference）上预训练的模型，进行目标数据集（HealthVer）的评估，不进行进一步微调。
4. **scifact.ckpt**
    - 从 `fever_sci.ckpt` 开始，在 SciFact 数据集上进行微调的模型。
5. **covidfact.ckpt**
    - 从 `fever_sci.ckpt` 开始，在 CovidFact 数据集上进行微调的模型。
6. **healthver.ckpt**
    - 从 `fever_sci.ckpt` 开始，在 HealthVer 数据集上进行微调的模型。

# 2 Methodology

## 2.1 现有方法 / 动机

---

Pipeline: 先从文档中 提取 (理由 / 证据 句) , 再标注 (标签) 

- **选取理由句（Rationale Selection）**：
    - 首先，从候选文档中选出支持或反驳声明的具体句子。这一步骤使用了一个独立的模型，通常是一个基于Transformer的模型，如Longformer或RoBERTa。
    - 模型根据输入的声明和候选文档，选出最有可能作为证据的句子。
- **标签预测（Label Prediction）**：
    - 然后，使用另一个独立的模型对选出的理由句进行标签预测。
    - 这个模型根据理由句和声明，预测声明是被支持（SUPPORT）还是被反驳（REFUTE），或者信息不足（NEI，Not Enough Info）。

### 2.1.1 问题

---

- 会忽略上下文
- 大量句子级别的标注

## 2.2 MultiVerS 的 改进方法 (架构设计)

---

- **共享编码**：
    - 使用Longformer模型对整个声明和候选文档进行编码。不会受到512个token的限制 (相比BERT, RoBERTa等)
    - 声明和候选摘要（包括标题和句子）被连接，使用特殊的 `<s>` 标记分隔。这允许模型处理整个文档上下文
    - 滑动窗口机制: 分割的每个部分之间有一定的重叠 → 窗口化的自注意力机制 + 全局注意力机制
- **联合训练 (多任务学习)**：
    - 在同一个模型中同时进行理由句选取和标签预测，利用共享的上下文信息，提高预测的一致性和准确性。
    - 模型同时预测理由和整体事实检查标签。
    - 理由通过对每个句子的全局上下文化标记进行二分类头来预测。
    - 整体标签通过对整个文档的全局上下文化标记进行三分类头来预测。
    - 损失函数结合了标签和理由预测的交叉熵损失，并在开发集上调整理由损失的可调参数。

# 3 Datasets

## 3.1 Pretrain (1 + 2)

### **3.1.1 常规域事实检查注释** 数据集 (1)  (longformer→ fever)

---

数据集充足，但对科学声明的泛化较差

- **FEVER**
    - 通过重新编写 Wikipedia 句子创建的
    - 原子声明，并与 Wikipedia 文章进行了验证

### **3.1.2 弱标签的 领域内数据** 数据集 (2) (fever→ fever_sci)

---

域内事实检查注释是“黄金标准”，但创建成本高且需要专家注释人员

可以使用高精度启发式方法（第4.2节描述）生成这些数据的文档级事实检查标签

- **EvidenceInference**
    - 从临床试验报告中提取的声明
    - 其中审查了干预对结果的影响, 使用基于规则的启发式方法将这些提示转换为声明
- **PubMedQA**
    - 生物医学研究摘要问答. 将论文标题作为声明，并将匹配的摘要作为证据来源

## 3.2 Fine-tuning (3) (fever_sci → scifact, healthver, covidfact)

---

不同水平的监督

- **零样本**：只进行预训练，不使用任何领域内数据。**直接用fever_sci**
- **少样本**：在目标数据集中的少量示例（45 个声明）上进行微调。
- **完全监督**：在目标数据集的所有声明上进行微调。

### 3.2.1 目标科学事实检查 数据集

---

适合MultiVerS, 模型可以在有或没有理由注释的数据上进行训练

- **SciFact**
    - 通过重新编写引文句子并根据所引用文档的摘要进行验证的生物医学声明
    - 原子性声明, 这些声明通过引用文档的摘要进行了验证
- **HealthVer**
    - 从 CORD-19 语料库中提取的摘要中验证 COVID 相关声明
    - 声明可能是复杂的。否定声明出现在文章片段中。每个声明提供候选摘要，但其中一些不包含足够的信息来支持/否定判决，因此标记为 NEI。
- **COVIDFact**
    - 从一个子论坛上抓取的 COVID 相关声明，并与链接的科学论文和通过 Google 搜索检索的文档进行验证
    - 声明可能是复杂的，每个声明都有提供的候选摘要。所有候选摘要都支持或否定声明。声明否定通过自动替换原始声明中的显著词创建，因此标签可能有噪声

# 4 评估

## 4.1 项目代码

---

验证用到另一个项目

https://github.com/allenai/scifact-evaluator

评估脚本

```bash
python libs/scifact-evaluator/evaluator/eval.py --labels_file data/healthver/claims_test.jsonl --preds_file prediction/healthver_new.jsonl --verbose
```

## 4.2 指标

---

1. 使用摘要级和句子级 F1 分数评估模型。
2. 摘要级评估考虑模型是否正确标记摘要。
3. 句子级评估考虑模型是否正确识别出证明标签的理由句子。

# 5 复现

---

论文数据如下: 

- 绿色: 能复现
- 黄色: 有较小偏差
- 红色: 无法复现(偏差太大)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/3a352054-ceaa-444b-a420-1b33e16e0047/f1118d8e-cf73-4dae-870a-5d31e01a431d/Untitled.png)

## 5.1 处理过程记录

---

https://github.com/ryanyhliu/multivers/commits/main/

## 5.2 结果

### 5.2.1 复现 HealthVer - Full ：可以复现

---

结论：可以复现

```json
{
  "abstract_label_only_f1": 0.7758913412563666, // 和 77.6 对应
  "abstract_rationalized_f1": 0.7419354838709679, // 带推理的摘要支持度 （监测模型是否正确识别 具体的推理句子， 从而支持标签）
  "sentence_selection_f1": 0.7821176470588236, // 选择了相关的句子
  "sentence_label_f1": 0.6927058823529412 // 和 69.1 几乎没差
}
```

### 5.2.2 复现 HealthVer - Zero：有问题 (feversci)

---

结论：abstract的能复现，sentence的不行. 跑了2遍, 结果没有波动

```json
{
  "abstract_label_only_precision": 0.6059113300492611,
  "abstract_label_only_recall": 0.20534223706176963,
  "abstract_label_only_f1": 0.30673316708229426,  // 和 Zero 的 F score 对上了， 30.7
  "abstract_rationalized_precision": 0.06403940886699508,
  "abstract_rationalized_recall": 0.021702838063439065,
  "abstract_rationalized_f1": 0.032418952618453865,
  "sentence_selection_precision": 0.4375,
  "sentence_selection_recall": 0.012704174228675136,
  "sentence_selection_f1": 0.02469135802469136,
  "sentence_label_precision": 0.40625,
  "sentence_label_recall": 0.011796733212341199,
  "sentence_label_f1": 0.022927689594356263 // 没对应上
}
```

### 5.2.3 复现 HealthVer - ~~Few~~ ：有问题 (longformer)

---

第一次predict, 不对

```json
{
  "abstract_label_only_precision": 0.4130675526024363,
  "abstract_label_only_recall": 0.6227045075125208,
  "abstract_label_only_f1": 0.49667110519307583, // 和 Few 的 54.7 差了一点
  "abstract_rationalized_precision": 0.17054263565891473,
  "abstract_rationalized_recall": 0.2570951585976628,
  "abstract_rationalized_f1": 0.20505992010652463,
  "sentence_selection_precision": 0.12830364419606474,
  "sentence_selection_recall": 1.0,
  "sentence_selection_f1": 0.22742751006088124,
  "sentence_label_precision": 0.08126673652346024,
  "sentence_label_recall": 0.6333938294010889,
  "sentence_label_f1": 0.14405118150861623 // 和 Few 的 35.7 完全不一样
}
```

第二次predict, 完全不对

```json
{
  "abstract_label_only_precision": 0.24916943521594684,
  "abstract_label_only_recall": 0.3756260434056761,
  "abstract_label_only_f1": 0.29960053262316905,
  "abstract_rationalized_precision": 0.08748615725359911,
  "abstract_rationalized_recall": 0.1318864774624374,
  "abstract_rationalized_f1": 0.10519307589880159,
  "sentence_selection_precision": 0.12012051349227142,
  "sentence_selection_recall": 0.8321234119782214,
  "sentence_selection_f1": 0.20993589743589747,
  "sentence_label_precision": 0.046371495939219284,
  "sentence_label_recall": 0.32123411978221417,
  "sentence_label_f1": 0.08104395604395605
}
```

第三次 predict, 跟第一次结果比较像

```json
{
  "abstract_label_only_precision": 0.3997785160575858,
  "abstract_label_only_recall": 0.6026711185308848,
  "abstract_label_only_f1": 0.48069241011984026, // 和 54.7 有差距
  "abstract_rationalized_precision": 0.04540420819490587,
  "abstract_rationalized_recall": 0.06844741235392321,
  "abstract_rationalized_f1": 0.05459387483355526,
  "sentence_selection_precision": 0.0847457627118644,
  "sentence_selection_recall": 0.09074410163339383,
  "sentence_selection_f1": 0.0876424189307625,
  "sentence_label_precision": 0.04745762711864407,
  "sentence_label_recall": 0.050816696914700546,
  "sentence_label_f1": 0.04907975460122699 // 不对
}
```

### 5.2.4 复现 CovidFact - Full ：可以复现

---

结论：可以复现

```json
{
  "abstract_label_only_precision": 0.7728706624605678,
  "abstract_label_only_recall": 0.7728706624605678,
  "abstract_label_only_f1": 0.7728706624605678, // 和 77.3 对应
  "abstract_rationalized_precision": 0.5962145110410094,
  "abstract_rationalized_recall": 0.5962145110410094,
  "abstract_rationalized_f1": 0.5962145110410094,
  "sentence_selection_precision": 0.5305084745762711,
  "sentence_selection_recall": 0.5839552238805971,
  "sentence_selection_f1": 0.5559502664298402,
  "sentence_label_precision": 0.41694915254237286,
  "sentence_label_recall": 0.458955223880597,
  "sentence_label_f1": 0.43694493783303723 // 和 43.7 对应
}
```

### 5.2.5 复现 CovidFact - Zero：有问题

---

结论：跑了2次, abstract 差了5%, sentence 完全对应不上

```json
{
  "abstract_label_only_precision": 0.5348837209302325,
  "abstract_label_only_recall": 0.5078864353312302,
  "abstract_label_only_f1": 0.5210355987055014, // 和 47 对应不上，但相差不是很多
  "abstract_rationalized_precision": 0.08970099667774087,
  "abstract_rationalized_recall": 0.08517350157728706,
  "abstract_rationalized_f1": 0.08737864077669903,
  "sentence_selection_precision": 0.825,
  "sentence_selection_recall": 0.061567164179104475,
  "sentence_selection_f1": 0.11458333333333333,
  "sentence_label_precision": 0.7,
  "sentence_label_recall": 0.05223880597014925,
  "sentence_label_f1": 0.09722222222222221 // 和 23 对应不上，相差很多
}
```

### 5.2.6 复现 CovidFact - ~~Few~~ : 有问题 (longformer)

---

第一次 predict, 不对

```json
{
  "abstract_label_only_precision": 0.7051282051282052,
  "abstract_label_only_recall": 0.17350157728706625,
  "abstract_label_only_f1": 0.27848101265822783, // 和 69.7 对不上
  "abstract_rationalized_precision": 0.21794871794871795,
  "abstract_rationalized_recall": 0.05362776025236593,
  "abstract_rationalized_f1": 0.08607594936708861,
  "sentence_selection_precision": 0.14934289127837516,
  "sentence_selection_recall": 0.2332089552238806,
  "sentence_selection_f1": 0.1820830298616169,
  "sentence_label_precision": 0.1063321385902031,
  "sentence_label_recall": 0.166044776119403,
  "sentence_label_f1": 0.1296431172614712 // 和 37.4 对不上
}
```

第二次 predict, abstract 接近

```json
{
  "abstract_label_only_precision": 0.6773162939297125,
  "abstract_label_only_recall": 0.668769716088328,
  "abstract_label_only_f1": 0.6730158730158731, // 和 69.7 相差不大
  "abstract_rationalized_precision": 0.26517571884984026,
  "abstract_rationalized_recall": 0.2618296529968454,
  "abstract_rationalized_f1": 0.2634920634920635,
  "sentence_selection_precision": 0.19219653179190752,
  "sentence_selection_recall": 0.9925373134328358,
  "sentence_selection_f1": 0.3220338983050847,
  "sentence_label_precision": 0.12933526011560695,
  "sentence_label_recall": 0.667910447761194,
  "sentence_label_f1": 0.2167070217917676 // 和 37.4 对不上
} 
```

第三次 predict, 不对

```json
{
  "abstract_label_only_precision": 0.3217665615141956,
  "abstract_label_only_recall": 0.3217665615141956,
  "abstract_label_only_f1": 0.3217665615141956, // 不对
  "abstract_rationalized_precision": 0.12302839116719243,
  "abstract_rationalized_recall": 0.12302839116719243,
  "abstract_rationalized_f1": 0.12302839116719243,
  "sentence_selection_precision": 0.18980169971671387,
  "sentence_selection_recall": 1.0,
  "sentence_selection_f1": 0.3190476190476191,
  "sentence_label_precision": 0.06196883852691218,
  "sentence_label_recall": 0.32649253731343286,
  "sentence_label_f1": 0.10416666666666667 // 不对
}
```

第四次 predict, 和第二次差不多

```json
{
  "abstract_label_only_precision": 0.6782334384858044,
  "abstract_label_only_recall": 0.6782334384858044,
  "abstract_label_only_f1": 0.6782334384858044,
  "abstract_rationalized_precision": 0.26813880126182965,
  "abstract_rationalized_recall": 0.26813880126182965,
  "abstract_rationalized_f1": 0.26813880126182965,
  "sentence_selection_precision": 0.19303054032889586,
  "sentence_selection_recall": 0.9197761194029851,
  "sentence_selection_f1": 0.3190938511326861,
  "sentence_label_precision": 0.1288175411119812,
  "sentence_label_recall": 0.6138059701492538,
  "sentence_label_f1": 0.21294498381877025
}
```

### 5.2.7 复现 SciFact - Full / Zero ：有问题

---

<aside>
💡 没有金标准数据

</aside>

项目中自带的数据，缺少evidence字段

> Traceback (most recent call last):
File "scifact-evaluator/evaluator/eval.py", line 248, in <module>
main()
File "scifact-evaluator/evaluator/eval.py", line 240, in main
golds = [unify_label(entry) for entry in golds]
File "scifact-evaluator/evaluator/eval.py", line 240, in <listcomp>
golds = [unify_label(entry) for entry in golds]
File "scifact-evaluator/evaluator/eval.py", line 31, in unify_label
for doc, ev in gold["evidence"].items():
KeyError: 'evidence'
> 

scifact-evaluator 项目中自带了一些scifact的文件，也都不行

# 6 预训练

---

```bash
python script/pretrain.py --datasets fever,pubmedqa,evidence_inference --gpus 1
```

训练时有问题， 仍在排查，如下

> RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.HalfTensor [16, 512, 30]], which is output 0 of AsStridedBackward0, is at version 1; expected version 0 instead. Hint: the backtrace further above shows the operation that failed to compute its gradient. The variable in question was changed in there or anywhere later. Good luck!

这个错误表明在计算梯度时，一个需要的变量被一个就地操作（inplace operation）修改了
关键词： PyTorch异常监测，就地操作
> 

# 7 推理

## 7.1 predict.sh 使用方法

---

```bash
bash script/predict.sh covidfact
```

## 7.2 predict.py 直接使用

---

指定 fever_sci 在healthver 上跑 （修改完hyperparam file问题， 正常运行即可）

```bash
python multivers/predict.py --checkpoint_path=checkpoints/fever_sci.ckpt --input_file=data/healthver/claims_test.jsonl --corpus_file=data/healthver/corpus.jsonl --output_file=prediction/healthver_new.jsonl
```

# 8 Unlimiformer

## 8.1 Intro

---

[arxiv.org](https://arxiv.org/pdf/2305.01625)

# 9 *Analysis

---

- **在已经训练好的模型上使用Unlimiformer**：这种方法不需要重新训练模型，而是在模型已经训练好的情况下，直接在推理（inference）过程中使用Unlimiformer。这种方法可以立即提升模型的性能，适用于那些希望在不重新训练模型的情况下获得性能提升的场景。
- **使用Unlimiformer进行训练**：这种方法是在训练过程中就集成Unlimiformer，用 Unlimiformer 替代 Longformer, 从而使模型在训练过程中就能学习到如何处理更长的输入序列。这样，模型的整体性能将会更高。对于需要最大化性能提升的场景，建议在训练过程中使用Unlimiformer。

## 9.1 Difference

---

- Longformer通过稀疏注意力机制扩展了Transformer的上下文窗口，使其能够处理更长的输入。
- Unlimiformer通过使用k近邻（k-NN）索引来增强预训练的编码器-解码器模型，使其能够处理无限长度的输入。它通过在每个解码层注入k-NN搜索，以选择每个注意力头应关注的输入令牌，避免了传统Transformer中处理长输入的计算瓶颈。