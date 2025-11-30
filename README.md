# Building Intelligent Troubleshooting Agents

This repository contains coursework and practice activities from the **Building Intelligent Troubleshooting Agents** course by Microsoft on Coursera. The course covers the development of AI-powered agents capable of diagnosing, analyzing, and resolving technical issues using machine learning and natural language processing.

## Course Structure

### 1. LLM Fine-tuning
Learn how to fine-tune large language models for troubleshooting applications.

**Activities:**
- **Model & Dataset** (`model&dataset.*`): Introduction to selecting appropriate models and datasets for fine-tuning
- **Prepare Data** (`prepdata.*`): Data preprocessing and preparation techniques for LLM training
- **Fine-tune LLM** (`finetunellm.*`): Step-by-step guide to fine-tuning language models
- **LoRA** (`lora.ipynb`, `lora_example.py`): Low-Rank Adaptation techniques for efficient fine-tuning
- **QLoRA** (`qlora.*`): Quantized Low-Rank Adaptation for memory-efficient training
- **PEFT** (`peft.ipynb`, `peft_example.py`): Parameter-Efficient Fine-Tuning methods
- **LLM Tuning** (`llmtuning.*`): Advanced tuning strategies and hyperparameter optimization
- **Compare Fine-tuning** (`comparefinetuning.*`): Comparative analysis of different fine-tuning approaches
- **Apply Evaluation Methods** (`applyevalmethods.*`): Techniques for evaluating fine-tuned models

### 2. Fundamentals of AI Agents
Understand the core concepts and requirements for building AI troubleshooting agents.

**Activities:**
- **Requirements** (`requirements.*`): Identifying and defining requirements for troubleshooting agents
- **Troubleshooting Agent** (`troubleshootagent.*`): Building a basic troubleshooting agent from scratch
- **Knowledge Base** (`data/troubleshooting_knowledge_base.json`): Creating and structuring knowledge bases for agent reference

### 3. NLP (Natural Language Processing)
Integrate NLP techniques to enable agents to understand and process user queries.

**Activities:**
- **NLP for Troubleshooting** (`nlpfortroubleshoot.*`): Applying NLP techniques to troubleshooting scenarios
- **Sentiment Analysis** (`sentimentanalysis.*`): Analyzing user sentiment to prioritize and understand issues
- **Integrating NLP** (`integratingnlp.*`): Combining NLP with troubleshooting logic
- **Agent Interface** (`Agent Interface/`): Building a web-based chatbot interface for user interaction
  - `index.html`: Front-end interface
  - `chatbot.js`: Chatbot logic and interaction handling

### 4. Troubleshooting Agent
Develop comprehensive troubleshooting agents with classification, decision-making, and solution recommendation capabilities.

**Activities:**
- **Classification Models** (`classificationmodels.*`): Training models to classify different types of issues
- **Evaluation of Classification** (`evalclassification.*`): Assessing the performance of classification models
- **Mechanisms** (`mechanisms.*`): Implementing error handling and validation mechanisms
- **Logging** (`logging.ipynb`, `logging_example.py`): Setting up logging for debugging and monitoring
- **Decision Making** (`decisionmaking.*`): Building decision trees and logic for root cause analysis
- **Solution Recommendation** (`solutionrec.*`): Creating recommendation systems using KNN and other algorithms
- **Implementation** (`implementation.*`): Integrating all components into a complete troubleshooting agent
- **Troubleshooting Agent** (`troubleshootagent.*`): Full implementation with anomaly detection, root cause analysis, and solution recommendation

### 5. Testing
Test, evaluate, and optimize troubleshooting agents for production deployment.

**Activities:**
- **Test Cases** (`testcases.*`): Designing comprehensive test cases for ML systems
  - Typical case testing
  - Edge case handling
  - Error handling with missing values
- **Evaluate Effectiveness** (`evaleffectiveness.*`): Measuring agent performance using:
  - Accuracy and precision metrics
  - Response time analysis
  - Resource usage monitoring (CPU/Memory)
  - Stress testing
  - Cross-validation
- **Optimization** (`optimization.*`): Implementing optimization techniques:
  - Model pruning
  - Quantization with TensorFlow Lite
  - Feature selection
- **Test & Optimize** (`test&optimize.*`): Combined testing and optimization workflow

## Key Technologies

- **Machine Learning**: scikit-learn, TensorFlow, PyTorch
- **NLP**: Transformers, BERT, Sentiment Analysis
- **Fine-tuning**: LoRA, QLoRA, PEFT
- **Optimization**: TensorFlow Model Optimization, Pruning, Quantization
- **Testing**: pytest, ipytest
- **Logging**: Python logging module

## Outputs

- **Fine-tuned Models**: `fine_tuned_bert/` - BERT model fine-tuned for sentiment analysis
- **Logs**: `ml_errors.log`, `ml_pipeline.log` - Error and pipeline logging outputs
- **Results**: Checkpoints and training results from various experiments

---

*Course by Microsoft on Coursera*