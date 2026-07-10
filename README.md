# RAGnRoll: Learning to Iteratively Retrieve and Generate Attributable Answer Snippets

This includes official code, model checkpoints, and training data from RAGnRoll: Learning to Iteratively Retrieve and Generate Attributable Answer Snippets (ACM TOIS)

 [13B Trained Model](AttriRAG/ragnroll_llama2_13b) | [Paper](https://akariasai.github.io/files/adaptive_retrieval_augmented_lm_arxiv.pdf) | [Synthetic Training data](https://huggingface.co/datasets/AttriRAG/ragnroll_train_data_hagrid_seg_proba) | 

**RAGnRoll**: is a framework to train an arbitrary LLM for optimized attributed answer generation for complex information needs through a multi-round Retrieval-Augmented Generation (RAG) architecture. The LLM is trained to build the final answer by incrementally generating answer snippets guided by subqueries dynamically generated in each iteration. The model is trained through specially crafted dataset. Both trained model and training data are available on Huggingface.


## Content
1. [Quick Start](#quick-start)
2. [Inference](#inference)
4. [Training](#training)
5. [Synthetic Data Generation](#synthetic-data-generation)
6. [Contact](#contact)



## Contact
If you have questions, please open an issue mentioning @hanane-djeddal or send an email to hanane.djeddal[at]irit.fr.
