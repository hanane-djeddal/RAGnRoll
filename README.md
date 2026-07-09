# RAGnRoll: Learning to Iteratively Retrieve and Generate Attributable Answer Snippets

This includes official code, model checkpoints, and training data from RAGnRoll: Learning to Iteratively Retrieve and Generate Attributable Answer Snippets (ACM TOIS)

 [13B Trained Model]() | [Paper](https://akariasai.github.io/files/adaptive_retrieval_augmented_lm_arxiv.pdf) | [Synthetic Training data](hanane/rag-and-roll-hagrid-attributable-seg-proba) | 

**RAGnRoll**: is a framework to train an arbitrary LLM for optimized attributed answer generation for complex information needs through a multi-round Retrieval-Augmented Generation (RAG) architecture. The LLM is trained to build the final answer by incrementally generating answer snippets guided by subqueries dynamically generated in each iteration. The model is trained through specially crafted dataset. Both trained model and training data are available on Huggingface.
