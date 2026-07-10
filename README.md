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


## Quick Start

You can download [RAGnRoll](AttriRAG/ragnroll_llama2_13b) from Huggingface, trained on using tools, and use the agent calls for inference as follows: 
```
cd RAGnRoll/agent_inference
```
```py
from agent import Agent
from tools import SearchTool, SearchToolWithinDocs, parse


ragnroll_name= "AttriRAG/ragnroll_llama2_13b"
config = PeftConfig.from_pretrained(ragnroll_name, load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
 "meta-llama/Llama-2-13b-chat-hf", device_map="auto" 
)
tokenizer = AutoTokenizer.from_pretrained() 
model = PeftModel.from_pretrained(model, ragnroll_name, device_map="auto")

model = model.merge_and_unload()

retireval_start_token = "[SEARCH]"
retireval_end_token = "[/SEARCH]"

kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}

question = 'Can you prevent traumatic brain injury?'
tools = [
        SearchTool(
            name="search",
            index="miracl-v1.0-en",
            start_token=retireval_start_token,
            end_token=retireval_end_token,
            reranker="GTR",
        )
    ]

agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=2,
        use_tools=True,
        num_docs=3
    )
docs_text, scores,answer = agent.generate(", **kwargs)
parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
output=""
if parsed_answers:
    output = " ".join(parsed_answers)
print(output)

```


## Contact
If you have questions, please open an issue mentioning @hanane-djeddal or send an email to hanane.djeddal[at]irit.fr.
