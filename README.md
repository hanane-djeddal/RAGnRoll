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

## Inference 

```
cd RAGnRoll/agent_inference
```
```
 python run_inference.py --nb_rounds 8 --ragnroll_model_name  AttriRAG/ragnroll_llama2_13b --ranker  GTR 
```
#### HAGRID 
By default it runs the test on HAGRID by calling the [Huggingface HAGRID Dataset](https://huggingface.co/datasets/miracl/hagrid)
#### ASQA/CORAL/Custom
You can also pass a custom data file to run inference. For **ASQA**, you can download the data files from the ALCE Framework [here](https://github.com/princeton-nlp/ALCE#data) and call the inference as follow:
```
cd RAGnRoll/agent_inference
```
```
 python run_inference.py --nb_rounds 8  --query_file ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  AttriRAG/ragnroll_llama2_13b --ranker  GTR
```
For CORAL and any Custom dataset, the same script can be used by passing an input file with a similar structure with top 100 (or K>nb_docs) documents.

### Compact/Expanded Variants & other custom runs:
By default the code runs the Expanded version, for compact variant (without intermediate subqueries) you can add the argument ```--inference_variant without_query```
```
 python run_inference.py --nb_rounds 8  --query_file ALCE-data/asqa_eval_gtr_top100.json  --ragnroll_model_name  AttriRAG/ragnroll_llama2_13b --ranker  GTR --inference_variant without_query
```

You can also use the intermediate answers for retrieval with  ```--retrieve_with_answer```, add system instruction with ```--add_instruction``` and append intermediate subqueries with user query for more context with ```--add_user_query```

## Training
You can use the synthetic training dataset on Huggingface [Here](AttriRAG/ragnroll_train_data_hagrid_seg_proba), to train any backbone LLM. We follow the [AlignmentHandbook](https://github.com/huggingface/alignment-handbook) to train our models. Follow the instructions in the [Git](https://github.com/huggingface/alignment-handbook#installation-instructions) to setup training environment. You can change the training reciepts and choice of trained model in training/recipes/llama-2chat-hf/config_qlora.yml.
```
cd RAGnRoll/training
```
```
 python scripts/train_sft_agent.py recipes/llama-2-chat-hf/config_qlora.yaml --load_in_4bit=true 
```

## Synthetic Data Generation
Training Data is already available on HuggingFace [RAGnRoll Data](AttriRAG/ragnroll_train_data_hagrid_seg_proba). The code for creating the synthetic data is also provided in this repository under ```training_data_creation```. 
#### Relevant Segements
First, to identify and extract relevant segments in the gold answers, you can run:

```
python training_data_creation/segment_extraction.py --method mixed --attributable_only
```
Different methods to identify relevant segments are available with the argument ```--method```, mainly: ["proba", "relevance","nli","mixed"]. The paper explores mainly the propabilistic approach  ```'proba' ``` set by defealt.
For segmentation, also different possibilities using ```--segmentation``` such as "sentence", "cumulated_sentences","removing_sent". By default we consider 'sentence'.
The probabilities are estimated using ```meta-llama/Llama-2-13b-chat-hf`` `which can be set to a different LLM using ```--model_name```

The result file is provided in  ```results/segment_extract_proba_sentence_Llama-2-13b-chat-hf_original_split_attributable_only.json ```

#### Subquery Generation
Once the relevant segments are identified (or using the provided result file  ```results/segment_extract_proba_sentence_Llama-2-13b-chat-hf_original_split_attributable_only.json ```) you can run the subquery generation. 

```
python training_data_creation/gen_queries.py   --vllm --original_split --attributable_only --results_folder  results/query_gen/ --segmented_dataset results/segmentation/tested_methods/segment_extract_proba_sentence_Llama-2-13b-chat-hf_original_split_attributable_only.json
```

## Contact
If you have questions, please open an issue mentioning @hanane-djeddal or send an email to hanane.djeddal[at]irit.fr.
