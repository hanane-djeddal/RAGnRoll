from tqdm import tqdm
import time
import pandas as pd
import json
import argparse
import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text
import argparse
import os
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from transformers import set_seed

global RAGAGENT_MODEL_NAME,  TRAINING_CORPUS
RAGAGENT_MODEL_NAME ="AttriRAG/ragnroll_llama2_13b"
TRAINING_CORPUS =  "HAGRID"

def parse(message, begin, end):
    """
    This function parses a message to find all substrings between
    a given begin_token and end_token.

    Args:
        message: The message to be parsed.
        begin_token: The starting token (inclusive).
        end_token: The ending token (inclusive).

    Returns:
        A list of all substrings found between the begin_token and end_token.
    """
    substrings = []
    start_index = 0
    while True:
        begin_loc = message.find(begin, start_index)
        if begin_loc == -1:
            break
        end_loc = message.find(end, begin_loc + len(begin))
        if end_loc == -1:
            end_loc = len(message)
        offset = 0
        if message[begin_loc + len(begin)] == ":":
            offset = 1
        substring = message[begin_loc + len(begin) + offset : end_loc]
        substrings.append(substring)
        start_index = end_loc + len(end)
    return substrings


from agent import Agent
from tools import SearchTool, SearchToolWithinDocs
import re

def reposition_period_after_citation(text):
    result = re.sub(r'\.\s*((\[[^\]]+\])+)(?!\S)', r' \1.', text)
    return result

def main():
    global RAGAGENT_MODEL_NAME,  TRAINING_CORPUS, model_result_file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--nb_rounds", type=int, default=4)
    parser.add_argument("--nb_docs", type=int, default=3)
    parser.add_argument("--resume_from_file", type=str, default=None)
    parser.add_argument("--ranker", type=str, default="GTR", choices=["GTR","MonoT5"])
    parser.add_argument("--retrieval", action="store_true")
    parser.add_argument("--inference_variant", type=str, default=None, choices=["normal", "without_query"])
    parser.add_argument(
        "--validating_code",
        action="store_true",
        help="Run short iteration to test code",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Results are written to outputdir with data suffix",
    ) 
    parser.add_argument(
        "--ragnroll_model_name",
        default= RAGAGENT_MODEL_NAME,
        type = str,
        help="Tested Model",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Tag to add in the results file",
    )
    parser.add_argument(
        "--training_corpus",
        type=str,
        default =TRAINING_CORPUS,
        help="Corpus used for training",
    )
    parser.add_argument(
        "--add_instruction",
        action="store_true",
        help="Adds instruction to prompt",
    )
    parser.add_argument(
        "--retrieve_with_answer",
        action="store_true",
        help="use generated answers for retrieval",
    )
    parser.add_argument(
        "--gen_config",
        type=str,
        default =None,
        help="Config to pass to the generator",
    )
    parser.add_argument(
        "--diverse_query_only",
        action="store_true",
        help="Only apply diversity config to query",
    )
    parser.add_argument(
        "--add_user_query",
        action="store_true",
        help="adding user query to the subqueries",
    )   
    parser.add_argument(
        "--retrieve_once",
        action="store_true",
        help="only retrieve once and use same docs in each round",
    )   
    parser.add_argument(
        "--startindex",
        type=int,
        default=0,
        help="Index to start iterations",
    )
    parser.add_argument(
        "--stopindex",
        type=int,
        default=None,
        help="Index to stop iterations",
    )
    args = parser.parse_args()
    #################
    # Logging params
    #################
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")
        
    results_path ="results/llm-agent/llama13/" #os.environ['WORK'] +"/ 
    results_dir = args.output_dir if args.output_dir else results_path
    RAGAGENT_MODEL_NAME = args.ragnroll_model_name
    TRAINING_CORPUS = args.training_corpus

    logger.info(f"Loading Language model...{RAGAGENT_MODEL_NAME}")
    logger.info(f"Retrieval : BM25 + {args.ranker}")
    logger.info(f"Appending user query to subqueries...{args.add_user_query}")
    tag = args.ragnroll_model_name.split('/')[-1]
    tag = tag + args.tag if args.tag else  tag
    tag = tag + args.ranker if args.ranker else tag
    tag = tag+"_instruction_prompt" if args.add_instruction else tag
    tag = tag +"_using_answer_for_retrieval_" if args.retrieve_with_answer else tag
    tag = tag +"_appendinguseruquery" if args.add_user_query else tag
    tag = tag +"_retrieve_once" if args.retrieve_once else tag
    logger.info(f"Used Tag {tag}")
    logger.info(f"Inference without query : {args.inference_variant}")
    if args.validating_code:
        logger.info(f"Only running two iterations to test")
        tag = tag + "code_validation"

    if args.gen_config:
        kwargs = json.loads(args.gen_config)
    else:
        kwargs = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
    logger.info(f"Generator config...{kwargs} to query only {args.diverse_query_only}")
    SEED = 42
    set_seed(SEED)
    dataset_name = "ALCE" if args.query_file else "HAGRID"  # "HAGRID"   "ALCE"
    input_file = None  


    config = PeftConfig.from_pretrained(RAGAGENT_MODEL_NAME, load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-13b-chat-hf", device_map="auto" 
    )
    tokenizer = AutoTokenizer.from_pretrained(RAGAGENT_MODEL_NAME) 
    model = PeftModel.from_pretrained(model, RAGAGENT_MODEL_NAME, device_map="auto")

    model = model.merge_and_unload()

    start = time.time()
    if args.inference_variant == "without_query":
        retireval_start_token ="[ANSWER]"
        retireval_end_token = "[/ANSWER]"
    elif args.retrieve_with_answer:
        retireval_start_token ="[ANSWER]"
        retireval_end_token = "[/SEARCH]"
    else:
        retireval_start_token = "[SEARCH]"
        retireval_end_token = "[/SEARCH]"
    tools = [
        SearchTool(
            name="search",
            index="miracl-v1.0-en",
            start_token=retireval_start_token,
            end_token=retireval_end_token,
            reranker=args.ranker,
        )
    ]
    #manual_stop_words= {"13B":[66028, 44645,   933],"7B":[28792, 28748,  1151, 17046, 3328],"3B":[16,  2354, 20756,    62], "3Badj": [32871,  2354, 20756,  7082]}
    agent = Agent(
        model=model,
        tokenizer=tokenizer,
        tools=tools,
        rounds=args.nb_rounds*2,
        use_tools=True, #args.retrieval,
        num_docs=args.nb_docs,
        train_corpus=TRAINING_CORPUS,
        adjusted= False, #True,
        model_params = "7B",
        manual_stop_words= False,
        without_query_gen = args.inference_variant == "without_query",
        add_instruction = args.add_instruction,
        diverse_query_only = args.diverse_query_only,
        add_user_query = args.add_user_query,
        retrieve_once = args.retrieve_once,
    )
    print("Adjusted", False)

    if dataset_name == "HAGRID":
        dataset = datasets.load_dataset("miracl/hagrid", split="dev")
        #dataset = datasets.load_from_disk(os.environ["WORK"] + "/hagrid-dev") 
        query_column = "query"
    else:
        with open(args.query_file) as f:
            dataset = json.load(f)
        query_column = "question"

    start_idx = args.startindex
    if args.resume_from_file:
        with open(args.resume_from_file) as f:
            data_with_config = json.load(f)
        results = data_with_config["data"]
        start_idx = len(results)
        print("Resuming test from file:",args.resume_from_file)
        print("Starting Iteration:",start_idx)
    else:
        results = []
    for nb_row, row in enumerate(tqdm(dataset)):
        if nb_row < start_idx:
            continue 
        if args.validating_code and nb_row == 2:
            break
        docs_text, scores,answer = agent.generate(row[query_column], **kwargs)
        parsed_answers = parse(answer, "[ANSWER]", "[/ANSWER]")
        output = None
        if parsed_answers:
            output = " ".join(parsed_answers)
        else:
            position = answer.find("[ANSWER]")

            if position != -1:
                start = position + len("[ANSWER]")
                output = answer[start:]

        if output is None:
            output = ""
        if TRAINING_CORPUS == "HAGRID":
             docs = []
             docids = []
             for statement_docs in docs_text:
                  for doc in statement_docs:
                       docid = doc["docid"]
                       if docid not in docids:
                            docs.append(doc)
                            docids.append(docid)
                  for i in range(len(docs)):
                       if docs and docs[i]["docid"] in output:
                            output = output.replace(docs[i]["docid"],str(i+1))
        output = reposition_period_after_citation(output)
        if dataset_name == "HAGRID":
            annotations = []
            for a in row["answers"]:
                annotations.append({"long_answer": a["answer"]})
            if len(annotations) < 2:
                annotations.append({"long_answer": a["answer"]})
            results.append(
                {
                    "question": row[query_column],
                    "generated_text": answer,
                    "output": output,
                    "docs": docs,
                    "gold_truth": row["answers"],
                    "gold_quotes": row["quotes"],
                    "answer": row["answers"][0]["answer"],
                    "annotations": annotations,
                }
            )
        else:
            results.append(
                {
                    "question": row[query_column],
                    "generated_text": answer,
                    "output": output,
                    "docs": docs,
                    "answer": row["answer"],
                    "annotations": row["annotations"],
                }
            )
        if (nb_row+1) % 150 == 0:
            results_df = {"data": results}
            results_file = results_dir + str(args.startindex)+"-"+"intr_test"+dataset_name+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json"  
            print("Saving intermediate results to file:", results_file)
            with open(results_file, "w") as writer:
                json.dump(results_df, writer)
        if args.stopindex and args.stopindex == nb_row:
            logger.info(f"Stoping after {args.stopindex - args.startindex + 1} iterartion, index {args.stopindex} finished")
            break
    end = time.time()

    execution_time = (end - start) / 60
    results_df = {"data": results, "params":vars(args)}
    if args.stopindex:
        results_file = results_dir+str(args.startindex)+"-"+str(args.stopindex)+"intr_test"+dataset_name+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json" 
        with open(results_file, "w") as writer:
            json.dump(results_df, writer)
    else:
        results_file = results_dir+"all_test"+dataset_name+tag+"_"+str(args.nb_rounds)+"rounds_"+str(args.nb_docs)+"docs.json" 
        with open(results_file, "w") as writer:
            json.dump(results_df, writer)

    print("Result file:", results_file)
    print("execution_time:", execution_time)





if __name__ == "__main__":
    main()
