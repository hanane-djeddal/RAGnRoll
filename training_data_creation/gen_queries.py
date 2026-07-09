import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from tqdm import tqdm
import re
import logging
import os
import nltk 
import torch.nn.functional as F
import sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from copy import deepcopy

#from bert_score import score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'
ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from tools.citation_tools import remove_citations

global model, tokenizer
model, tokenizer = None, None

global selection_model, selection_tokenizer
selection_model, selection_tokenizer = None, None

MAX_MODEL_LEN = 2048 
MAX_NEW_TOKENS = 512 # desired maximum output length

# Calculate the maximum allowed prompt length
MAX_PROMPT_FOR_TRUNCATION = MAX_MODEL_LEN - MAX_NEW_TOKENS


user_prompt_with_query= "GENERAL QUERY: {query} \n\n SENTENCE: {segment} \n\n SUGGESTED QUERIES:"
user_prompt_without_query="SENTENCE: {segment} \n\n SUGGESTED QUERIES:"
user_prompt_withonly_query="QUERY: {query} \n\n SUGGESTED QUERIES:"

query_gen_prompt= {
        "seg_with_query":{
            "system": "Given a general query and a sentence with relevant information, convert the given sentence into one or more specific subqueries that can be used to retrieve the sentence from a longer passage. You can leverage the information mentioned in the sentence to generate subqueries that closely match the sentence's content. Only generate your suggested subqueries without explanation. The subqueries should be independent and decontextualized. The maximum number of subqueries is {nb_queries}",
            "user":user_prompt_with_query
        },
        "seg_without_query":{
            "system": "Convert the given sentence into one or more specific queries that can be used to retrieve the sentence from a longer passage. The queries should closely match the sentence's content. Only generate your suggested queries without explanation. The queries should be independent and decontextualized. The maximum number of queries is {nb_queries}",
            "user":user_prompt_without_query
        },
        "snippet_with_query":{
            "system": "Given a general query and a sentence containing relevant information, generate multiple specific subqueries that focus on the key details of the sentence. The subqueries should closely match the sentence's content. Only generate your suggested subqueries without explanation. The subqueries should be independent and decontextualized. The maximum number of subqueries is {nb_queries}",
            "user":user_prompt_with_query
        },
        "snippet_without_query": {
            "system": "Convert the given sentence into one or more specific queries that focus on the key details of the sentence. The queries should closely match the sentence's content. Only generate your suggested queries without explanation. The queries should be independent and decontextualized. The maximum number of queries is {nb_queries}",
            "user":user_prompt_without_query
        },
        "parametric_knowledge": {
            "system": "Given a query, generate a set of specific subqueries that break the query into key aspects that should be considered when answering it. These subqueries should be specific and not too general. Only generate your suggested subqueries without explanation. The subqueries should be independent and decontextualized. The maximum number of subqueries is {nb_queries}",
            "user":user_prompt_withonly_query
        }
    }
fewshot_examples = {
    "segment_based":[
        {
            "query":"Why is Menopause important?",
            "sentence":"Menopause can be divided into early and late transition periods, also known as perimenopause and postmenopause.",
            "generated_queries": ["What are the early and late transition periods of menopause?", "What are the early and late transition periods of menopause called?"]
            
        },
        {
            "query":"Do other animals have different blood types like humans?",
            "sentence":"Various species require different levels of testing to ensure a compatible match.",
            "generated_queries": ["why do various species require different levels of testing?", "What levels of testing are required to ensure blood type compatibility in different species?"]
            
        },
        
        {
            "query":"What is the longest wall in the world?",
            "sentence":"It stretches over 13,000 miles and was built to protect against invasions.",
            "generated_queries": ["How many miles does the Great Wall of china stretch and why was it built?", "How long is the Great Wall of China and what was its purpose?"]

        },
        {
            "query":"Do other animals have different blood types like humans?",
            "sentence":"For example, cats have 3 known blood types, cattle have 11, dogs have 12, pigs 16, and horses have 34.",
            "generated_queries": ["How many blood types do cats, dogs, and horses have?", "Examples of blood types in animals"]
            
        },
        {
            "query":"Who was Marie Curie?",
            "sentence":"Marie Curie won two Nobel Prizes, one in Physics and another in Chemistry, for her work on radioactivity",
            "generated_queries": ["What did Marie Curie win two Nobel Prizes for and what was her work on?",  "Which Nobel Prizes did Marie Curie win and for what?"]
        },
    ],
    "knowledge_based":[
         {
            "query":" What is the deepest point in the Pacific Ocean?",
            "generated_queries": ["Where is the Mariana Trench located?","What is the maximum depth of the Mariana Trench?","What scientific expeditions have explored the Mariana Trench?"]
            
        },
        {
            "query":"Why is menopause important?",
            "generated_queries": ["How does menopause affect hormone levels in women?", "What are the symptoms associated with menopause?","How does menopause impact long-term health, such as bone density and heart health?"]
            
        },
        
        {
            "query":"What is the longest wall in the world?",
            "generated_queries": ["What is the length of the Great Wall of China?","What is the purpose of the Great Wall of China?", "How long did it take to build the Great Wall of China?"]

        },
        {
            "query":"Do other animals have different blood types like humans?",
            "generated_queries": ["Which animals have distinct blood types?", "How do blood type systems in animals compare to those in humans?","Why is blood type compatibility important in veterinary medicine?"]
            
        },

    ]
}
def parse_generated_queries(answer: str):
    answer = re.sub(r"\d+\.", "", answer)
    answer = re.sub(r"- ", "", answer)
    quries = answer.split("\n")
    quries = list(filter(None, quries))
    return quries



def prepare_prompt(segment=None, prompt=None, query=None, fewshot_examples=[],model_name = "meta-llama/Llama-2-13b-chat-hf",nb_queries=2):
    global model, tokenizer
    if model is None:
        logger.info(f"Loading Language model...{model_name}")
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name) 

        model = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
            dtype='bfloat16',
            enforce_eager=False,
            # for LORA
            trust_remote_code=True,
            enable_lora=True,
        )

    user_prompt = prompt["user"]
    if query:
        user_prompt = prompt["user"].replace("{query}", query)
    if segment:
        user_prompt = user_prompt.replace("{segment}", segment)

    system_prompt = re.sub("\{nb_queries\}", str(nb_queries), prompt["system"])
    if fewshot_examples:
        formatted_examples = []
        selected_examples = fewshot_examples
        # selected_examples.reverse()
        for example in selected_examples:
            if  "sentence" in example.keys(): 
                if query and "query" in example.keys():
                    formatted_example = (
                        "GENERAL QUERY: " + example["query"] + "\n\nSENTENCE: " + example["sentence"] + " \n\nSUGGESTED QUERIES: \n"
                    )
                else:
                    formatted_example = (
                        "SENTENCE: " + example["sentence"] + " \n\nSUGGESTED QUERIES: \n"
                    )
            else:
                if query and "query" in example.keys():
                    formatted_example = (
                        "QUERY: " + example["query"] + " \n\nSUGGESTED QUERIES: \n"
                    )
                else:
                    logger.warning(f"Fewshot example does not contain a sentence nor query: {example}")

            if len(example["generated_queries"]) >= nb_queries:
                selected_queries = example["generated_queries"][:nb_queries]
            else:
                selected_queries = example["generated_queries"]
            # selected_queries.reverse()
            for i in range(len(selected_queries)):
                formatted_example = (
                    formatted_example + str(i + 1) + ". " + selected_queries[i] + " \n"
                )

            formatted_examples.append(formatted_example)
        complete_user_prompt = "\n\n".join(formatted_examples) + "\n\n" + user_prompt
        user_prompt = user_prompt

    input_text = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", 
            "content": user_prompt}
        ]
    inputs = tokenizer.apply_chat_template(
            input_text, add_generation_prompt=True, return_tensors="pt",tokenize=False
        )
    return inputs

def generate_vllm(all_inputs,model_name = "meta-llama/Llama-2-13b-chat-hf"):
    global model, tokenizer
    if model is None:
        logger.info(f"Loading Language model...{model_name}")
        # Load the model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForCausalLM.from_pretrained(model_name,output_attentions=True) 

        model = LLM(
            model=model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.9,
            max_model_len=MAX_MODEL_LEN,
            dtype='bfloat16',
            enforce_eager=False,
            # for LORA
            trust_remote_code=True,
            enable_lora=True,
        )


    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=MAX_NEW_TOKENS,
        truncate_prompt_tokens=MAX_PROMPT_FOR_TRUNCATION,
        #stop_token_ids=[151643, 151645]
    )
    all_outputs=model.generate(all_inputs, sampling_params) # [inputs]

    all_answers=[]
    for idx_output, output in enumerate(all_outputs):

        answer = output.outputs[0].text

        keyword = "[/INST]"
        index_kw = answer.rfind(keyword)
        filetred_answer = answer
        if index_kw != -1:
            filetred_answer = answer[index_kw+len(keyword)+1:]

        suggest_queries_index = filetred_answer.rfind("SUGGESTED QUERIES:")
        if suggest_queries_index != -1:
            filetred_answer = filetred_answer[suggest_queries_index+len("SUGGESTED QUERIES:")+1:]
        else:
            suggest_queries_index = filetred_answer.rfind("Suggested Subqueries:")
            if suggest_queries_index !=-1:
                filetred_answer = filetred_answer[suggest_queries_index+len("Suggested Subqueries:")+1:]
            else:
                suggest_queries_index = filetred_answer.rfind("\n\n")
                if suggest_queries_index !=-1:
                    filetred_answer = filetred_answer[suggest_queries_index+len("\n\n")+1:]

        parsed_queries = parse_generated_queries(filetred_answer)
        logger.info(f"Generated queries: {parsed_queries}")

        all_answers.extend(parsed_queries)
    return all_answers



def generate_query(segment=None, prompt=None, query=None, fewshot_examples=[],model_name = "meta-llama/Llama-2-13b-chat-hf",nb_queries=2,):
    global model, tokenizer
    if model is None:
        logger.info(f"Loading Language model...{model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name) 

        model = AutoModelForCausalLM.from_pretrained(model_name,output_attentions=True)
    user_prompt = prompt["user"]
    if query:
        user_prompt = prompt["user"].replace("{query}", query)
    if segment:
        user_prompt = user_prompt.replace("{segment}", segment)

    system_prompt = re.sub("\{nb_queries\}", str(nb_queries), prompt["system"])
    if fewshot_examples:
        formatted_examples = []
        selected_examples = fewshot_examples
        for example in selected_examples:
            if  "sentence" in example.keys(): 
                if query and "query" in example.keys():
                    formatted_example = (
                        "GENERAL QUERY: " + example["query"] + "\n\nSENTENCE: " + example["sentence"] + " \n\nSUGGESTED QUERIES: \n"
                    )
                else:
                    formatted_example = (
                        "SENTENCE: " + example["sentence"] + " \n\nSUGGESTED QUERIES: \n"
                    )
            else:
                if query and "query" in example.keys():
                    formatted_example = (
                        "QUERY: " + example["query"] + " \n\nSUGGESTED QUERIES: \n"
                    )
                else:
                    logger.warning(f"Fewshot example does not contain a sentence nor query: {example}")

            if len(example["generated_queries"]) >= nb_queries:
                selected_queries = example["generated_queries"][:nb_queries]
            else:
                selected_queries = example["generated_queries"]
            for i in range(len(selected_queries)):
                formatted_example = (
                    formatted_example + str(i + 1) + ". " + selected_queries[i] + " \n"
                )

            formatted_examples.append(formatted_example)
        complete_user_prompt = "\n\n".join(formatted_examples) + "\n\n" + user_prompt
        user_prompt = complete_user_prompt


    input_text = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", 
         "content": user_prompt}
    ]
    inputs = tokenizer.apply_chat_template(
            input_text, add_generation_prompt=True, return_tensors="pt"
        )
    
    tokens = model.generate(
            inputs.to(model.device),
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    tokens = tokens.sequences 
    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    keyword = "[/INST]"
    index_kw = answer.rfind(keyword)
    filetred_answer = answer
    if index_kw != -1:
        filetred_answer = answer[index_kw+len(keyword)+1:]

    suggest_queries_index = filetred_answer.rfind("SUGGESTED QUERIES:")
    if suggest_queries_index != -1:
        filetred_answer = filetred_answer[suggest_queries_index+len("SUGGESTED QUERIES:")+1:]
    else:
        suggest_queries_index = filetred_answer.rfind("Suggested Subqueries:")
        if suggest_queries_index !=-1:
            filetred_answer = filetred_answer[suggest_queries_index+len("Suggested Subqueries:")+1:]
        else:
            suggest_queries_index = filetred_answer.rfind("\n\n")
            if suggest_queries_index !=-1:
                filetred_answer = filetred_answer[suggest_queries_index+len("\n\n")+1:]

    parsed_queries = parse_generated_queries(filetred_answer)
    logger.info(f"Generated queries: {parsed_queries}")
    return parsed_queries


def compute_log_prob(text,tokenizer,model):
    """Compute log probability of a given text"""
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)
    token_ids = inputs["input_ids"].squeeze(0)  # Ensure batch dimension is removed

    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)

    # Ensure token_ids is 2D: (1, seq_len)
    token_ids = token_ids.unsqueeze(0)

    # Extract probability of actual tokens
    token_probs = probs[:, :-1, :].gather(2, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # Compute total log probability
    log_prob = torch.log(token_probs).sum().item()
    return log_prob

def query_selection(query, paragraph,selection_model_name="meta-llama/Llama-2-13b-chat-hf"):
    """
    P(s | q) 
    """
    global selection_model, selection_tokenizer
    if selection_model is None:
        logger.info(f"Loading Language model...{selection_model_name}")
        # Load the model and tokenizer
        selection_tokenizer = AutoTokenizer.from_pretrained(selection_model_name)

        selection_model = AutoModelForCausalLM.from_pretrained(selection_model_name)


    # Tokenize input
    sentences = nltk.sent_tokenize(paragraph)
    # Compute log probabilities
    log_p_q = compute_log_prob(query,selection_tokenizer,selection_model)  # P(q)

    segments_scores = []
    for i, sentence in enumerate(sentences):
        log_p_qs = compute_log_prob(query + " " + sentence,selection_tokenizer,selection_model)  # P(q + s) #+" "+answer
        log_p_s_given_q = log_p_qs - log_p_q  # P(s | q)

        segments_scores.append({"segment":sentence,"score":log_p_s_given_q,"indice":i})

    sorted_segments = sorted(segments_scores, key=lambda x: x["score"], reverse=True)
    return sorted_segments

def best_query(queries, relevant_segments, passages, selection_model_name="meta-llama/Llama-2-13b-chat-hf"):
    deduplicated_queries = list(set(queries))
    queries = deduplicated_queries
    scored_queries=[]
    for rlv_segment in relevant_segments:
        for query in queries:
            if query not in ["","ssistant"]:
                sorted_segments = query_selection(query, passages[rlv_segment["pssg_indice"]]["text"],selection_model_name)
                
                for seg in sorted_segments:
                    if seg["segment"] == rlv_segment["segment"]['segment']:
                        scored_queries.append({
                            "query": query,
                            "relevant_seg_score": seg["score"],
                            "weighed_seg_score": seg["score"] * rlv_segment["segment"]["log_probabilities"]
                        })
                    break
    ordered_queries = sorted(scored_queries, key=lambda x: x["relevant_seg_score"], reverse=True)
    return ordered_queries

def assign_query_to_statement(queries, statements):
    updated_statement = []
    for statement in statements:
       best_match = None
       best_score = -1
       for q in queries:
           P, R, F1 = score([q], [statement["text"]], lang="en", verbose=True)
           if F1.item() > best_score:
               best_score = F1.item()
               best_match = q

       if best_match:
          statement["subqueries"] = best_match
       updated_statement.append(statement)

    return updated_statement

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name for generating queries"
    ) 
    parser.add_argument(
        "--segmented_dataset", type=str, default=None, help="Dataset (.json) with segmented passages", required=True,
    )
    parser.add_argument(
        "--select_model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="Model name for compute probabilities and select the best query"
    ) 
    parser.add_argument(
        "--query_gen_prompt", type=str, default="seg_without_query", help="prompt for query generation"
    ) 
    parser.add_argument(
        "--results_folder", type=str, default="results/query_gen/", help="Results folder"
    )
    parser.add_argument(
        "--query_gen_source", type=str, default="segment",
        choices=["segment", "segment_concatenation","statement","all","parametric_knowledge",None], help="What is provided in the prompt to generate the queries"
    )
    parser.add_argument(
        "--query_selection", type=str, default="seg_proba",
        choices=["seg_proba", "first","retrieval",None], help="Best query selection method"
    )
    parser.add_argument(
        "--validating_code", action="store_true",  help="validate_code"
    )
    parser.add_argument(
        "--attributable_only", action="store_true", help="Only attributable answers"
    )
    parser.add_argument(
        "--query_gen_without_selection", action="store_true", help="Only generate subqueries without choosing/ranking"
    )
    parser.add_argument(
        "--resume_from_file", type=str, default=None, help="Resume from file"
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
    parser.add_argument(
        "--original_split", action="store_true", help="Statement using original split"
    )
    parser.add_argument(
        "--vllm", action="store_true", help="use Vllm"
    )
    parser.add_argument(
        "--missing_values", action="store_true", help="use Vllm"
    )

    args = parser.parse_args()

    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)} - {parser.get_default(arg)}")

    with open(args.segmented_dataset, 'r') as file:
        dataset = json.load(file)
        if isinstance(dataset, dict):
            dataset = dataset["data"]
    updated_dataset = []

    prompt_model_name =args.prompt_model_name.split('/')[-1]
    selection_model_name = args.select_model_name.split('/')[-1]
    query_gen_source = "_gen_from_"+args.query_gen_source 
    attributed= "_attributable_only" if args.attributable_only else "all"
    code_validation="_validating_code" if args.validating_code else ''
    og_slit= "_original_split" if args.original_split else ''
    vllm= "_vllm" if args.vllm else ''
    missing_value='missing_values' if args.missing_values else ''
    results_file =  "subquery_gen_"+args.query_gen_prompt+query_gen_source+"_gen_"+prompt_model_name+"_select_"+selection_model_name+attributed+code_validation+og_slit+vllm+missing_value+".json"

    start_idx = args.startindex 
    if args.resume_from_file:
        with open(args.resume_from_file) as f:
            data_with_config = json.load(f)
        updated_dataset = data_with_config["data"]
        start_idx = len(updated_dataset)
        print("Resuming test from file:",args.resume_from_file)
    print("Starting Iteration:",start_idx)

    for idx, row in enumerate(tqdm(dataset)):
        if idx < start_idx:
            continue
        if args.validating_code and  idx == 2:
             break
        
        all_inputs=[]
        if args.query_gen_source == "parametric_knowledge":
            if args.vllm:
                all_inputs.append(prepare_prompt("",query_gen_prompt["parametric_knowledge"],row["query"],fewshot_examples["knowledge_based"][:3],model_name=args.prompt_model_name, nb_queries=len(row["answers"][0]["sentences"])))
                gen_queries=generate_vllm(all_inputs)
            else:
                gen_queries = generate_query("",query_gen_prompt["parametric_knowledge"],row["query"],fewshot_examples["knowledge_based"][:3],model_name=args.prompt_model_name, nb_queries=len(row["answers"][0]["sentences"]))
           
            row["answers"][0]["updated_statement"] =  assign_query_to_statement(gen_queries, row["answers"][0]["sentences"] )
            updated_dataset.append(row)
            continue
        selected_answer_indice = None
        if args.attributable_only:
            for i in range(len(row["answers"])):
                if row["answers"][i]["attributable"]:
                    if selected_answer_indice:
                        if len(row["answers"][i]> row["answers"][selected_answer_indice]):
                            selected_answer_indice = i
                    else:
                        selected_answer_indice = i 
            logger.info(f"Choosing answer {selected_answer_indice}")
        queries_per_statement= [] # list of queries per statement
        if not args.attributable_only or (args.attributable_only and selected_answer_indice!= None):
            if not selected_answer_indice:
                selected_answer_indice = 0
            for statement in row["answers"][selected_answer_indice]['segments']:
                if args.missing_values and "subqueries" in statement.keys() and len(statement["subqueries"]):
                    continue
                logger.info(f"----Statement: {statement['statement']}")
               
                ### saving passages and relevant segments per statement to compute proba
                relevant_segments =[]
                passages = []
                all_inputs=[]
                if args.query_gen_source:
                    for passage in statement["cited_doc"][:4]: 
                        passages.append(row["quotes"][passage["source"]-1])

                        if args.query_gen_source in ["segment", "segment_concatenation", "all"] :
                            logger.info(f"Generating queries per segment")
                            for segment in passage["segments"][:2]: 
                                relevant_segments.append({"pssg_indice": len(passages)-1, "segment":segment})
                                if args.query_gen_source in ["segment", "all"] :
                                    if args.vllm:
                                        all_inputs.append(prepare_prompt(segment["segment"],query_gen_prompt["seg_without_query"],None,fewshot_examples["segment_based"][:3],model_name=args.prompt_model_name))
                                    else:
                                        gen_queries= generate_query(segment["segment"],query_gen_prompt["seg_without_query"],None,fewshot_examples["segment_based"][:3],model_name=args.prompt_model_name)
                                        queries_per_statement.extend(gen_queries)

                    if args.query_gen_source in ["segment_concatenation", "all"] and len(relevant_segments) :
                        logger.info(f"Generating queries with the concatenation of relevant segments")
                        ### generate queries from concatenation of all segments
                        concat_segments =  " ".join([seg["segment"]["segment"] for seg in relevant_segments])
                        if args.vllm:
                            all_inputs.append(prepare_prompt(concat_segments,query_gen_prompt["seg_without_query"],None,fewshot_examples["segment_based"][:3],model_name=args.prompt_model_name))
                        else:
                            gen_queries = generate_query(concat_segments,query_gen_prompt["seg_without_query"],None,fewshot_examples["segment_based"][:3],model_name=args.prompt_model_name)
                            queries_per_statement.extend(gen_queries)
                    if args.query_gen_source in ["statement", "all"] or (args.missing_values and len(relevant_segments)==0):
                        logger.info(f"Generating queries with statement")
                        ### generate queries from statement
                        if args.vllm:
                            all_inputs.append(prepare_prompt(concat_segments,query_gen_prompt["seg_without_query"],None,fewshot_examples["segment_based"][:3],model_name=args.prompt_model_name))
                        else:
                            gen_queries = generate_query(remove_citations(statement['statement']["text"]),query_gen_prompt["snippet_without_query"],None,fewshot_examples["segment_based"][:3],model_name=args.prompt_model_name)
                            queries_per_statement.extend(gen_queries)

                    if args.vllm:
                        queries_per_statement=generate_vllm(all_inputs)
                    generated_queries= deepcopy(queries_per_statement)
                    if args.query_selection == "seg_proba":
                        logger.info(f"Choosing subquery")
                        query = best_query(queries_per_statement,relevant_segments, passages,args.select_model_name)
                    else:
                        query = queries_per_statement ### returning all list queries_per_statement[0]
                    if len(query):
                        statement["subqueries"] = query
                        logger.info(f"ranked subqueries:{query}")
                    else:
                        statement["subqueries"] = generated_queries
                        logger.info(f"ranked subqueries:{generated_queries}")
            updated_dataset.append(row)
            if (idx+1) % 80 == 0:
                inter_results_file = "iter_"+results_file
                inter_results_file = os.path.join(args.results_folder, inter_results_file)
                new_set={"data":updated_dataset, "params":vars(args)}
                logger.info(f"Saving intermediate results to {inter_results_file}")
                with open(inter_results_file, "w") as writer:
                    json.dump(new_set, writer)
        if args.stopindex and args.stopindex == idx:
            logger.info(f"Stoping after {args.stopindex - args.startindex + 1} iterartion, index {args.stopindex} finished")
            break
    if args.stopindex:
        assert (args.stopindex == idx)
        inter_results_file = str(args.startindex)+"-"+str(args.stopindex)+"iter_"+results_file
        results_file = inter_results_file
    
    results_file = os.path.join(args.results_folder, results_file)
    new_set={"data":updated_dataset, "params":vars(args)}
    logger.info(f"Saving results to {results_file}")
    with open(results_file, 'w') as file:
        json.dump(new_set, file)

main()
