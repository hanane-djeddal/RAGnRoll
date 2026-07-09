import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os
import sys
import json
import argparse
from nltk import sent_tokenize
from tqdm import tqdm
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#os.environ['HF_HOME'] = os.environ['WORK'] + '/.cache/huggingface'

ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
sys.path.append(ROOT_PATH)

from tools.citation_tools import _run_nli_autoais, split_text_by_citations,remove_citations, get_source_from_text, get_statement_by_sentence


global proba_model, proba_tokenizer
proba_model, proba_tokenizer = None, None


# Function to compute conditional probability P(o | segment)
def compute_probability_per_sentence(segment, output ):
    global proba_model, proba_tokenizer
    output_tokens = proba_tokenizer(output, return_tensors="pt")["input_ids"]

    if not segment.strip():
        return None, None  # Skip empty sentences
    segment += "." if not segment.endswith(".") else ""  # Ensure proper punctuation
    input_text =  segment + " " + output
    inputs = proba_tokenizer(input_text, return_tensors="pt")

    # Get model output logits
    with torch.no_grad():
        outputs = proba_model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]




    log_prob_sum = 0.0
    for i, token_id in enumerate(output_tokens[0]):
        if i ==0:
            continue
        position = len(proba_tokenizer(segment, return_tensors="pt")["input_ids"][0]) + i - 2
        token_logits = logits[0, position]
        token_prob = torch.softmax(token_logits, dim=-1)[token_id].item()
        log_prob_sum += torch.log(torch.tensor(token_prob))  # Sum log probabilities

    # Normalization: Compute average log probability per token
    avg_log_prob = log_prob_sum / len(output_tokens[0])
    normalized_prob = torch.exp(avg_log_prob).item()  # Convert back to probability scale

    return   normalized_prob, avg_log_prob.item() #joint_probability, log_prob_sum.item() #


def get_segment_proba(passage, statement,segmentation="sentence"):
    global proba_model, proba_tokenizer
    if proba_model is None:
        model_name = "meta-llama/Llama-2-7b-chat-hf" 
        logger.info(f"Loading Language model...{model_name}")
        # Load the model and tokenizer
        proba_tokenizer = AutoTokenizer.from_pretrained(model_name)
        proba_model = AutoModelForCausalLM.from_pretrained(model_name,output_attentions=True)

    pssg_segments = sent_tokenize(passage)
    proba_of_all_segments = []
    for i in range(len(pssg_segments)):
        seg = pssg_segments[i]
        if segmentation == "cumulated_sentences":
            seg = " ".join(pssg_segments[:pssg_segments.index(seg)+1])
        probabilities, log_probabilities= compute_probability_per_sentence(seg, statement)
        proba_of_all_segments.append({"segment": pssg_segments[i], "index":i, "log_probabilities":log_probabilities,"probabilities":probabilities})
    sorted_proba = sorted(proba_of_all_segments, key=lambda x: x["log_probabilities"], reverse=True)
    return sorted_proba


def get_segment_nli(passage, statement,segmentation="sentence"):
    pssg_segments = sent_tokenize(passage)
    score_of_all_segments = []
    sorted_score_of_all_segments = []
    first_one = False
    for i in range(len(pssg_segments)):
        seg = pssg_segments[i] # segment used to compute nli
        if segmentation == "cumulated_sentences":
            seg = " ".join(pssg_segments[:pssg_segments.index(seg)+1])
        elif segmentation == "removing_sent":
            remaining_segments = pssg_segments[:i] + pssg_segments[i+1:]
            seg = " ".join(remaining_segments)
        nli_score = _run_nli_autoais(seg, statement)
        if segmentation == "removing_sent": 
            seg=pssg_segments[i]
        if segmentation == "cumulated_sentences": # put the first segment with nli=1 at the top of the list
            if nli_score > 0 and not first_one:
                first_one = True
                sorted_score_of_all_segments=[{"segment": pssg_segments[i], "index":i, "nli_score":nli_score}]
                sorted_score_of_all_segments.extend(score_of_all_segments)
            elif first_one:
                sorted_score_of_all_segments.append({"segment": pssg_segments[i], "index":i, "nli_score":nli_score})
        score_of_all_segments.append({"segment": pssg_segments[i], "index":i, "nli_score":_run_nli_autoais(seg, statement)})
    if segmentation == "cumulated_sentences":
        return sorted_score_of_all_segments
    if segmentation == "removing_sent": # if a removing the segmenet give a score nli = 0 means that it's important i.e. segments with score 0 first
        return sorted(score_of_all_segments, key=lambda x: x["nli_score"], reverse=False)
    sorted_scores= sorted(score_of_all_segments, key=lambda x: x["nli_score"], reverse=True)
    return sorted_scores

def get_segment_nli_proba(passage, statement,segmentation="cumulated_sentences"):
    """
    Computes NLI segmentation, if no segment has a score > 0, it returns the segments sorted by probability
    """
    combined_scores = []
    segments = get_segment_nli(passage, statement,segmentation)
    segments_proba = get_segment_proba(passage, statement,"sentence")
    for seg in segments:
        for seg_proba in segments_proba:
            if seg["segment"] == seg_proba["segment"]:
                combined_scores.append({"segment": seg["segment"], "index":seg["index"], "nli_score":seg["nli_score"], "log_probabilities":seg_proba["log_probabilities"]})
                break
    if segmentation == "removing_sent":
        at_least_one_one = any([seg["nli_score"] > 0 for seg in combined_scores])
        at_least_one_zero = any([seg["nli_score"] ==  0 for seg in combined_scores])
        at_least_one_one = at_least_one_zero and at_least_one_one
    else: 
        at_least_one_one = any([seg["nli_score"] > 0 for seg in combined_scores])
    if at_least_one_one:
        return combined_scores
    else:
        sorted_scores= sorted(combined_scores, key=lambda x: x["log_probabilities"], reverse=True)
        return sorted_scores


def get_sources_by_best_segment(seg_ranking):
    doc_score={}
    for doc in seg_ranking:
        doc_score[doc["source"]] = doc["segments"][0]["probabilities"]
    sorted_scores= sorted(doc_score.items(), key=lambda x: x[1], reverse=True)
    sorted_scores_dict = {k: v for k, v in sorted_scores}
    return sorted_scores_dict


def segmentation(args):
    logger.info(f'Extracting relevant segments from Hagrid {args.split}')
    logger.info(f"Methods to choose segments: {args.method}")
    logger.info(f"Segment split: {args.segmentation}")
    logger.info(f"Keep citation: {args.keep_citation}")
    logger.info(f"Uses the split provided in the dataset: {args.original_sent_split}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Results folder: {args.results_folder}")
    logger.info(f"Attributable only: {args.attributable_only}")
    if args.validating_code:
        logger.info(f'Running two iterations to validate code validating_code = {args.validating_code}')
    scores=[]
    hagrid = datasets.load_dataset("miracl/hagrid", split=args.split)

    for idx, row in enumerate(tqdm(hagrid)):
        if idx == 287 and args.validating_code:
            break
        for answer in row["answers"]:
            if not args. attributable_only or (args.attributable_only and answer["attributable"] == 1):
                segment_ranking_per_answer = []
                if args.answer_split =="auto_statement":
                    sentences =  split_text_by_citations(answer["answer"])
                elif args.answer_split =="sentences":
                    sentences =  get_statement_by_sentence(answer["answer"])
                elif args.answer_split == "original_sent_split" or args.original_sent_split:
                    sentences =[]
                    for sindice in range(len(answer["sentences"])):
                        s =answer["sentences"][sindice]
                        s["source"] = get_source_from_text(s["text"])
                        sentences.append(s)
                else:
                    sentences =  split_text_by_citations(answer["answer"])
                for i in range(len(sentences)):
                    sentences[i]["index"] = i
                    segment_ranking = []
                    if args.keep_citation:
                        sent = sentences[i]["text"]
                    else:
                        sent = remove_citations(sentences[i]["text"])
                    sources = sentences[i]["source"]
                    for src in sources:
                        if src <= len(row["quotes"]):
                            passage = row["quotes"][src-1]["text"]
                            if args.method == "proba":
                                segment_ranking.append({"source":src, "segments":get_segment_proba(passage,sent,args.segmentation) })
                            elif args.method == "nli":
                                segment_ranking.append({"source":src, "segments":get_segment_nli(passage,sent,args.segmentation)})
                            elif args.method == "mixed":
                                segment_ranking.append({"source":src, "segments":get_segment_nli_proba(passage, sent,args.segmentation)})
                    segment_ranking_per_answer.append({"statement":sentences[i], "cited_doc":segment_ranking})
                answer["segments"]= segment_ranking_per_answer
                scores.append(row)
                    
    model_name = args.model_name.split('/')[-1]
    attributed= "_attributable_only" if args.attributable_only else "all"
    code_validation="_validating_code" if args.validating_code else ''
    keep_citation = "_keep_citation" if args.keep_citation else ""
    og_split = "_original_split" if args.original_sent_split else "_auto_statement"
    split = args.answer_split if args.answer_split else og_split
    results_file =  "segment_extract_"+args.method+"_"+args.segmentation+"_"+model_name+split+keep_citation+attributed+code_validation+".json"
    results_file = os.path.join(args.results_folder, results_file) #ROOT_PATH, 
    logger.info(f"Saving results to {results_file}")
    new_set={"data":scores, "params":vars(args)}
    with open(results_file, "w") as f:
        json.dump(new_set, f, indent=4)



def eval_faithfulness(args):
    logger.info(f'Extracting segments to evaluate faithfulness of citations: {args.file}')
    logger.info(f"Methods to choose segments: {args.method}")
    logger.info(f"Segment split: {args.segmentation}")
    logger.info(f"Keep citation: {args.keep_citation}")
    logger.info(f"Uses the split provided in the dataset: {args.original_sent_split}")
    logger.info(f"Model name: {args.model_name}")
    logger.info(f"Results folder: {args.results_folder}")
    if args.validating_code:
        logger.info(f'Running two iterations to validate code validating_code = {args.validating_code}')
    updated_rows=[]
    with open(args.file, 'r') as file:
        dataset = json.load(file)
        if isinstance(dataset, dict):
            dataset = dataset["data"]
    for idx, row in enumerate(tqdm(dataset)):
        segment_ranking_per_answer = []
        if idx == 2 and args.validating_code:
            break
        if args.original_sent_split:
            print("Not implemented")
            break
        else:
            sentences =  split_text_by_citations(row["output"])
        for i in range(len(sentences)):
            segment_ranking = []
            if args.keep_citation:
                sent = sentences[i]
            else:
                sent = remove_citations(sentences[i]["text"])

            for src in range(len(row["docs"])):
                passage = row["docs"][src]["text"]
                if args.method == "proba":
                    segment_ranking.append({"source":src, "segments":get_segment_proba(passage,sent,args.segmentation) })
                elif args.method == "nli":
                    segment_ranking.append({"source":src, "segments":get_segment_nli(passage,sent,args.segmentation)})
                elif args.method == "mixed":
                    segment_ranking.append({"source":src, "segments":get_segment_nli_proba(passage, sent,args.segmentation)})

            #### rank docs by seg by proba
            sent_with_faithful_citation = sent[:-1] if sent.endswith(".") else sent
            sorted_scores = get_sources_by_best_segment(segment_ranking)
            sent_with_faithful_citation = sent_with_faithful_citation + "".join(["["+str(ex_cit+1)+"]" for ex_cit in list(sorted_scores.keys())[:len(sentences[i]["source"])]]) +" ."
            segment_ranking_per_answer.append({"statement":sentences[i], "generated_citations":sentences[i]["source"], "faithful_citations":sorted_scores, "faithful_statement":sent_with_faithful_citation})
        row["faithful_cit"]= segment_ranking_per_answer
        row["faith_output"] = " ".join([l["faithful_statement"] for l in segment_ranking_per_answer])
        updated_rows.append(row)


    eval_file_path = args.file.split('/')
    eval_file = eval_file_path[-1] if eval_file_path[-1] !="" else eval_file_path[-2]
    results_file = eval_file.split('.')[-2] + "faithful_eval"
    code_validation="_validating_code" if args.validating_code else ''
    method= args.method
    seg = args.segmentation

    results_file =  results_file+code_validation+"_"+method+"_"+seg+".json" 
    results_file = os.path.join(args.results_folder, results_file) #ROOT_PATH, 

    logger.info(f"Saving results to {results_file}")
    new_set={"data":updated_rows, "params":vars(args)}
    with open(results_file, "w") as f:
        json.dump(new_set, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task", 
        type=str,
        default="segmentation",
        choices=["segmentation","faithfulness_eval"], help="Which function to call"
    )

    parser.add_argument(
        "--file", type=str, default=None, help="File to evaluate"
    )

    parser.add_argument(
        "--method", 
        type=str,
        default="proba",
        choices=["proba", "prompt", "relevance","nli","mixed"], help="Methods to extract segments"
    )

    parser.add_argument(
        "--segmentation", 
        type=str,
        default="sentence",
        choices=["sentence", "cumulated_sentences","removing_sent"], help="Methods to segment passages"
    )

    parser.add_argument(
        "--keep_citation", action="store_true", help="Keeps citation in the text")

    parser.add_argument(
        "--original_sent_split", action="store_true", help="Uses the split provided in the dataset."
    )
    parser.add_argument(
        "--answer_split", 
        type=str,
        default="original_sent_split",
        choices=["original_sent_split", "auto_statement", "sentences"], help="Methods to extract segments"
    )

    parser.add_argument(
        "--model_name", type=str, default="meta-llama/Llama-2-13b-chat-hf", help="Model name for the proba method"
    )
    parser.add_argument(
        "--results_folder", type=str, default="results/segmentation/", help="Folder to save the results"
    )
    parser.add_argument(
        "--attributable_only", action="store_true", help="Only attributable answers"
    )

    parser.add_argument(
        "--validating_code", action="store_true", help="Running two iterations to validate code"
    )    
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split"
    )  
    parser.add_argument(
        "--vllm", action="store_true", help="use Vllm"
    )
    args = parser.parse_args()

    if args.task == "segmentation":
        segmentation(args)
    elif args.task == "faithfulness_eval":
        eval_faithfulness(args)
    else:
        print("Task not specified")



