from transformers import StoppingCriteriaList, StoppingCriteria, StoppingCriteriaList
import torch
import re

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            # Compare the last part of input_ids with the stop sequence
           if input_ids.shape[1] >= len(stop):  # Ensure the input is long enough
                if torch.all(stop == input_ids[0][-len(stop):]).item():
                    return True
        return False



class Agent:
    def __init__(
        self,
        model,
        tokenizer,
        tools,
        rounds=6,
        use_tools=True,
        num_docs=2,
        train_corpus="HAGRID",
        adjusted=False,
        model_params = "7B",
        manual_stop_words= None,
        without_query_gen = None,
        add_instruction = None,
        diverse_query_only = False,
        add_user_query = None,
        add_answer=None,
        retrieve_once=None,
    ):

        self.model = model
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tools = tools
        self.adjusted = adjusted
        self.rounds = rounds
        self.num_docs = num_docs
        self.use_tools = use_tools
        self.train_corpus = train_corpus
        self.without_query_gen = without_query_gen
        self.add_instruction = add_instruction
        self.diverse_query_only = diverse_query_only
        self.add_user_query =add_user_query
        self.add_answer = add_answer
        self.retrieve_once =retrieve_once
        stop_words = self.get_stop_token()
        stop_words_ids = [
            self.tokenizer(stop_word, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ].squeeze()
            for stop_word in stop_words
        ]
        if manual_stop_words:
            if model_params == "13B":
                stop_words_ids.append(torch.tensor(manual_stop_words["13B"]))
            if model_params == "7B":
                stop_words_ids.append(torch.tensor(manual_stop_words["13B"]))  
            if model_params == "3B":
                if self.adjusted:
                    stop_words_ids.append(torch.tensor(manual_stop_words["13Badj"])) 
                else:        
                    stop_words_ids.append(torch.tensor(manual_stop_words["3B"]))
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

    def detect_tool(self, message):
        """
        Finds the ID of the latest substring in a string.

        Args:
            string: The string to search.
            substrings: A list of substrings to search for.

        Returns:
            The ID of the latest substring found in the string, or None if no substrings are found.
        """
        latest_id = None
        latest_start = -1
        substrings = [tool.end_token for tool in self.tools]
        for i, substring in enumerate(substrings):
            start = message.rfind(substring)
            if start != -1 and start > latest_start:
                latest_id = i
                latest_start = start
        return latest_id

    def get_stop_token(self):
        list_end_gen = []
        for tool in self.tools:
            list_end_gen.append(tool.end_token)
        return list_end_gen

    def generate(self, question, docs=None, **kwargs):
        all_docs = []
        all_scores = []
        used_docids = {}
        pattern = r'\[DOCS\].*?\[/DOCS\]'
        query_pattern = r'(\[SEARCH\])(.*?)(\[/SEARCH\])'
        if self.add_instruction:
            instruction= "Given the user query, provide a long answer that tackles different related aspects. To construct your answer, you will alternate between generating a subquery between [SEARCH][/SEARCH] tokens that describes what you will talk about, then use the provided doucments [DOCS][/DOCS] to generate an answer to the subquery and cite the documents you use. Repeat the process until the query is fully answered. Use your generated answer to generate the next subquery based on what you intend to tackle next. The subqueries should be diverse and different from previous ones and allow you to gather new information."
            message = [{"role": "system", "content":instruction},{"role": "user", "content": question}]
            print("Adding system instruction:", instruction)
        else:
            message = [{"role": "user", "content": question}]
        inputs = self.tokenizer.apply_chat_template(
            message, tokenize=True, add_generation_prompt=True, return_tensors="pt", truncation=True
        )
        max_length = self.tokenizer.model_max_length
        last_gen = 0
        generated_tool = False
        if self.diverse_query_only:
            query_gen_kw = kwargs
            standard_gen = {"do_sample": True, "top_p": 0.5, "max_new_tokens": 1000}
        for i in range(self.rounds):
            try:
                output = self.model.generate(
                    inputs.to(self.model.device),
                    stopping_criteria=self.stopping_criteria,
                    **kwargs,
                )
            except :
                output =""
                continue
            if self.diverse_query_only:
                kwargs = standard_gen
            output = self.tokenizer.batch_decode(output)[0]
            if output[-1] == "[":
                output= output[:-1]
            if i == 0 and not self.add_user_query and not self.retrieve_once:
                output=output.replace("[SEARCH]","[ANSWER][SEARCH]")

            if self.retrieve_once:
                try:
                    output = re.sub(query_pattern, "[SEARCH]"+question+"[/SEARCH]", output)
                except re.error as e:
                    print(f"Regex error during replacement: {e}")
                    print("output in exception:",output)
                    print("question in exception:",question)
                    safe_question = question.replace("\\", r"\\")
                    safe_output=output.replace("\\", r"\\")
                    output = re.sub(query_pattern, "[SEARCH]"+safe_question+"[/SEARCH]", safe_output)
            if self.adjusted:
                hallucinated_docs = output.find("[DOCS]")
                if generated_tool == False and hallucinated_docs != -1:
                    output = output[:hallucinated_docs]
                output  =  re.sub(pattern, '', output)
            cuurent_output = output[last_gen:]
         
            if self.adjusted:
                patternA = r'\[ANSWER\](.*?)\[/ANSWER\]'
                matches = re.findall(patternA, cuurent_output, re.DOTALL)
                if len(matches) >= 3:
                    a_idx = cuurent_output.find("[ANSWER]")
                    cuurent_output = cuurent_output[:a_idx] +"[ANSWER]" + matches[0] +"[/ANSWER][ANSWER]"+matches[1]+"[/ANSWER]"
                    output = output[:last_gen] + cuurent_output
                        
            last_gen = len(output)
            tool_id = self.detect_tool(cuurent_output)
            if tool_id is not None:
                if self.add_user_query:
                    # Find all matches
                    matches = list(re.finditer(query_pattern, output))
                    if matches:
                        last_match = matches[-1]
                        start, end = last_match.span()
                        before = output[:start]
                        middle = last_match.group(1) + question + last_match.group(2) +  last_match.group(3)
                        after = output[end:]
                        output = before + middle + after
                        appended_subquery = middle
             
                start = output.rfind(self.tools[tool_id].start_token)
                if start == -1:
                    break 
                generated_tool = True
                if self.use_tools:
                    if docs:
                        docs_text, scores, inputs = self.tools[tool_id](
                            output, k=self.num_docs, initial_docs=docs
                        )
                    else:
                        docs_text, scores, inputs = self.tools[tool_id](
                            output, k=self.num_docs
                        )
                    all_scores.append(scores)
                    if self.train_corpus == "WEBGPT":
                        for doc in docs_text:
                            docid = doc["docid"]
                            if docid not in used_docids:
                                used_docids[docid] = len(used_docids) + 1
                                doc["indice"] = used_docids[docid]
                                all_docs.append(doc)
                                inputs = inputs.replace(
                                    str(docid), str(used_docids[docid])
                                )
                    else:
                        all_docs.append(docs_text)

                else:
                    ### without retrieval
                    inputs = output + f"\n[DOCS] {[]} [/DOCS]\n"
                if self.add_user_query:
                    if len(matches):
                        inputs=inputs.replace(appended_subquery,original_subquery)
            else:
                if self.adjusted:
                    inputs = output
                    generated_tool = False
                else:
                    inputs = output
                    generated_tool = False
                    inputs = inputs.replace("<|endoftext|>","")
                    inputs = inputs.replace("</s>","")
                    inputs = inputs +"[SEARCH]"
                if self.diverse_query_only:
                    kwargs = query_gen_kw
            if i == 0:
                inputs=inputs.replace("[ANSWER][SEARCH]","[SEARCH]")
            inputs = inputs.replace("<|endoftext|>","")
            inputs = inputs.replace("</s>","")
            inputs = self.tokenizer(
                 inputs, return_tensors="pt", add_special_tokens=False, truncation=True
            )["input_ids"] 
            if inputs.size(1) > max_length:
                print("Exceeding Max length: ", max_length)
                inputs = inputs[:, :max_length]
                break
        return all_docs, all_scores, output
