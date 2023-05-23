#autonmous agent load balancing add more agents to spec interpreter, add more agents to spec generate or unit tester, default is 3 1 for each, use your own llms, starcoder, reinforcement backed individual agent learning, particle swarm tree of thoughts -> run tree of thoughts on the space of all thoughts ran by agents as trees of an networked swarm system, use autogpt config and env with embedding retriveal search, map all thoughts to an embedding database, while now tester is not done keep running in a while loop, agentgpt with autoforestgpt, program optimizer agent works in unison with the generator to optimize code. Create 5 optimizations for this code: {code}, for generator ask generator to ask questions about spec and unit test and then answer those questions by transforming it into algoritmic pseudoce then code with the logic, also add spawns where boss node spawns new agents for each task, https://github.com/Araq/malebolgia, have agent select programming langauges and infrastcuture to pass into the unit tester and generator, prompt, each agent stops when the tree has found an optimal solution
#inital implementation of the compiler
import subprocess
import json
import re

import concurrent.futures
from abc import ABC, abstractmethod
import openai
import os
import re
import time

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass


class CustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        pass
class OpenAILanguageModel(AbstractLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True):
        if api_key == "" or api_key == None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base == None:
            api_base = os.environ.get("OPENAI_API_BASE", "")  # if not set, use the default base path of "https://api.openai.com/v1"
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')
            
        if api_model == "" or api_model == None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model

        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."
        
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def openai_api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        while True:
            try:
                if self.use_chat_api:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = openai.ChatCompletion.create(
                        model=self.api_model,
                        messages=messages,
                        max_tokens=1000,
                        temperature=temperature,
                    )
                else:
                    response = openai.Completion.create(
                        engine=self.api_model,
                        prompt=prompt,
                        n=k,
                        max_tokens=max_tokens,
                        stop=stop,
                        temperature=temperature,
                    )
                return response
            except openai.error.RateLimitError as e:
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)

    def openai_choice2text_handler(self, choice):
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        
        prompt = f"Given the current state of reasoning: '{state_text}', generate {1} coherent thoughts to continue the reasoning process:"
        prompt += self.ReAct_prompt
        if self.use_chat_api:
            new_prompt_success = False
            """
            # Try prompt and parse in a single shot to save tokens (but if we fail, we end up spending more tokens)
            new_prompt = prompt + "Thought string should be output in a format that can be parsed into python array in format [xxx,xxx,xxx]"
            response = self.openai_api_call_handler(new_prompt, 100 * k, 0.5, 1)
            text = self.openai_choice2text_handler(response.choices[0])
            re_parse = re.search(r'\[(.*?)\]', text)
            if re_parse:
                thoughts_str = re_parse.group(1)
                if thoughts_str:
                    thoughts = thoughts_str.split(',')
                    new_prompt_success = len(thoughts) == k 
                    if not new_prompt_success:
                        print(f"Fall back to multi-prompt for chat-completion due to parse fail {text}")

            """
            if not new_prompt_success:
                thoughts = []
                for _ in range(k):
                    response = self.openai_api_call_handler(prompt, 50, 0.5, k)
                    text = self.openai_choice2text_handler(response.choices[0])
                    thoughts += [text]
            
        else:
            response = self.openai_api_call_handler(prompt, 50, 0.5, k)
            thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts

    def evaluate_states(self, states):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, and NOTHING ELSE:"
                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            prompt = f"Given the following states of reasoning, vote for the best state:\n{states_text}\n\nVote, and NOTHING ELSE:"
            response = self.openai_api_call_handler(prompt, 50, 1)
            best_state_text = self.openai_choice2text_handler(response.choices[0])
            print(f"Best state text: {best_state_text}")
            best_state = tuple(best_state_text.split())
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=True):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    


class TreeofThoughts:
    """
    1. Thought Decomposition --> based on problem properties

    2. Thought Generator -> create a thought generator function G(p0, s, k) with 2 strategies a sample iid thoughts from a cot prompt b. propose thoughts
    sequentially using a propose prompt

    3. create a state evaluator function V(p0, S) with 2 strategies a value each state independently b. vote across states

    4. Choose a search algo based on tree structure [BFS or DFS]

    Implement chosen search algorithm for bfs (algo1):
        init S0 with the input x
        for t = 1 to T (step limit):
            generate candidate thoughts for each state in St-1
            eveluate the candiate states using the state evaluator V
            select the b most promising states for St

        return the final output by genertaing the thought for the best state in St for DFS(algo2)

        defien a recurseive DFS function with the current state s, step t, and other required params

        if t > T record the output by generating the thought for current state S

        for each candidate state s in the sorted list of generated thoughts for s:
            
            if the evaluated value of s is greater the the threshold of vth call the dfs function recursively
            with s and t + 1

    execute the chosen search algo with the input problem, thought generator, and state evaluator, and other required params
    """

    def __init__(self, model, search_algorithm):
        self.model = model
        self.search_algorithm = search_algorithm

    def solve(self, x, k, T, b, vth, timeout=None):
        start_time = time.time()
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_dfs(x, k, T, vth)
                if result:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")

    def tot_bfs(self, x, k, T, b):
        S0 = {x}
        for t in range(1, T + 1):
            S0_t = {(*s, z) for s in S0 for z in self.model.generate_thoughts(s, k)}
            Vt = self.model.evaluate_states(S0_t)
            St = sorted(S0_t, key=lambda s: Vt[s], reverse=True)[:b]
            S0 = set(St)
        return self.model.generate_thoughts(max(St, key=lambda s: Vt[s]), 1)

    def tot_dfs(self, x, k, T, vth, pruning_threshold=0.5, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        output = []
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None
        tree_traversed = None

        def dfs(s, t):
            nonlocal consecutive_convergence_count, prev_best_value, iteration_count
            if t > T:
                thought = self.model.generate_thoughts(s, 1)
                value = self.model.evaluate_states({s})[s]
                output.append((thought, value))

                if confidence_threshold is not None and value >= confidence_threshold:
                    return True

                if prev_best_value is not None and convergence_threshold is not None:
                    if abs(value - prev_best_value) < convergence_threshold:
                        consecutive_convergence_count += 1
                    else:
                        consecutive_convergence_count = 0

                prev_best_value = value
                iteration_count += 1

                if (max_iterations is not None and iteration_count >= max_iterations) or (convergence_count is not None and consecutive_convergence_count >= convergence_count):
                    tree_traversed = True
                    return True

                return False

            for s_prime in sorted(self.model.generate_thoughts(s, k)):
                state_value = self.model.evaluate_states({s_prime})[s_prime]
                print(f'State values: {state_value}')
                if state_value > vth and (pruning_threshold is None or state_value >= pruning_threshold):
                    if dfs((*s, s_prime), t + 1):
                        return True

            return False

        dfs(x, 1)
        return (max(output, key=lambda x: x[1]) if output else None), tree_traversed


class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k=5, T=3, b=5, vth=0.5, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        start_time = time.time()
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                print(f"result: {result}")
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result, tree_traversed = self.tot_dfs(x, k, T, vth, confidence_threshold=confidence_threshold, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
                print(f'Result {result}')
                if result or tree_traversed:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")


class TerminalExecutor:
    def __init__(self, config_file="config.json"):
        self.load_config(config_file)

    def load_config(self, config_file):
        with open(config_file, "r") as f:
            config = json.load(f)
        self.allow_terminal_execution = config.get("Allow_terminal_execution", False)

    def execute(self, command):
        if self.allow_terminal_execution:
            try:
                result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
                return result.stdout
            except subprocess.CalledProcessError as e:
                print(f"Error executing command: {e}")
                return None
        
        else:
            print("Terminal execution is not allowed please enable it in the config file")
            return None
        
#example usage
executor = TerminalExecutor()

#test command execution
command = "echo 'Hello, World'"
output = executor.execute(command)
if output:
    print(f"Command output: {output}")

# class SpecInterpreter:
#     def __init__(self, LLM, terminal_executor):
#         self.LLM = LLM
#         self.terminal_executor = terminal_executor

#     def translate(self, spec):
#         prompt = f"Create an architectural technical analysis with the architecture, classes, algorithms, and logic needed to create a {spec} in Python. If any terminal commands are needed, provide them in the format 'TerminalCommand: <command>'."
#         translated_spec = self.LLM.generate_thoughts(prompt, 1)[0]
#         return translated_spec

# class TestGenerator:
#     def __init__(self, LLM, terminal_executor):
#         self.LLM = LLM
#         self.terminal_executor = terminal_executor

#     def generate(self, spec):
#         prompt = f"Generate a suite of unit tests in the most appropiate language for a the program that meets the following product specification: {spec}. If any terminal commands are needed, provide them in the format 'TerminalCommand: <command>'."
#         unit_tests = self.LLM.generate_thoughts(prompt, 1)[0]
#         return unit_tests

# class ToTRunner:
#     def __init__(self, LLM, terminal_executor):
#         self.LLM = LLM
#         self.terminal_executor = terminal_executor

#     def generate_program(self, spec, unit_tests):
#         prompt = f"Generate an program in the language specified by the TestGenerator that meets the following product specification: {spec}. Use the following unit tests as an evaluation score: {unit_tests}. If any terminal commands are needed, provide them in the format 'TerminalCommand: <command>'."
#         program = self.LLM.generate_thoughts(prompt, 1)[0]
#         return program
class SpecInterpreter:
    def __init__(self, LLM, terminal_executor):
        self.LLM = LLM
        self.terminal_executor = terminal_executor

    def translate(self, spec):
        prompt = f"Create an architectural analysis in markdown in the most optimal programming language for a {spec}, provide the fastest, reliable architecture, and the break down that architecture into classes and algorithms needed to create {spec} If any terminal commands are needed, provide them in the format 'TerminalCommand: <command>'."
        translated_spec = self.LLM.generate_thoughts(prompt, 1)[0]
        return translated_spec

class TestGenerator:
    def __init__(self, LLM, terminal_executor):
        self.LLM = LLM
        self.terminal_executor = terminal_executor

    def generate(self, spec):
        prompt = f"Generate a suite of unit tests for a Python program that meets the following product specification: {spec}. If any terminal commands are needed, provide them in the format 'TerminalCommand: <command>'."
        unit_tests = self.LLM.generate_thoughts(prompt, 1)[0]
        return unit_tests

class ToTRunner:
    def __init__(self, LLM, terminal_executor):
        self.LLM = LLM
        self.terminal_executor = terminal_executor

    def generate_program(self, spec, unit_tests):
        prompt = f"Generate a Python program that meets the following product specification: {spec}. Use the following unit tests as an evaluation score: {unit_tests}. If any terminal commands are needed, provide them in the format 'TerminalCommand: <command>'."
        program = self.LLM.generate_thoughts(prompt, 1)[0]
        return program

class TheCompiler:
    def __init__(self, LLM, terminal_executor, search_algorithm="DFS"):
        self.spec_interpreter = SpecInterpreter(LLM, terminal_executor)
        self.test_generator = TestGenerator(LLM, terminal_executor)
        self.tot_runner = ToTRunner(LLM, terminal_executor)
        self.tree_of_thoughts = OptimizedTreeofThoughts(LLM, search_algorithm)

    def compile(self, input_spec, k, T, b, vth, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        while True:
            translated_spec = self.spec_interpreter.translate(input_spec)
            if self.is_spec_complete(translated_spec):
                unit_tests = self.test_generator.generate(translated_spec)
                program = self.tot_runner.generate_program(translated_spec, unit_tests)
                return program
            #add a condition loop to break the if necessary

    def is_spec_complete(self, translated_spec):
        #implement an condition to check if the translated spec is complete
        #for example you can check if the translated spec contains keywords or snippets
        #return tree if the translated spec is complete otherwise return
        pass

# Initialize the LLM
api_key = "api key"
LLM = OptimizedOpenAILanguageModel(api_key)

# Initialize the TerminalExecutor
terminal_executor = TerminalExecutor(config_file="/Users/defalt/Desktop/Athena/research/the-compiler/config.json")

# Initialize The Compiler with the LLM and TerminalExecutor
compiler = TheCompiler(LLM, terminal_executor)

# Example usage
input_spec = "Create a simple calculator that can perform addition, subtraction, multiplication, and division in python"
k = 10
T = 6
b = 10
vth = 1.0
timeout = 10
confidence = 1.0 #cmodel is confident on performance
max_iterations = 40 #tree branh nodes 
convergence_threshold = 0.01
convergence_count = 5

# Call the compile method with the input problem and other params
solution = compiler.compile(input_spec, k, T, b, vth, timeout, confidence_threshold=confidence, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)

# Use the solution
print(f"solution: {solution}")

# Extract terminal commands from the solution
terminal_commands = re.findall(r"TerminalCommand: (.+)", solution)

# Execute terminal commands
for command in terminal_commands:
    output = terminal_executor.execute(command)
    if output:
        print(f"Command output: {output}")