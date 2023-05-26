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
import concurrent.futures
from abc import ABC, abstractmethod
import openai

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
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value"):
        openai.api_key = api_key
        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def generate_thoughts(self, state, k):
        state_text = ' '.join(state)
        
        prompt = f"Given the current state of reasoning: '{state_text}', generate {k} coherent thoughts to continue the reasoning process:"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            n=k,
            max_tokens=50,
            stop=None,
            temperature=0.5,
        )
        thoughts = [choice.text.strip() for choice in response.choices]
        print(thoughts)
        return thoughts

    def evaluate_states(self, states):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1:"
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    n=1,
                    max_tokens=10,
                    stop=None,
                    temperature=0.5,
                )
                try:
                    # print(response.choices[0].text.strip())
                    value = float(response.choices[0].text.strip())
                    print(value)
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])
            prompt = f"Given the following states of reasoning, vote for the best state:\n{states_text}\n\nVote:"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                n=1,
                max_tokens=50,
                stop=None,
                temperature=0.5,
            )
            best_state_text = response.choices[0].text.strip()
            print(best_state_text)
            best_state = tuple(best_state_text.split())
            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")

class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True):
        super().__init__(api_key, strategy, evaluation_strategy)
        self.cache_enabled = cache_enabled
        self.thought_cache = {}
        self.state_evaluation_cache = {}

    def parallel_generate_thoughts(self, states, k):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(thoughts)
        return thoughts

    def parallel_evaluate_states(self, states):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states))
            print(state_values)
        return state_values
    

# model = OptimizedOpenAILanguageModel('your_openai_api_key_here')

#update tree of thoughts to use optimized models mehtods

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

    def solve(self, x, k, T, b, vth):
        if self.search_algorithm == 'BFS':
            return self.tot_bfs(x, k, T, b)
        elif self.search_algorithm == 'DFS':
            return self.tot_dfs(x, k, T, vth)
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

    def tot_dfs(self, x, k, T, vth):
        output = []

        def dfs(s, t):
            if t > T:
                output.append(self.model.generate_thoughts(s, 1))
                return
            for s_prime in sorted(self.model.generate_thoughts(s, k)):
                if self.model.evaluate_states({s_prime})[s_prime] > vth:
                    dfs((*s, s_prime), t + 1)

        dfs(x, 1)
        return output


class OptimizedTreeofThoughts(TreeofThoughts):
    def tot_bfs(self, x, k, T, b):
        S0 = {x}
        for t in range(1, T + 1):
            S0_t = {(*s, z) for s in S0 for z in self.model.parallel_generate_thoughts(s, k)}
            Vt = self.model.parallel_evaluate_states(S0_t)
            St = sorted(S0_t, key=lambda s: Vt[s], reverse=True)[:b]
            S0 = set(St)
        return self.model.generate_thoughts(max(St, key=lambda s: Vt[s]), 1)

    def tot_dfs(self, x, k, T, vth):
        output = []

        def dfs(s, t):
            if t > T:
                output.append(self.model.generate_thoughts(s, 1))
                return
            for s_prime in sorted(self.model.generate_thoughts(s, k)):
                if self.model.evaluate_states({s_prime})[s_prime] > vth:
                    dfs((*s, s_prime), t + 1)

        dfs(x, 1)
        return output

    


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
terminal_executor = TerminalExecutor(config_file="config.json")

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