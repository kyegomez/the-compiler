# The-Compiler
Seed, Code, Harvest: Grow Your Own App with Tree of Thoughts!

![the-compiler](the-compiler.png)

Welcome to _The Compiler_, a novel child project under the Tree of Thoughts (ToT) paradigm. [Implementation here](https://github.com/kyegomez/tree-of-thoughts) This project is crafted with the intent of making autonomous programming not just a reality, but an effortless task for you. 

In essence, _The Compiler_ allows you to "grow" any program you can dream of. By providing a high-level specification of the product you would like, you can sit back and let _The Compiler_ do the heavy lifting. 

# Agora, Creators United
The Compiler is brought to you by Agora, we're an community of creators united under the banner of Humanity.
We utilize AI research as a means to solve Humanity's biggest obstacles like food production, planetary security, disease, and death

[Join us and advance Humanity](https://discord.gg/qUtxnK2NMf)

## Overview 

_The Compiler_ leverages the ToT framework and large language models (LLMs) to handle the programming process, from abstract specifications to a working program. 

Here's a basic breakdown of the workflow:

1. **Input**: You provide an abstract specification for the product you would like.
2. **Unit Tests Generation**: We use an LLM on ToT to produce a suite of unit tests for the code.
3. **Run ToT**: We run the Tree of Thoughts LLM on the given specification, using the generated unit tests as the evaluation score.
4. **Output**: Ready to use program!


# Usage
Get started: 

```git clone https://github.com/kyegomez/the-compiler.git ```

and or 

``` pip install tree-of-thoughts```

```pip install the-compiler```

2nd step
``` cd the-compiler ```

3rd step
Create an file called config_json.json and put this inside:

```json
{
    "allow_terminal_execution": true
}
```

4th step -- create an new file and place this inside
``` python

from tree_of_thoughts import OptimizedOpenAILanguageModel
from the_compiler import TerminalExecutor, TheCompiler

# Initialize the LLM
api_key = "api key"
LLM = OptimizedOpenAILanguageModel(api_key)

# Initialize the TerminalExecutor
terminal_executor = TerminalExecutor(config_file="config_file.json")

# Initialize The Compiler with the LLM and TerminalExecutor
compiler = TheCompiler(LLM, terminal_executor)

# Example usage
input_spec = "Create a simple calculator that can perform addition, subtraction, multiplication, and division"
k = 5
T = 3
b = 5
vth = 0.5
timeout = 10
confidence = 0.9
max_iterations = 5
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
        print(f"Command output: {output}") #

```


## Architecture

The Compiler, leveraging the Tree of Thoughts paradigm, consists of several primary components, including the Specification Parser, Thought Decomposer, Thought Generator, State Evaluator, and the Search Algorithm. 

1. **Specification Parser**: This interprets your high-level input specifications and translates them into a format that the Thought Decomposer can understand and work with.

2. **Thought Decomposer**: This component breaks down the programming problem into manageable "thoughts" or steps.

3. **Thought Generator**: It generates potential thoughts or steps from the current state using two strategies, either sampling thoughts independently or proposing thoughts sequentially.

4. **State Evaluator**: It evaluates the progress of different states towards solving the programming problem, acting as a heuristic for the Search Algorithm.

5. **Search Algorithm**: This module determines which states to keep exploring and in which order. It employs either Breadth-First Search (BFS) or Depth-First Search (DFS), depending on the nature of the problem.

## Share The Compiler

If you find this project exciting and think others might benefit from it, feel free to share it. Use the buttons below to share it on various social media platforms:

- [Share on Twitter](http://twitter.com/share?text=Check%20out%20The%20Compiler%20project%20on%20GitHub!%20It%20allows%20you%20to%20autonomously%20create%20programs%20using%20abstract%20specifications.&url=https://github.com/kyegomez/the-compiler)
- [Share on LinkedIn](http://www.linkedin.com/shareArticle?mini=true&url=https://github.com/kyegomez/the-compiler&title=The%20Compiler%20Project&summary=This%20project%20is%20a%20revolution%20in%20autonomous%20programming!%20Check%20it%20out%20on%20GitHub.)
- [Share on Facebook](http://www.facebook.com/sharer.php?u=https://github.com/kyegomez/the-compiler)

Let's revolutionize the world of programming together with _The Compiler_!



