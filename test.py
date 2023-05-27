from tree_of_thoughts import OptimizedOpenAILanguageModel
from the_compiler import TerminalExecutor, TheCompiler

# Initialize the LLM
api_key = "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"
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
