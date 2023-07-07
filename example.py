from the_compiler import TheCompiler

api_key = ""  # Your OpenAI API key

create = "a simple calculator program"
compiler = TheCompiler(api_key)

code = compiler.run(create)
print("Generated Code:\n", code)


