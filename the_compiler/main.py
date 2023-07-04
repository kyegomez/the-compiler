import os
from swarms import Swarms


class Architect:
    def __init__(self, create, api_key):
        self.create = create
        self.boss_node = boss_node(openai_api_key=api_key)

    def generate_architecture(self):
        objective = f"""
        Create an architectural analysis specification in markdown in the most optimal programming language for {self.create}, provide the fastest, reliable architecture, and then break down that architecture into classes and algorithms needed to create {self.create}
        """
        task = self.boss_node.create_task(objective=objective)
        return self.boss_node.execute(task)


class CodeGenerator:
    def __init__(self, boss1, create, api_key, unit_tests):
        self.boss1 = boss1
        self.create = create
        self.unit_tests = unit_tests
        self.boss_node = boss_node(openai_api_key=api_key)

    def generate_code(self):
        objective = f"""
        Generate a Python program that meets the following product specification: {self.boss1} to create: {self.create}. Use the following unit tests as an evaluation score: {self.unit_tests}.
        """
        task = self.boss_node.create_task(objective=objective)
        return self.boss_node.execute(task)


class TestCreator:
    def __init__(self, boss1, api_key):
        self.boss1 = boss1
        self.boss_node = boss_node(openai_api_key=api_key)

    def generate_tests(self):
        objective = f"""
        Generate a suite of unit tests for a Python program that meets the following product specification: {self.boss1}
        """
        task = self.boss_node.create_task(objective=objective)
        return self.boss_node.execute(task)



class TheCompiler:
    def __init__(self, api_key):
        self.swarms = Swarms(api_key=api_key)

    def run(self, create):
        architecture = self.swarms.run_swarms(
            objective=f"Create an architectural analysis specification in markdown in the most optimal programming language for {create}, provide the fastest, reliable architecture, and then break down that architecture into classes and algorithms needed to create {create}"
        )

        unit_tests = self.swarms.run_swarms(
            objective=f"Generate a suite of unit tests for a Python program that meets the following product specification: {architecture}"
        )

        code = self.swarms.run_swarms(
            objective=f"Generate a Python program that meets the following product specification: {architecture} to create: {create}. Use the following unit tests as an evaluation score: {unit_tests}."
        )
        return code

api_key = ""  # Your OpenAI API key
create = "a simple calculator program"

compiler = TheCompiler(api_key)
code = compiler.run(create)

print("Generated Code:\n", code)











# class TheCompiler(Swarms):
#     def __init__(self, api_key):
#         super().__init__(api_key=api_key)

#     def _generate_architecture(self, create):
#         objective = f"""
#         Create an architectural analysis specification in markdown in the most optimal programming language for {create}, provide the fastest, reliable architecture, and then break down that architecture into classes and algorithms needed to create {create}
#         """
#         return self.run_swarms(objective=objective)

#     def _generate_tests(self, boss1):
#         objective = f"""
#         Generate a suite of unit tests for a Python program that meets the following product specification: {boss1}
#         """
#         return self.run_swarms(objective=objective)

#     def _generate_code(self, boss1, create, unit_tests):
#         objective = f"""
#         Generate a Python program that meets the following product specification: {boss1} to create: {create}. Use the following unit tests as an evaluation score: {unit_tests}.
#         """
#         return self.run_swarms(objective=objective)

#     def run(self, create):
#         architecture = self._generate_architecture(create)
#         unit_tests = self._generate_tests(architecture)
#         code = self._generate_code(architecture, create, unit_tests)
#         return code, unit_tests




# class TheCompiler:
    # def __init__(self, create, api_key):
    #     self.create = create
    #     self.api_key = api_key

    # def generate_code(self):
    #     architect = Architect(self.create, self.api_key)
    #     architecture = architect.generate_architecture()

    #     test_creator = TestCreator(architecture, self.api_key)
    #     unit_tests = test_creator.generate_tests()

    #     code_generator = CodeGenerator(architecture, self.create, self.api_key, unit_tests)
    #     code = code_generator.generate_code()

    #     return code, unit_tests


# # Sample usage
# openai_api_key = os.environ['OPENAI_API_KEY'] = 'api key here'  # Replace with your OpenAI API key
# compiler = TheCompiler(create="an e-commerce website", api_key=openai_api_key)
# code, unit_tests = compiler.generate_code()



# create = "What do you want to create?"

# architect_prompt = f"""
# Create an architectural analysis specification in markdown in the most optimal programming language for {create}, provide the fastest, reliable architecture, and then break down that architecture into classes and algorithms needed to create {create}
# """

# objective = f"{architect_prompt}"


# #create a task
# task1 = boss_node.create_task(objective=objective)

# boss1 = boss_node.execute(task1)




# ##### 2nd agent - code generator
# generator_prompt = f"""
# Generate a Python program that meets the following product specification: {boss1} to create: {create}. Use the following unit tests as an evaluation score: {unit_tests}.
# """

# task2 = boss_node.create(objective=f"{generator_prompt}")

# boss2 = boss_node.execute(task2)

# ############### 3rd agent -- Unit test creator

# generator_prompt = f"""
# Generate a suite of unit tests for a Python program that meets the following product specification: {boss1}
# """

# task3 = boss_node.create(objective=f"{generator_prompt}")

# boss3 = boss_node.execute(task2)