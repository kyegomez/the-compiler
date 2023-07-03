from swarms import worker_node, boss_node

class Architect:
    def __init__(self, create):
        self.create = create

    def generate_architecture(self):
        objective = f"""
        Create an architectural analysis specification in markdown in the most optimal programming language for {self.create}, provide the fastest, reliable architecture, and then break down that architecture into classes and algorithms needed to create {self.create}
        """
        boss_node = boss_node()
        task = boss_node.create_task(objective=objective)
        return boss_node.execute(task)




class CodeGenerator:
    def __init__(self, boss1, create):
        self.boss1 = boss1
        self.create = create

    def generate_code(self):
        objective = f"""
        Generate a Python program that meets the following product specification: {self.boss1} to create: {self.create}. Use the following unit tests as an evaluation score: {unit_tests}.
        """
        boss_node = boss_node()
        task = boss_node.create_task(objective=objective)
        return boss_node.execute(task)



class TestCreator:
    def __init__(self, boss1):
        self.boss1 = boss1

    def generate_tests(self):
        objective = f"""
        Generate a suite of unit tests for a Python program that meets the following product specification: {self.boss1}
        """
        boss_node = boss_node()
        task = boss_node.create_task(objective=objective)
        return boss_node.execute(task)



class TheCompiler:
    def __init__(self, create):
        self.create = create

    def generate_code(self):
        architect = Architect(self.create)
        architecture = architect.generate_architecture()

        test_creator = TestCreator(architecture)
        unit_tests = test_creator.generate_tests()

        code_generator = CodeGenerator(architecture, self.create)
        code = code_generator.generate_code()

        return code, unit_tests

# create = "What do you want to create?"
# compiler = TheCompiler(create)
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