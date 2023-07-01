from swarms import worker_node, boss_node

create = "What do you want to create?"

architect_prompt = f"""
Create an architectural analysis specification in markdown in the most optimal programming language for {create}, provide the fastest, reliable architecture, and then break down that architecture into classes and algorithms needed to create {create}
"""

objective = f"{architect_prompt}"


#create a task
task1 = boss_node.create_task(objective=objective)

boss1 = boss_node.execute(task1)




##### 2nd agent - code generator
generator_prompt = f"""
Generate a Python program that meets the following product specification: {boss1} to create: {create}. Use the following unit tests as an evaluation score: {unit_tests}.
"""

task2 = boss_node.create(objective=f"{generator_prompt}")

boss2 = boss_node.execute(task2)

############### 3rd agent -- Unit test creator

generator_prompt = f"""
Generate a suite of unit tests for a Python program that meets the following product specification: {boss1}
"""

task2 = boss_node.create(objective=f"{generator_prompt}")

boss2 = boss_node.execute(task2)