from the_compiler import Architect, Developer, UnitTester
import os

openai_api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')   


#   Initialize the components
architect = Architect(openai_api_key=openai_api_key)
developer = Developer(openai_api_key=openai_api_key)
unit_tester = UnitTester(openai_api_key=openai_api_key)

#   Define the task
task = "Create a simple calculator in Python"

#   Use the Architect to create the architecture and breakdown
architecture, breakdown = architect.create_architecture(task)

#   Use the Developer to write the code
code = developer.write_code(breakdown)

#   Use the UnitTester to create the tests
tests = unit_tester.create_tests(code)

#   Print the code and tests
print("Code:", code)
print("Tests:", tests)

