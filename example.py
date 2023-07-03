from the_compiler import Architect, Developer, UnitTester


# Initialize the components
architect = Architect()
developer = Developer()
unit_tester = UnitTester()

# Define the task
task = "Create a simple calculator in Python"

# Use the Architect to create the architecture and breakdown
architecture, breakdown = architect.create_architecture(task)

# Use the Developer to write the code
code = developer.write_code(breakdown)

# Use the UnitTester to create the tests
tests = unit_tester.create_tests(code)

# Print the code and tests
print("Code:", code)
print("Tests:", tests)