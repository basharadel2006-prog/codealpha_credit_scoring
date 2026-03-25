import subprocess
import sys

# Input: option 2 (predict), then all the financial data, then option 3 (exit)
input_data = """2
75000
20000
0
0.4
60
5
40
debt_consolidation
employed
3
"""

proc = subprocess.Popen(
    [sys.executable, 'credit_scoring.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = proc.communicate(input=input_data)
print("STDOUT:")
print(stdout)
if stderr:
    print("\nSTDERR:")
    print(stderr)

# Check if file was created
import os
if os.path.exists('credit_predictions.txt'):
    print("\n✓ File created successfully!")
    with open('credit_predictions.txt', 'r') as f:
        print("\nFile contents:")
        print(f.read())
else:
    print("\n✗ File was not created")
