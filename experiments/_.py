# %%
from debug.generators import make_sequence
import sys

code, intermediates = make_sequence(5)
intermediate = intermediates
PROMPT_TEMPLATE = (
    "You are given a short Python program. "
    "Your task is to compute the final value of the variable x. "
    "Return only the integer, without commas, an equal sign, or any additional text. The integer should appear immediately after the word 'is: '.\n" 
    "```python\n{code}\n```\n"
    "The final value of x is: "
)

# print(code, "\n\n", intermediate, "\n\n", PROMPT_TEMPLATE.format(code=code))
print(PROMPT_TEMPLATE.format(code=code))




# %%
import numpy as np
print(np.array(intermediate))

x = 0
x = int(np.abs(x - 3))
x //= 1
x += 1
x = int(np.abs(x - 3))
x //= 2

print(x)