# %%
from debug.sequence_generation import make_sequence
import sys

code, intermediate = make_sequence(10)
PROMPT_TEMPLATE = (
    "You are given a short Python program. "
    "Your task is to compute the final value of the variable x. "
    "Return only the integer, without commas, an equal sign, or any additional text. The integer should appear immediately after the word 'is: '.\n" 
    "```python\n{code}\n```\n"
    "The final value of x is: "
)

print(code, "\n\n", intermediate, "\n\n", PROMPT_TEMPLATE.format(code=code))


# %%
