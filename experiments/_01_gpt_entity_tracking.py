#%%
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
# prompt = "Write a one-sentence bedtime story about a unicorn."

# stream = client.responses.create(
#     model="gpt-4o", 
#     input=[
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ],
#     stream=True
# )

# events = []
# for event in stream:
#     events.append(event)
#     try:
#         print(event.delta)
#     except AttributeError:
#         pass


#%%
def prompt_gpt(prompt):
    response = client.responses.create(
        model="gpt-4o", 
        instructions="What is the value of x? Only give a number, do nothing else. Do not use commas in the number. For instance, '1,067' should be '1067'.",
        input=[
            {"role": "user", "content": prompt}
        ],
    )
    return response.output_text

#%%
def get_value(n, plus_val=123456789):
    prompt = f"""
    x = 0
    for i in range({n}):
        x += {plus_val}
    """
    return prompt_gpt(prompt)

# n = 2
# plus_val = 123456789
# i = 0 
# while i < 1000:
#     out = int(get_value(n, plus_val=plus_val))
#     if out == plus_val * n:
#         print(out)
#         i += 1
#         n *= 2
#         continue
#     else:
#         print(f"true value: {plus_val * n}, gpt value: {out}")
#         break

# n = 3802951800684688204490109616128
# exp = 100
# print(get_value((2**exp) * n), 2**(exp+1)*n)

#%%
for range_val in range(5, 15):
    prompt = f"""x = 0
for i in range({range_val}):
    if i % 2 == 0:
        x += 1
"""
    # print(prompt_gpt(prompt))
    out_val = exec(prompt + "print(x)")
    gpt_val = int(prompt_gpt(prompt))
    print(f"out_val: {out_val}, gpt_val: {gpt_val}, equal: {out_val == gpt_val}")

#%%
import torch as to
import numpy as np

def make_sequence(n, num_range=(0, 10)):
    plus_minus = np.random.randint(0, 2, n)
    nums = np.random.randint(*num_range, n)

    expressions = ["x = 0"]
    [expressions.append(f"x {'+' if plus_minus[i] else '-'}= {nums[i]}") for i in range(n)]
    code = '\n'.join(expressions)
    
    # Execute the code in a local namespace and return the final value of x
    local_vars = {}
    exec(code, {}, local_vars)
    return code, local_vars['x']

code, result = make_sequence(10)
print(f"Code:\n{code}\nResult: {result}")

#%%
# Test GPT's accuracy on sequences of different lengths
for n in range(1, 50):
    code, true_result = make_sequence(n)
    gpt_result = int(prompt_gpt(code))
    print(f"n={n:2d} | True: {true_result:4d} | GPT: {gpt_result:4d} | Match: {true_result == gpt_result}")


# %%

for n in range(1, 30):
    code, true_result = make_sequence(n, num_range=(0, 1000))
    gpt_result = int(prompt_gpt(code))
    print(f"n={n:2d} | True: {true_result:4d} | GPT: {gpt_result:4d} | Match: {true_result == gpt_result}")


#%%


for n in range(1, 30):
    code, true_result = make_sequence(n, num_range=(0, 10_000))
    gpt_result = int(prompt_gpt(code))
    print(f"n={n:2d} | True: {true_result:4d} | GPT: {gpt_result:4d} | Match: {true_result == gpt_result}")