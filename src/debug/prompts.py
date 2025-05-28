"""Prompt templates for debug experiments."""

# Exception handling path tracking
EXCEPTION_HANDLING = """You are given a short Python program with exception handling. Your task is to determine the final value of the variable 'result'.
At each step, the program may or may not raise a ZeroDivisionError. Follow the execution path carefully.
Return only the final value, without any additional text. The answer should appear immediately after the word 'is: '.

```python
{code}
```

The final value of `result` is: """ 


# Basic arithmetic tracking
RANGE_TRACKING = """You are given a short Python program. Your task is to compute the final value of the variable x.
Return only the integer, without commas, an equal sign, or any additional text. The integer should appear immediately after the word 'is: '.

```python
{code}
```

The final value of x is: """

# Boolean logic tracking  
BOOLEAN_LOGIC = """You are given a short Python program with boolean operations. Your task is to determine the final boolean value.
Return only 'True' or 'False', without any additional text. The answer should appear immediately after the word 'is: '.

```python
{code}
```

The final boolean value is: """

# Step-by-step reasoning
STEP_BY_STEP = """You are given a short Python program. Let's trace through it step by step, then give the final answer.

```python
{code}
```

Let me trace through this:
1. First, I'll identify the initial state
2. Then I'll follow each operation
3. Finally, I'll give the result

The final value of x is: """

# Chain of thought reasoning
CHAIN_OF_THOUGHT = """You are given a short Python program. Think step by step about what this code does.

```python
{code}
```

Let me work through this step by step:

The final value of x is: """

# Minimal prompt
MINIMAL = """Compute the final value of x:

```python
{code}
```

Answer: """

# Entity tracking (generic)
ENTITY_TRACKING = """You are given a short Python program. Your task is to determine the final state.
Return only the requested value, without any additional text. The answer should appear immediately after the word 'is: '.

```python
{code}
```

The final result is: """

# Variable binding resolution
VARIABLE_BINDING = """You are given a Python program with variable assignments followed by a query. Your task is to trace through the variable assignments and determine what value the queried variable refers to.
Return only the integer value, without any additional text. The answer should appear immediately after the word 'is: '.

```python
{code}
```

The value is: """
