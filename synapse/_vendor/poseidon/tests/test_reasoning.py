import pytest
from poseidon.reasoning import solve, MathError, extract_math

@pytest.mark.parametrize("expression,expected", [("(17+5)*3", "66"), ("1/3 + 1/6", "1/2"), ("2^10", "1024"), ("3*x+7=22", "5"), ("2*(x-3)=8", "7"), ("0.1+0.2", "3/10")])
def test_exact_answers(expression, expected):
    assert solve(expression)["answer"] == expected
    assert solve(expression)["verified"]

@pytest.mark.parametrize("expression", ["__import__('os').system('dir')", "1/0", "2**1000", "x*x=4", "1/(x-1)=2", "[1]*10000", "x=x+1"])
def test_budget_and_grammar(expression):
    with pytest.raises(MathError):
        solve(expression)

def test_prompt_extraction():
    assert extract_math("What is (12+8)/4?") == "(12+8)/4"
    assert extract_math("Tell me about 42") is None
