from dataclasses import dataclass

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel


@dataclass
class MyDeps:
    foo: int
    bar: int


@dataclass
class WrongDeps:
    baz: str


@dataclass
class MySubDeps(MyDeps):
    extra: str = 'hello'


agent = Agent(TestModel(), deps_type=MyDeps)


@agent.tool
async def example_tool(ctx: RunContext[MyDeps]) -> str:
    return f'{ctx.deps}'


def test_deps_used():
    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.output == '{"example_tool":"MyDeps(foo=1, bar=2)"}'


def test_deps_wrong_type_raises_error():
    """Passing a deps object of the wrong type should raise UserError immediately."""
    with pytest.raises(UserError, match='deps must be an instance of MyDeps, got WrongDeps'):
        agent.run_sync('foobar', deps=WrongDeps(baz='wrong'))


def test_deps_subclass_accepted():
    """Subclasses of the declared deps_type should be accepted."""
    result = agent.run_sync('foobar', deps=MySubDeps(foo=1, bar=2, extra='world'))
    assert 'MySubDeps' in result.output


def test_deps_none_when_deps_type_is_none():
    """An agent with no deps_type (default NoneType) should accept None without error."""
    no_deps_agent = Agent(TestModel())
    result = no_deps_agent.run_sync('hello')
    assert result.output is not None

def test_deps_override():
    with agent.override(deps=MyDeps(foo=3, bar=4)):
        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.output == '{"example_tool":"MyDeps(foo=3, bar=4)"}'

        with agent.override(deps=MyDeps(foo=5, bar=6)):
            result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
            assert result.output == '{"example_tool":"MyDeps(foo=5, bar=6)"}'

        result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
        assert result.output == '{"example_tool":"MyDeps(foo=3, bar=4)"}'

    result = agent.run_sync('foobar', deps=MyDeps(foo=1, bar=2))
    assert result.output == '{"example_tool":"MyDeps(foo=1, bar=2)"}'
