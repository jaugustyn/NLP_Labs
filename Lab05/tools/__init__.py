"""Tool registry for the Lab 5 agent.

Each tool exposes:
- a callable with simple Python args/kwargs returning a JSON-serialisable dict
- a `SCHEMA` dict in OpenAI/Ollama tool-calling format
"""
from . import (
    calculator as _calc,
    weather as _weather,
    web_search as _web,
    vision as _vision,
    local_kb as _kb,
    nlp_tools as _nlp,
    datetime_tool as _dt,
)


# name -> (callable, schema)
TOOL_REGISTRY = {
    "calculator": (_calc.calculator, _calc.SCHEMA),
    "get_weather": (_weather.get_weather, _weather.SCHEMA),
    "web_search": (_web.web_search, _web.SCHEMA),
    "analyze_image": (_vision.analyze_image, _vision.SCHEMA),
    "local_knowledge": (_kb.local_knowledge, _kb.SCHEMA),
    "nlp_tools": (_nlp.nlp_tools, _nlp.SCHEMA),
    "datetime_now": (_dt.datetime_now, _dt.SCHEMA),
}


def get_tools_payload(names=None):
    """Return the list of tool schemas to send to Ollama. If `names` is
    given, only those tools are included."""
    if names is None:
        return [schema for _, schema in TOOL_REGISTRY.values()]
    return [
        TOOL_REGISTRY[n][1] for n in names if n in TOOL_REGISTRY
    ]


def call_tool(name, arguments):
    """Invoke a registered tool by name with a dict of arguments."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    fn, _ = TOOL_REGISTRY[name]
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        return fn(**arguments)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} raised: {e}"}


def list_tool_names():
    return sorted(TOOL_REGISTRY.keys())
