"""Tool registry for the Lab 5/6 agent.

Each tool exposes:
- a callable with simple Python args/kwargs returning a JSON-serialisable dict
- a `SCHEMA` dict in OpenAI/Ollama tool-calling format
"""


TOOL_REGISTRY = {}
TOOL_LOAD_ERRORS = {}


def _register(name, module_name, fn_name, schema_name="SCHEMA"):
    try:
        module = __import__(
            f"{__name__}.{module_name}",
            fromlist=[fn_name, schema_name],
        )
        TOOL_REGISTRY[name] = (
            getattr(module, fn_name),
            getattr(module, schema_name),
        )
    except Exception as e:
        TOOL_LOAD_ERRORS[name] = str(e)


_register("calculator", "calculator", "calculator")
_register("get_weather", "weather", "get_weather")
_register("web_search", "web_search", "web_search")
_register("analyze_image", "vision", "analyze_image")
_register("local_knowledge", "local_kb", "local_knowledge")
_register("nlp_tools", "nlp_tools", "nlp_tools")
_register("datetime_now", "datetime_tool", "datetime_now")

try:
    from . import moderation_tools as _mod
    TOOL_REGISTRY.update(_mod.TOOL_SPECS)
except Exception as e:
    TOOL_LOAD_ERRORS["moderation_tools"] = str(e)


def get_tools_payload(names=None):
    """Return the list of tool schemas to send to Ollama. If `names` is
    given, only those tools are included."""
    if names is None:
        return [schema for _, schema in TOOL_REGISTRY.values()]
    return [
        TOOL_REGISTRY[n][1] for n in names if n in TOOL_REGISTRY
    ]


def _tool_parameters(schema):
    return ((schema.get("function") or {}).get("parameters") or {})


def _sanitize_arguments(schema, arguments):
    params = _tool_parameters(schema)
    properties = params.get("properties") or {}
    required = params.get("required") or []
    if not isinstance(arguments, dict):
        arguments = {}

    if properties:
        arguments = {
            key: value for key, value in arguments.items()
            if key in properties
        }

    missing = [
        key for key in required
        if arguments.get(key) is None or arguments.get(key) == ""
    ]
    if missing:
        return None, f"Missing required argument(s): {', '.join(missing)}"
    return arguments, None


def call_tool(name, arguments):
    """Invoke a registered tool by name with a dict of arguments."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    fn, schema = TOOL_REGISTRY[name]
    arguments, error = _sanitize_arguments(schema, arguments)
    if error:
        return {"error": f"Bad arguments for {name}: {error}"}
    try:
        return fn(**arguments)
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:
        return {"error": f"{name} raised: {e}"}


def list_tool_names():
    return sorted(TOOL_REGISTRY.keys())
