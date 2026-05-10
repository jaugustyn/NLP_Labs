"""Multi-step tool-calling agent built on top of Ollama /api/chat."""
import json

import ollama_client
import tools as tools_mod
from config import AGENT_MODEL, MAX_AGENT_ITERATIONS


SYSTEM_PROMPT = (
    "You are an NLP assistant for a Telegram bot. You have access to "
    "tools (web_search, get_weather, analyze_image, calculator, "
    "local_knowledge, nlp_tools, datetime_now). "
    "When a user question requires a fact, computation, picture "
    "analysis, weather data, or a Lab 1-4 NLP operation, CALL the "
    "appropriate tool with structured arguments instead of guessing. "
    "You may call several tools in sequence (e.g. weather for two "
    "cities, or web_search followed by nlp_tools.summarize). "
    "After tool results arrive, INTERPRET them in a concise final "
    "answer to the user — do not just repeat raw JSON. "
    "Reply in the same language the user used (Polish or English). "
    "Keep answers short and useful."
)


def _normalise_tool_call(tc):
    """Ollama returns tool_calls in OpenAI-ish format; flatten to
    (name, arguments_dict, id)."""
    fn = tc.get("function") or {}
    name = fn.get("name") or tc.get("name") or ""
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {}
    return name, args, tc.get("id", "")


class Agent:
    def __init__(self, model=None, max_iters=MAX_AGENT_ITERATIONS,
                 enabled_tools=None, system_prompt=None):
        self.model = model or AGENT_MODEL
        self.max_iters = max_iters
        self.enabled_tools = enabled_tools  # None = all
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def run(self, user_text, images=None, history=None,
            extra_context=None):
        """Run one agent turn.

        Args:
            user_text: the user's message.
            images: optional list of local image paths attached.
            history: previous messages (list of {role, content}).
            extra_context: optional system-level note appended to the
                system prompt (e.g. "An image is attached at <path>").

        Returns dict with: answer, tool_trace, iterations, messages.
        """
        sys_content = self.system_prompt
        if extra_context:
            sys_content = sys_content + "\n\n" + extra_context

        messages = [{"role": "system", "content": sys_content}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})

        tool_payload = tools_mod.get_tools_payload(self.enabled_tools)
        tool_trace = []
        iterations = 0

        # First call gets the images (if any)
        first_call = True

        for i in range(self.max_iters):
            iterations = i + 1
            try:
                resp = ollama_client.chat(
                    messages=messages,
                    tools=tool_payload,
                    model=self.model,
                    images=images if first_call else None,
                )
            except Exception as e:
                return {
                    "answer": f"Agent error: {e}",
                    "tool_trace": tool_trace,
                    "iterations": iterations,
                    "messages": messages,
                    "error": str(e),
                }
            first_call = False

            msg = (resp or {}).get("message") or {}
            content = (msg.get("content") or "").strip()
            tool_calls = msg.get("tool_calls") or []

            # Append assistant message (preserve tool_calls for protocol)
            assistant_msg = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            if not tool_calls:
                return {
                    "answer": content or "(empty answer)",
                    "tool_trace": tool_trace,
                    "iterations": iterations,
                    "messages": messages,
                }

            # Execute tools and append their results
            for tc in tool_calls:
                name, args, tc_id = _normalise_tool_call(tc)
                result = tools_mod.call_tool(name, args)
                tool_trace.append({
                    "iteration": iterations,
                    "tool": name,
                    "arguments": args,
                    "result": result,
                })
                tool_msg = {
                    "role": "tool",
                    "name": name,
                    "content": json.dumps(result, ensure_ascii=False),
                }
                if tc_id:
                    tool_msg["tool_call_id"] = tc_id
                messages.append(tool_msg)

        # Loop budget exhausted
        return {
            "answer": (
                "(stopped after max iterations) "
                + ((messages[-1].get("content") or "")
                   if messages and messages[-1].get("role") == "assistant"
                   else "")
            ).strip(),
            "tool_trace": tool_trace,
            "iterations": iterations,
            "messages": messages,
            "error": "max_iterations_reached",
        }
