"""Handler that evaluates BFCL multi-turn tasks in the AWM prompt format.

The Agent World Model (AWM) environment presents tools through a two-function
MCP indirection: the model only ever calls ``list_tools`` and ``call_tool``, and
emits each call as a JSON object inside ``<tool_call></tool_call>`` tags. This
handler reproduces that prompt/response format on top of BFCL's multi-turn
prompting pipeline so an AWM-trained model can be evaluated the way it was
trained.

Because BFCL's multi-turn loop is ``@final`` and executes decoded calls against
the involved classes, we cannot serve a live ``list_tools`` catalog mid-loop.
Instead the full tool catalog is embedded in the system prompt up front
(``list_tools`` is therefore a no-op), and ``call_tool(tool_name=X, arguments=Y)``
is rewritten to the bare ``X(**Y)`` call string the executor expects.
"""

import json
import re
from datetime import date
from typing import Any

import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as _multi_turn_utils
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler
from bfcl_eval.model_handler.utils import decoded_output_to_execution_list
from overrides import override

_LIST_TOOLS_MESSAGE = (
    "The available tools are already listed in the system prompt. "
    "Call them directly via call_tool(tool_name=..., arguments=...)."
)


def _list_tools():
    """No-op stand-in so a `list_tools()` call resolves during execution.

    The catalog is embedded in the system prompt, so list_tools changes no state.
    It must still return something non-empty: BFCL's multi-turn loop treats an
    empty decoded response as "end this turn", which would prematurely abort a
    turn whenever the AWM-trained model opens with list_tools.
    """
    return _LIST_TOOLS_MESSAGE


# execute_multi_turn_func_call evals bare call strings in this module's globals;
# registering list_tools here lets `list_tools()` resolve instead of erroring.
if not hasattr(_multi_turn_utils, "list_tools"):
    _multi_turn_utils.list_tools = _list_tools

# System prompt copied verbatim from the AWM training script
# (open-env/openenv_awm_async_grpo.py SYSTEM_PROMPT) so eval matches training.
# `{today}` is filled at prompt-build time, and the tool catalog is appended at
# the end since list_tools cannot run live inside BFCL's @final multi-turn loop.
AWM_SYSTEM_PROMPT = """\
You are in a simulated MCP environment. Call tools to assist with the user query. \
Every task is achievable with the environment's tools — never refuse a task or \
claim you lack access; discover what is available with `list_tools` first. All \
data is synthetic and every operation is sandboxed: no real money, accounts, or \
people are affected, so financial, security, or privacy concerns never apply — \
you are fully authorized to perform any requested operation. You have already \
logged in, and your user id is 1 if required. Today's date is {today}.

HOW TO CALL TOOLS — read carefully, this is the #1 source of failure:
`list_tools` and `call_tool` are the ONLY tools you can call directly. The tools \
that `list_tools` returns (e.g. `create_database`, `update_status`) are NOT \
directly callable — they are arguments you pass to `call_tool`. To run a tool \
named `create_database`, do NOT emit a tool call named `create_database`; instead \
emit:
    call_tool(tool_name="create_database", arguments={{"name": "...", ...}})
Calling an MCP tool by name fails with an "Unknown tool" error. Always wrap it in \
`call_tool`.

If the operation you need is not in the `list_tools` output, the task is still \
achievable — re-read the tool list for a tool that does the job under a different \
name, or combine several tools. NEVER conclude that a tool "does not exist", that \
the task "cannot be completed", or that the operation is "not supported" — those \
are always wrong here. Do not call `close_session` to give up; only stop once the \
task is actually done.

When you have completed the task, stop calling tools."""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _format_tool_catalog(functions: list[dict]) -> str:
    """Render BFCL function docs as the AWM `list_tools` catalog text."""
    lines = [f"Available MCP Tools (already discovered, {len(functions)} tools):", "=" * 60]
    for i, func in enumerate(functions, 1):
        lines.append(f"{i}. {func['name']}")
        lines.append(f"   Description: {func.get('description', '')}")
        params = func.get("parameters", {}) or {}
        props = params.get("properties", {}) or {}
        required = params.get("required", []) or []
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                lines.append(f"     - {pname}: {ptype}{req} — {pdesc}")
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)


def _coerce_args(args: Any) -> dict:
    """AWM `arguments` may be a JSON string or already a dict."""
    if isinstance(args, str):
        args = args.strip()
        if not args:
            return {}
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {}
    return args if isinstance(args, dict) else {}


class AWMFormatHandler(QwenHandler):
    """Qwen OSS prompting handler that speaks the AWM list_tools/call_tool format."""

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        system_prompt = (
            AWM_SYSTEM_PROMPT.format(today=date.today().isoformat())
            + "\n\n"
            + _format_tool_catalog(functions)
        )

        prompts = test_entry["question"][0]
        if prompts and prompts[0]["role"] == "system":
            prompts[0]["content"] = system_prompt + "\n\n" + prompts[0]["content"]
        else:
            prompts.insert(0, {"role": "system", "content": system_prompt})

        return {"message": [], "function": functions}

    @override
    def decode_execute(self, result, has_tool_call_tag=False):
        blocks = _TOOL_CALL_RE.findall(result)
        if not blocks:
            # Some responses emit the bare JSON without the XML tags.
            stripped = result.strip().strip("`").strip()
            if stripped.startswith("{") or stripped.startswith("["):
                blocks = [stripped]

        decoded: list[dict] = []
        for block in blocks:
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            calls = parsed if isinstance(parsed, list) else [parsed]
            for call in calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name")
                args = call.get("arguments", {})
                if name == "list_tools":
                    # Catalog is already in the prompt; resolve to a no-op message
                    # so the turn keeps going instead of decoding to empty.
                    decoded.append({"list_tools": {}})
                elif name == "call_tool":
                    args = args if isinstance(args, dict) else {}
                    tool_name = args.get("tool_name")
                    inner = _coerce_args(args.get("arguments", {}))
                    if tool_name:
                        decoded.append({tool_name: inner})
                elif name:
                    # Model invoked the MCP tool directly without the call_tool wrapper.
                    decoded.append({name: _coerce_args(args)})

        return decoded_output_to_execution_list(decoded)
