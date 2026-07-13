"""Agent loop: send→receive→execute tools→repeat (imported by main.py).

Simplified from OpenHarness query.py: no async, no streaming, no compact.
Single model, single-turn focus for paper validation experiments.
"""

from __future__ import annotations

import json
from typing import Optional

from engine.messages import (
    SystemMessage, UserMessage, AssistantMessage, ToolResultMsg,
    Message, messages_to_api,
)
from engine.api_client import call_api, extract_text, extract_tool_calls, extract_logprobs
from engine.tools import execute_tool, TOOL_DEFINITIONS


def run_query(
    system_prompt: str,
    user_prompt: str,
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    max_turns: int = 50,
    logprobs: bool = False,
    tools: bool = True,
    **api_kwargs,
) -> dict:
    """Run the agent loop: send messages, execute tool calls, repeat.

    Returns dict with keys: text, logprobs, messages, turns, tool_calls, usage.
    """
    messages: list[Message] = [
        SystemMessage(system_prompt),
        UserMessage(user_prompt),
    ]

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_call_count = 0
    final_text = ""
    final_logprobs = None
    tool_defs = TOOL_DEFINITIONS if tools else None
    turn = 0

    for turn in range(max_turns):
        api_messages = messages_to_api(messages)
        call_kwargs = dict(api_kwargs)
        if api_key:
            call_kwargs["api_key"] = api_key
        if model:
            call_kwargs["model"] = model

        response = call_api(
            api_messages, logprobs=logprobs, tools=tool_defs, **call_kwargs,
        )

        if response is None:
            final_text = "[API call failed]"
            break

        usage = response.get("usage", {})
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

        text = extract_text(response)
        tool_calls_list = extract_tool_calls(response)

        if logprobs:
            final_logprobs = extract_logprobs(response)

        assistant_msg = AssistantMessage(
            content=text, tool_calls=tool_calls_list, logprobs=final_logprobs,
        )
        messages.append(assistant_msg)

        if not tool_calls_list:
            final_text = text
            break

        for tc in tool_calls_list:
            fn_name = tc.get("function", {}).get("name", "")
            fn_args_str = tc.get("function", {}).get("arguments", "{}")
            tc_id = tc.get("id", f"call_{tool_call_count}")

            try:
                fn_args = json.loads(fn_args_str) if isinstance(fn_args_str, str) else fn_args_str
            except json.JSONDecodeError:
                fn_args = {}

            result = execute_tool(fn_name, fn_args)
            tool_call_count += 1

            messages.append(ToolResultMsg(
                tool_call_id=tc_id, name=fn_name, content=result.content,
            ))

        final_text = text
    else:
        final_text = final_text or "[Max turns reached]"

    return {
        "text": final_text,
        "logprobs": final_logprobs,
        "messages": messages,
        "turns": turn + 1,
        "tool_calls": tool_call_count,
        "usage": total_usage,
    }
