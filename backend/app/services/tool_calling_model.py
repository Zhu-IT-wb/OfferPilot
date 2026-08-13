import json
from typing import Any, Dict, List, Protocol


import httpx

from app.models.tool_calling import (
    ModelOptions,
    ModelTurn,
    ModelToolCall,
    TokenUsage,
)
from app.services.llm_service import (
    LLMService,
    LLMRequestError,
    LLMConfigurationError,
)

class ToolCallingModel(Protocol):
    async def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        options: ModelOptions,
    ) -> ModelTurn:
        ...

class DeepSeekToolCallingModel(LLMService):
    async def complete(
          self,
          messages: List[Dict[str, Any]],
          tools: List[Dict[str, Any]],
          options: ModelOptions,
    ) -> ModelTurn:
        # 先检验apiKey有没有配置
        if not self.api_key:
            raise LLMConfigurationError("LLM API key is not configured.")
        try:
            response = await self._request_completion(messages, tools, options)
        except httpx.HTTPStatusError as exc:
            raise self._build_http_status_error(exc) from exc
        except httpx.TimeoutException as exc:
            raise self._build_timeout_error(exc) from exc
        except httpx.HTTPError as exc:
            raise self._build_http_error(exc) from exc
        return self._parse_response(
            response_data=response,
            selected_model=options.model or self.default_model,
        )

    async def _request_completion(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        options: ModelOptions,
    ) -> Dict[str, Any]:
     selected_model = options.model or self.default_model
     payload = {
            "model": selected_model,
            "messages": messages,
            "tools": tools,
            "temperature": options.temperature,
            "max_tokens": options.max_tokens,
            "stream": False,
        }
     if options.thinking is not None:
        payload["thinking"] = options.thinking
     if options.response_format is not None:
        payload["response_format"] = (
            options.response_format
        )
     if options.tool_choice is not None:
        payload["tool_choice"] = options.tool_choice
     return await self._post_chat_completions(payload)

    def _parse_response(
        self,
        response_data: Dict[str, Any],
        selected_model: str,
    ) -> ModelTurn:
        choice = self._first_choice(response_data)
        assistant_message =  choice.get("message")
        if not isinstance(assistant_message,dict):
            raise LLMRequestError(
                "LLM response does not contain a vaild message"
            )
        content = assistant_message.get("content")
        if content is None:
            content = ""
        if not isinstance(content,str):
            raise LLMRequestError(
                "LLM message content is not a string"
            )
        reasoning_content = assistant_message.get("reasoning_content")
        if (
            reasoning_content is not None
            and not isinstance(reasoning_content, str)
        ):
            raise LLMRequestError(
                "LLM reasoning_content is not a string."
            )

        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str):
            raise LLMRequestError(
                "LLM response does not contain finish_reason."
            )
        return ModelTurn(
            assistant_message=dict(assistant_message),
            tool_calls=self._parse_tool_calls(
                assistant_message.get("tool_calls")
            ),
            content=content,
            reasoning_content=reasoning_content,
            finish_reason=finish_reason,
            usage=self._parse_usage(response_data.get("usage")),
            provider=self.provider,
            model=selected_model,
        )

    # 判断choice字段是否正确
    @staticmethod
    def _first_choice(
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        choices = response_data.get("choices")

        if not isinstance(choices, list) or not choices:
            raise LLMRequestError(
                "LLM response does not contain choices."
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMRequestError(
                "LLM choice is not an object."
            )

        return choice

    @classmethod
    def _parse_tool_calls(
        cls,
        raw_tool_calls: Any,
    ) -> List[ModelToolCall]:
        if raw_tool_calls is None:
            return []

        if not isinstance(raw_tool_calls, list):
            raise LLMRequestError(
                "LLM tool_calls is not a list."
            )

        tool_calls: List[ModelToolCall] = []

        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                raise LLMRequestError(
                    "LLM tool call is not an object."
                )

            tool_call_id = raw_tool_call.get("id")
            function = raw_tool_call.get("function")

            if not isinstance(tool_call_id, str):
                raise LLMRequestError(
                    "LLM tool call does not contain an id."
                )

            if not isinstance(function, dict):
                raise LLMRequestError(
                    "LLM tool call does not contain a function."
                )

            function_name = function.get("name")
            if not isinstance(function_name, str):
                raise LLMRequestError(
                    "LLM tool call does not contain a name."
                )

            arguments = cls._parse_arguments(
                function.get("arguments")
            )

            tool_calls.append(
                ModelToolCall(
                    id=tool_call_id,
                    name=function_name,
                    arguments=arguments,
                )
            )

        return tool_calls

    @staticmethod
    def _parse_arguments(
        raw_arguments: Any,
    ) -> Dict[str, Any]:
        if isinstance(raw_arguments, dict):
            return dict(raw_arguments)

        if not isinstance(raw_arguments, str):
            raise LLMRequestError(
                "LLM tool arguments must be JSON."
            )

        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise LLMRequestError(
                "LLM tool arguments contain invalid JSON."
            ) from exc

        if not isinstance(arguments, dict):
            raise LLMRequestError(
                "LLM tool arguments must be an object."
            )

        return arguments

    @staticmethod
    def _parse_usage(raw_usage: Any) -> TokenUsage:
        if not isinstance(raw_usage, dict):
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )

        return TokenUsage(
            prompt_tokens=int(
                raw_usage.get("prompt_tokens") or 0
            ),
            completion_tokens=int(
                raw_usage.get("completion_tokens") or 0
            ),
            total_tokens=int(
                raw_usage.get("total_tokens") or 0
            ),
        )
