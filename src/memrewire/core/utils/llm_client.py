"""
LLM Client - Handles all LLM interactions
"""
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Dict, Any, Optional, Iterator
from openai import OpenAI
from memwire.core.settings import settings as config


_USAGE_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar(
    "memwire_usage_context", default={}
)


# Keep the raw provider total for backward compatibility, while also carrying
# cache-aware fields so a provider prefix is not mistaken for retrieved
# memory. Missing cache details remain visible through their detail-call counts.
_USAGE_SUM_FIELDS = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "cache_write_prompt_tokens",
    "cached_prompt_detail_calls",
    "cache_write_prompt_detail_calls",
    "uncached_prompt_tokens",
    "uncached_plus_completion_tokens",
)


class LLMClient:
    """
    Unified LLM client interface
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING

        self.usage_log_path = getattr(config, "TOKEN_USAGE_LOG_PATH", None)
        if self.usage_log_path:
            os.makedirs(os.path.dirname(self.usage_log_path) or ".", exist_ok=True)
        self._usage_lock = threading.Lock()
        self._usage_events: List[Dict[str, Any]] = []
        self._tokenizer_lock = threading.Lock()
        self._tokenizer_loaded = False
        self._tokenizer = None
        self._tokenizer_error: Optional[Exception] = None

        # Initialize OpenAI client with optional base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            print(f"Using custom OpenAI base URL: {self.base_url}")

        if self.enable_thinking:
            print(f"Deep thinking mode enabled")

        # self.client = OpenAI(**client_kwargs)
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 1,
        stage: str = "unspecified"
    ) -> str:
        """
        Standard chat completion with optional thinking mode and retry mechanism
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(getattr(config, "MAX_TOKENS", 4096)),
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Enable thinking mode if configured (for Qwen and compatible models only)
        # Only add enable_thinking parameter for Qwen API (identified by base_url)
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        is_qwen_model = "qwen" in (self.model or "").lower()
        template_kwargs = {}
        
        if is_qwen_api:
            # Qwen API requires explicit enable_thinking parameter
            # - Streaming + thinking: enable_thinking=True
            # - Non-streaming: enable_thinking=False (required, not optional)
            # - JSON format: enable_thinking=False (incompatible with thinking mode)
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                # Explicitly set to False for non-streaming calls or JSON format
                kwargs["extra_body"] = {"enable_thinking": False}
        elif is_qwen_model:
            template_kwargs["enable_thinking"] = bool(
                self.enable_thinking and not response_format
            )
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": template_kwargs["enable_thinking"]
                }
            }
        # For OpenAI and other APIs, don't add extra_body parameters

        tokenizer_prompt_tokens = self.count_prompt_tokens(
            messages,
            template_kwargs=template_kwargs,
        )
        input_limit = getattr(config, "ANSWER_MAX_INPUT_TOKENS", None)
        if stage == "answer" and input_limit:
            if tokenizer_prompt_tokens is None:
                raise RuntimeError(
                    "ANSWER_MAX_INPUT_TOKENS is set but the configured local tokenizer is unavailable"
                )
            if tokenizer_prompt_tokens > int(input_limit):
                raise RuntimeError(
                    f"Complete answer prompt is {tokenizer_prompt_tokens} tokens, "
                    f"above ANSWER_MAX_INPUT_TOKENS={int(input_limit)}"
                )

        # Retry mechanism
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    # vLLM/OpenAI sends usage in the final stream chunk only when
                    # explicitly requested.  Keep streaming behavior unchanged.
                    kwargs["stream_options"] = {"include_usage": True}
                    content, usage = self._handle_streaming_response(**kwargs)
                    self._record_usage(
                        messages=messages,
                        response_text=content,
                        usage=usage,
                        stage=stage,
                        attempt=attempt + 1,
                        streaming=True,
                        tokenizer_prompt_tokens=tokenizer_prompt_tokens,
                    )
                    return content
                else:
                    response = self.client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content
                    self._record_usage(
                        messages=messages,
                        response_text=content,
                        usage=getattr(response, "usage", None),
                        stage=stage,
                        attempt=attempt + 1,
                        streaming=False,
                        tokenizer_prompt_tokens=tokenizer_prompt_tokens,
                    )
                    return content
                
                # kwargs["stream"] = True
                # return self._handle_streaming_response(**kwargs)
                    
            except Exception as e:
                last_exception = e
                self._record_usage(
                    messages=messages,
                    response_text="",
                    usage=None,
                    stage=stage,
                    attempt=attempt + 1,
                    success=False,
                    error=str(e),
                    streaming=self.use_streaming,
                    tokenizer_prompt_tokens=tokenizer_prompt_tokens,
                )
                if attempt < max_retries - 1:
                    import time
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(f"LLM API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"LLM API call failed after {max_retries} attempts: {e}")
        
        # If all retries failed, raise the last exception.
        raise last_exception

    def _handle_streaming_response(self, **kwargs):
        """
        Handle streaming response and collect full content
        """
        full_content = []
        usage = None
        stream = self.client.chat.completions.create(**kwargs)

        # for chunk in stream:
        #     if chunk.choices is not None:
        #         print(chunk.choices[0].delta.content)
        
        # print('---------')

        for chunk in stream:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            # print(chunk)
            # fix list index out of range
            choices = getattr(chunk, "choices", None) or []
            if len(choices) > 0 and choices[0].delta.content is not None:
                content = choices[0].delta.content
                full_content.append(content)
                # print(full_content)
                # Optional: print streaming content in real-time
                # print(content, end='', flush=True)
        # print(full_content)
        print()
        return ''.join(full_content), usage

    @contextmanager
    def usage_scope(self, **context: Any) -> Iterator[None]:
        """Attach run/question metadata to all calls in the current thread."""
        previous = _USAGE_CONTEXT.get()
        merged = dict(previous)
        merged.update({key: value for key, value in context.items() if value is not None})
        token = _USAGE_CONTEXT.set(merged)
        try:
            yield
        finally:
            _USAGE_CONTEXT.reset(token)

    def usage_context(self) -> Dict[str, Any]:
        """Return the current thread's run/question metadata."""
        return dict(_USAGE_CONTEXT.get())

    def count_prompt_tokens(
        self,
        messages: List[Dict[str, str]],
        template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        """Count the complete prompt, including the configured model chat template."""
        model_path = getattr(config, "TOKENIZER_MODEL_PATH", None)
        if not model_path:
            return None
        with self._tokenizer_lock:
            if not self._tokenizer_loaded:
                self._tokenizer_loaded = True
                try:
                    from transformers import AutoTokenizer

                    self._tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                except Exception as error:
                    self._tokenizer_error = error
            if self._tokenizer_error is not None:
                raise RuntimeError(
                    f"Could not load local tokenizer from {model_path}: "
                    f"{self._tokenizer_error}"
                ) from self._tokenizer_error
            tokenizer = self._tokenizer

        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(template_kwargs or {}),
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    @staticmethod
    def _usage_values(usage: Any) -> Optional[Dict[str, int]]:
        if usage is None:
            return None

        def value(name: str):
            if isinstance(usage, dict):
                return usage.get(name)
            return getattr(usage, name, None)

        def nested(container: Any, name: str) -> int:
            if container is None:
                return 0
            if isinstance(container, dict):
                return int(container.get(name) or 0)
            return int(getattr(container, name, 0) or 0)

        def has_nested(container: Any, name: str) -> bool:
            if isinstance(container, dict):
                return name in container
            return container is not None and hasattr(container, name)

        prompt = value("prompt_tokens")
        completion = value("completion_tokens")
        total = value("total_tokens")
        if prompt is None and completion is None and total is None:
            return None
        prompt = int(prompt or 0)
        completion = int(completion or 0)
        total = int(total if total is not None else prompt + completion)
        prompt_details = value("prompt_tokens_details")
        completion_details = value("completion_tokens_details")
        cached_prompt = nested(prompt_details, "cached_tokens")
        cache_write_prompt = nested(prompt_details, "cache_write_tokens")
        uncached_prompt = max(0, prompt - cached_prompt)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
            "cached_prompt_tokens": cached_prompt,
            "cache_write_prompt_tokens": cache_write_prompt,
            "cached_prompt_detail_calls": int(has_nested(prompt_details, "cached_tokens")),
            "cache_write_prompt_detail_calls": int(has_nested(prompt_details, "cache_write_tokens")),
            "uncached_prompt_tokens": uncached_prompt,
            "uncached_plus_completion_tokens": uncached_prompt + completion,
            "reasoning_tokens": nested(completion_details, "reasoning_tokens"),
        }

    @staticmethod
    def _estimate_tokens(messages: List[Dict[str, str]], response_text: str) -> Dict[str, int]:
        """Conservative fallback only; estimates are explicitly marked as such."""
        prompt_chars = sum(len(str(message.get("content", ""))) for message in messages)
        completion_chars = len(response_text or "")
        prompt = max(1, round(prompt_chars / 4)) if prompt_chars else 0
        completion = max(1, round(completion_chars / 4)) if completion_chars else 0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cached_prompt_tokens": 0,
            "cache_write_prompt_tokens": 0,
            "cached_prompt_detail_calls": 0,
            "cache_write_prompt_detail_calls": 0,
            "uncached_prompt_tokens": prompt,
            "uncached_plus_completion_tokens": prompt + completion,
        }

    def _record_usage(
        self,
        messages: List[Dict[str, str]],
        response_text: str,
        usage: Any,
        stage: str,
        attempt: int,
        success: bool = True,
        error: Optional[str] = None,
        streaming: bool = False,
        tokenizer_prompt_tokens: Optional[int] = None,
    ) -> None:
        exact_usage = self._usage_values(usage)
        if exact_usage is None:
            token_values = self._estimate_tokens(messages, response_text)
            usage_source = "char_estimate"
        else:
            token_values = exact_usage
            usage_source = "api"
        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "model": self.model,
            "request_type": "chat.completions",
            "stage": stage,
            "context": dict(_USAGE_CONTEXT.get()),
            "attempt": attempt,
            "streaming": streaming,
            "success": success,
            "usage_source": usage_source,
            "usage_exact": usage_source == "api",
            "tokenizer_prompt_tokens": tokenizer_prompt_tokens,
            "server_prompt_minus_tokenizer": (
                int(token_values["prompt_tokens"]) - int(tokenizer_prompt_tokens)
                if tokenizer_prompt_tokens is not None and usage_source == "api"
                else None
            ),
            "prompt_chars": sum(len(str(message.get("content", ""))) for message in messages),
            "completion_chars": len(response_text or ""),
            **token_values,
        }
        if error:
            event["error"] = error
        with self._usage_lock:
            self._usage_events.append(event)
            if self.usage_log_path:
                with open(self.usage_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, ensure_ascii=True) + "\n")

    def usage_summary(self, question_count: Optional[int] = None) -> Dict[str, Any]:
        """Return raw and candidate cost views without discarding alternate readings."""
        with self._usage_lock:
            events = list(self._usage_events)
        stages: Dict[str, Dict[str, int]] = {}
        exact_calls = estimated_calls = 0
        tokenizer_prompt_tokens_by_stage: Dict[str, int] = {}
        tokenizer_prompt_count_calls_by_stage: Dict[str, int] = {}
        server_prompt_minus_tokenizer = 0
        server_prompt_delta_calls = 0
        for event in events:
            if event.get("success"):
                bucket = stages.setdefault(event["stage"], {
                    field: 0 for field in _USAGE_SUM_FIELDS
                })
                bucket["calls"] += 1
                for key in _USAGE_SUM_FIELDS[1:]:
                    bucket[key] += int(event.get(key, 0))
                if event.get("usage_exact"):
                    exact_calls += 1
                else:
                    estimated_calls += 1
                tokenizer_count = event.get("tokenizer_prompt_tokens")
                if tokenizer_count is not None:
                    stage = event["stage"]
                    tokenizer_prompt_tokens_by_stage[stage] = (
                        tokenizer_prompt_tokens_by_stage.get(stage, 0)
                        + int(tokenizer_count)
                    )
                    tokenizer_prompt_count_calls_by_stage[stage] = (
                        tokenizer_prompt_count_calls_by_stage.get(stage, 0)
                        + 1
                    )
                delta = event.get("server_prompt_minus_tokenizer")
                if delta is not None:
                    server_prompt_minus_tokenizer += int(delta)
                    server_prompt_delta_calls += 1

        def totals(prefixes):
            selected = [bucket for name, bucket in stages.items()
                        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)]
            return {
                field: sum(item[field] for item in selected)
                for field in _USAGE_SUM_FIELDS
            }

        maintenance = totals(("maintenance",))
        retrieval = totals(("retrieval",))
        answer = totals(("answer",))
        judge = totals(("judge",))
        online = {
            field: retrieval[field] + answer[field]
            for field in _USAGE_SUM_FIELDS
        }
        denominator = question_count or 0
        return {
            "definition": {
                "candidate_online_token_cost": "retrieval + answer; maintenance/build excluded",
                "candidate_total_token_cost": "maintenance + retrieval + answer",
                "token_units": "prompt + completion tokens",
                "exactness": "api usage is exact; char_estimate is fallback only",
                "uncached_plus_completion_token_view": (
                    "prompt tokens minus reported cached prompt tokens, plus completion tokens; "
                    "detail-call counts expose missing cache annotations. Cached tokens are still "
                    "billed under provider cache pricing, so this is not a monetary cost"
                ),
            },
            "calls_recorded": len(events),
            "successful_calls": sum(1 for event in events if event.get("success")),
            "failed_attempts_recorded": sum(1 for event in events if not event.get("success")),
            "exact_calls": exact_calls,
            "estimated_calls": estimated_calls,
            "tokenizer_prompt_tokens_by_stage": tokenizer_prompt_tokens_by_stage,
            "tokenizer_prompt_count_calls_by_stage": tokenizer_prompt_count_calls_by_stage,
            "server_prompt_minus_tokenizer_total": server_prompt_minus_tokenizer,
            "server_prompt_minus_tokenizer_calls": server_prompt_delta_calls,
            "by_stage": stages,
            "maintenance": maintenance,
            "retrieval": retrieval,
            "answer": answer,
            "judge": judge,
            "candidate_online_token_cost": online,
            "candidate_total_token_cost": {
                key: maintenance[key] + online[key]
                for key in _USAGE_SUM_FIELDS
            },
            "candidate_online_uncached_plus_completion": {
                key: online[key]
                for key in ("calls", "cached_prompt_detail_calls", "cache_write_prompt_detail_calls", "uncached_prompt_tokens", "completion_tokens", "uncached_plus_completion_tokens")
            },
            "candidate_total_uncached_plus_completion": {
                key: maintenance[key] + online[key]
                for key in ("calls", "cached_prompt_detail_calls", "cache_write_prompt_detail_calls", "uncached_prompt_tokens", "completion_tokens", "uncached_plus_completion_tokens")
            },
            "question_count": denominator,
            "candidate_online_average_total_tokens_per_question": (
                online["total_tokens"] / denominator if denominator else None
            ),
            "candidate_total_average_tokens_per_question": (
                (maintenance["total_tokens"] + online["total_tokens"]) / denominator
                if denominator else None
            ),
            "candidate_online_average_uncached_plus_completion_per_question": (
                online["uncached_plus_completion_tokens"] / denominator if denominator else None
            ),
            "candidate_total_average_uncached_plus_completion_per_question": (
                (maintenance["uncached_plus_completion_tokens"] + online["uncached_plus_completion_tokens"]) / denominator
                if denominator else None
            ),
        }

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
