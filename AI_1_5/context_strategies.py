import copy
from abc import ABC, abstractmethod
from typing import Any


class ContextStrategy(ABC):
    @abstractmethod
    def get_messages(self) -> list[dict]:
        pass

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        pass

    @abstractmethod
    def add_user_message(self, content: str) -> None:
        pass

    @abstractmethod
    def add_assistant_message(self, content: str) -> None:
        pass

    @abstractmethod
    def get_context_info(self) -> dict:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class SlidingWindowStrategy(ContextStrategy):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def get_messages(self) -> list[dict]:
        return self.messages.copy()

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim_window()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def _trim_window(self) -> None:
        system_msg = self.messages[0] if self.messages[0]["role"] == "system" else None
        non_system = [m for m in self.messages if m["role"] != "system"]
        
        if len(non_system) > self.window_size:
            non_system = non_system[-self.window_size:]
        
        if system_msg:
            self.messages = [system_msg] + non_system
        else:
            self.messages = non_system

    def get_context_info(self) -> dict:
        return {
            "strategy": "sliding_window",
            "window_size": self.window_size,
            "total_messages": len(self.messages),
            "non_system_messages": len([m for m in self.messages if m["role"] != "system"])
        }

    def clear(self) -> None:
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]


class StickyFactsStrategy(ContextStrategy):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.facts = {}
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self._init_default_facts()

    def _init_default_facts(self) -> None:
        self.facts = {
            "goal": "Conversation with user",
            "constraints": [],
            "preferences": {},
            "decisions": [],
            "agreements": []
        }

    def get_messages(self) -> list[dict]:
        facts_text = self._build_facts_prompt()
        result = [{"role": "system", "content": facts_text}]
        result.extend(self.messages[1:])
        return result

    def _build_facts_prompt(self) -> str:
        facts_lines = ["You are a helpful assistant.", "", "=== KEY FACTS ==="]
        for key, value in self.facts.items():
            if value:
                facts_lines.append(f"{key.upper()}: {value}")
        facts_lines.append("======================")
        return "\n".join(facts_lines)

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        
        if role == "user":
            self._extract_facts(content)
        
        self._trim_window()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def _extract_facts(self, text: str) -> None:
        text_lower = text.lower()
        
        if "want" in text_lower or "need" in text_lower or "goal" in text_lower or "создать" in text_lower:
            if self.facts.get("goal") == "Conversation with user" or not self.facts.get("goal"):
                self.facts["goal"] = text[:100]
        
        if ("not" in text_lower or "don't" in text_lower or "can't" in text_lower or "не" in text_lower) and ("want" in text_lower or "use" in text_lower or "мочь" in text_lower or "нельзя" in text_lower):
            self.facts["constraints"].append(text[:100])
        
        if "better" in text_lower or "prefer" in text_lower or "like" in text_lower or "лучше" in text_lower or "предпочитаю" in text_lower or "нравится" in text_lower:
            pref_key = f"preference_{len(self.facts['preferences']) + 1}"
            self.facts["preferences"][pref_key] = text[:100]
        
        if "decided" in text_lower or "agreed" in text_lower or "ok" in text_lower or "sure" in text_lower or "решили" in text_lower or "договорились" in text_lower:
            self.facts["agreements"].append(text[:100])

    def update_fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def _trim_window(self) -> None:
        non_system = [m for m in self.messages if m["role"] != "system"]
        
        if len(non_system) > self.window_size:
            non_system = non_system[-self.window_size:]
        
        self.messages = [self.messages[0]] + non_system

    def get_context_info(self) -> dict:
        return {
            "strategy": "sticky_facts",
            "window_size": self.window_size,
            "facts": self.facts.copy(),
            "total_messages": len(self.messages)
        }

    def clear(self) -> None:
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]
        self._init_default_facts()


class BranchingStrategy(ContextStrategy):
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.branches = {"main": []}
        self.current_branch = "main"
        self.checkpoints = {}
        
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]

    def get_messages(self) -> list[dict]:
        branch_messages = self.branches.get(self.current_branch, [])
        checkpoint = self.checkpoints.get(self.current_branch, self.messages[:1])
        
        return checkpoint + branch_messages

    def add_message(self, role: str, content: str) -> None:
        msg = {"role": role, "content": content}
        
        if self.current_branch == "main":
            self.messages.append(msg)
        
        self.branches[self.current_branch].append(msg)
        self._trim_branch()

    def add_user_message(self, content: str) -> None:
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        self.add_message("assistant", content)

    def create_checkpoint(self, name: str | None = None) -> dict:
        if name is None:
            name = f"checkpoint_{len(self.checkpoints) + 1}"
        
        current_messages = self.get_messages()
        self.checkpoints[name] = current_messages.copy()
        
        return {"checkpoint": name, "messages_count": len(current_messages)}

    def create_branch(self, branch_name: str, from_checkpoint: str | None = None) -> bool:
        if branch_name in self.branches:
            return False
        
        if from_checkpoint and from_checkpoint in self.checkpoints:
            self.branches[branch_name] = []
            self.checkpoints[branch_name] = self.checkpoints[from_checkpoint].copy()
        else:
            self.branches[branch_name] = []
            self.checkpoints[branch_name] = self.messages.copy()
        
        return True

    def switch_branch(self, branch_name: str) -> bool:
        if branch_name not in self.branches:
            return False
        self.current_branch = branch_name
        return True

    def list_branches(self) -> list[str]:
        return list(self.branches.keys())

    def get_current_branch(self) -> str:
        return self.current_branch

    def _trim_branch(self) -> None:
        branch_messages = self.branches[self.current_branch]
        
        if len(branch_messages) > self.window_size:
            self.branches[self.current_branch] = branch_messages[-self.window_size:]

    def get_context_info(self) -> dict:
        return {
            "strategy": "branching",
            "current_branch": self.current_branch,
            "branches": list(self.branches.keys()),
            "checkpoints": list(self.checkpoints.keys()),
            "messages_in_branch": len(self.branches.get(self.current_branch, []))
        }

    def clear(self) -> None:
        self.branches = {"main": []}
        self.current_branch = "main"
        self.checkpoints = {}
        self.messages = [{"role": "system", "content": "You are a helpful assistant."}]


class ContextManager:
    STRATEGIES = {
        "sliding": SlidingWindowStrategy,
        "sticky": StickyFactsStrategy,
        "branching": BranchingStrategy
    }

    def __init__(self, strategy_name: str = "sliding", **kwargs):
        strategy_class = self.STRATEGIES.get(strategy_name, SlidingWindowStrategy)
        self.strategy = strategy_class(**kwargs)
        self.strategy_name = strategy_name

    def set_strategy(self, strategy_name: str, **kwargs) -> bool:
        if strategy_name not in self.STRATEGIES:
            return False
        
        old_messages = self.strategy.get_messages()
        strategy_class = self.STRATEGIES[strategy_name]
        self.strategy = strategy_class(**kwargs)
        
        if old_messages:
            for msg in old_messages[1:]:
                self.strategy.add_message(msg["role"], msg["content"])
        
        self.strategy_name = strategy_name
        return True

    def get_messages(self) -> list[dict]:
        return self.strategy.get_messages()

    def add_user_message(self, content: str) -> None:
        self.strategy.add_user_message(content)

    def add_assistant_message(self, content: str) -> None:
        self.strategy.add_assistant_message(content)

    def get_context_info(self) -> dict:
        info = self.strategy.get_context_info()
        info["strategy_name"] = self.strategy_name
        return info

    def clear(self) -> None:
        self.strategy.clear()

    def __getattr__(self, name: str):
        if hasattr(self.strategy, name):
            return getattr(self.strategy, name)
        raise AttributeError(f"'{type(self.strategy).__name__}' has no attribute '{name}'")
