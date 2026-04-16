import re

COT_START = "<|cot_start|>"
COT_END = "<|cot_end|>"


class AlpamayoFormatReward:
    """Minimal reward that only checks CoC formatting."""

    def __init__(self, config):
        self.require_non_empty = bool(config.get("require_non_empty", True))
        self.pattern = re.compile(
            rf"{re.escape(COT_START)}(?P<cot>.*?){re.escape(COT_END)}",
            re.DOTALL,
        )

    def get_reward(self, completions: list[str], answers: list[dict] | list[str]) -> list[float]:
        rewards: list[float] = []
        for text in completions:
            # Reward worker decodes only generated continuation tokens, while the
            # Alpamayo prompt already pre-fills the assistant turn with
            # ``<|cot_start|>``. Reconstruct the full formatting surface before
            # applying the existing format check.
            normalized_text = text if COT_START in text else f"{COT_START}{text}"
            match = self.pattern.search(normalized_text)
            if not match:
                rewards.append(0.0)
                continue
            cot = match.group("cot").strip()
            if self.require_non_empty and not cot:
                rewards.append(0.0)
                continue
            reward = 1.0
            if cot[-1] in ".":
                reward += 0.5
            rewards.append(reward)
        return rewards
