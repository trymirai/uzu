from dataclasses import dataclass

__all__ = ["HarmonyConfig", "Model"]


@dataclass(frozen=True)
class HarmonyConfig:
    tokenizer_url: str


@dataclass(frozen=True)
class Model:
    repo_id: str
    encodings: list[dict]
    tokenizer_repo_id: str | None = None
    harmony: HarmonyConfig | None = None
    is_translation: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        harmony_data = data.get("harmony")
        harmony = HarmonyConfig(**harmony_data) if harmony_data else None
        return cls(
            repo_id=data["repo_id"],
            encodings=data.get("encodings", []),
            tokenizer_repo_id=data.get("tokeuzer_repo_id"),
            harmony=harmony,
            is_translation=data.get("is_translation", False),
        )

    @property
    def name(self) -> str:
        parts = self.repo_id.split("/")
        return f"{parts[0]}_{parts[1]}"

    @property
    def tokenizer_name(self) -> str:
        repo_id = self.repo_id
        if self.tokenizer_repo_id is not None:
            repo_id = self.tokenizer_repo_id
        parts = repo_id.split("/")
        return f"{parts[0]}_{parts[1]}"
