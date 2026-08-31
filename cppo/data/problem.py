from dataclasses import dataclass, field, asdict


@dataclass
class TestCase:
    input: str
    expected_output: str
    is_hidden: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(**d)


@dataclass
class Problem:
    id: str
    prompt: str
    test_cases: list[TestCase]
    domain: str
    difficulty: str
    source: str
    io_mode: str = "stdin"
    entry_point: str | None = None
    starter_code: str | None = None
    gold_answer: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_code(self) -> bool:
        return self.domain == "code"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test_cases"] = [tc.to_dict() for tc in self.test_cases]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Problem":
        d = dict(d)
        d["test_cases"] = [TestCase.from_dict(tc) for tc in d["test_cases"]]
        return cls(**d)
