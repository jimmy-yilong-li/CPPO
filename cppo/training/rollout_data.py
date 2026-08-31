from dataclasses import dataclass, field


@dataclass
class RolloutTokenData:
    """Token-level data for one generated sequence (plan or solver output)."""
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    old_logprobs: list[float]    # log p_old(token | prefix) at rollout time
    advantage: float = 0.0
    region: str = "planner"      # "planner" | "solver"
    problem_id: str = ""
    branch_idx: int = -1
    prompt_text: str = ""
    response_text: str = ""
    hit_max_tokens: bool = False
    skip_loss_reason: str = ""


@dataclass
class CPPORolloutBundle:
    """All data from one plan tuple rollout."""
    problem_id: str
    domain: str
    plan_data: RolloutTokenData | None = None
    scored_plan_text: str = ""
    raw_plan_text: str = ""
    solver_data: list[RolloutTokenData] = field(default_factory=list)
    raw_rm_score: float = 0.0
    rm_score_source: str = "unset"
    c_psi: float = 0.0
    j_psi: float = 0.0
    rm_winner: bool = False
    rm_reward: float = 0.0
    plan_winner: bool = False
    branch_rewards: list[float] = field(default_factory=list)
    branch_pass_rates: list[float] = field(default_factory=list)
    outcome_reward: float = 0.0
    plan_reward: float = 0.0
    methods: list[str] | None = None
    is_parseable: bool = False
