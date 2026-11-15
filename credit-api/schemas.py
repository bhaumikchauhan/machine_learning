# The API's contract: what a client sends (one borrower) and what it gets back (a probability and a request id),
# and the log table that stores both. Same idea as CarInput / CarOutput / Car in fast-api/schemas.py.
#
# The JSON fields are snake_case; the model was trained on the Kaggle column names (some with hyphens, which aren't
# valid Python names). serialization_alias maps each field to its column: model_dump(by_alias=True) gives a dict
# keyed by column name, ready for a DataFrame. This mapping is the one place where API names meet model names.
import uuid
from datetime import datetime, timezone

from pydantic import ConfigDict
from sqlalchemy import DateTime
from sqlmodel import Field, SQLModel


class BorrowerInput(SQLModel):
    # extra="forbid": an unknown field is a 422, not silently ignored. Without it, a typo such as "monthly_incom"
    # would be dropped, monthly_income would default to None, and the model would score the borrower as
    # "income not reported": a wrong prediction with no error anywhere (like Jackson's FAIL_ON_UNKNOWN_PROPERTIES).
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{        # Kaggle test borrower 1; prefilled in /docs ("Try it out")
            "revolving_utilization": 0.8855, "age": 43, "times_30_59_days_late": 0, "debt_ratio": 0.1775,
            "monthly_income": 5700, "open_credit_lines": 4, "times_90_days_late": 0, "real_estate_loans": 0,
            "times_60_89_days_late": 0, "dependents": 0}]},
    )

    # Required: these columns were never blank in training, so the model never learned what a blank means for them.
    # The bounds reject impossible values (validation is part of the contract); within them the model takes
    # whatever it gets. Trees don't extrapolate: a value beyond the training range is treated like the largest
    # one seen.
    revolving_utilization: float = Field(ge=0, serialization_alias="RevolvingUtilizationOfUnsecuredLines",
                                         description="Credit card balances / credit limits (1.0 = at the limit)")
    age: int = Field(ge=18, le=120, serialization_alias="age")
    times_30_59_days_late: int = Field(ge=0, serialization_alias="NumberOfTime30-59DaysPastDueNotWorse")
    debt_ratio: float = Field(ge=0, serialization_alias="DebtRatio",
                              description="Monthly debt payments and living costs / monthly income")
    open_credit_lines: int = Field(ge=0, serialization_alias="NumberOfOpenCreditLinesAndLoans")
    times_90_days_late: int = Field(ge=0, serialization_alias="NumberOfTimes90DaysLate")
    real_estate_loans: int = Field(ge=0, serialization_alias="NumberRealEstateLoansOrLines")
    times_60_89_days_late: int = Field(ge=0, serialization_alias="NumberOfTime60-89DaysPastDueNotWorse")

    # Optional: blank for 20% (income) and 3% (dependents) of training borrowers. HistGradientBoosting learned
    # where a blank goes at each split, so None (JSON null, or the field left out) becomes NaN and is scored
    # like a blank in training.
    monthly_income: float | None = Field(default=None, ge=0, serialization_alias="MonthlyIncome")
    dependents: int | None = Field(default=None, ge=0, serialization_alias="NumberOfDependents")


class PredictionOutput(SQLModel):
    # Only the probability, not an approve/decline decision: the cut-off is a business choice (what a missed
    # default costs against a lost customer), and it should be set on calibrated probabilities (a roadmap leftover).
    probability: float = Field(ge=0, le=1, description="Probability of a serious late payment (90+ days) within 2 years")
    # The request id: the client keeps it, so the real outcome (known up to 2 years later) can be matched to this
    # prediction. No default: a forgotten id is an error, not a new random one that matches no record.
    id: uuid.UUID


class PredictionLog(BorrowerInput, PredictionOutput, table=True):
    # One row per prediction: the 10 inputs and the output (inherited), plus when, which model and how fast.
    # Used later for drift monitoring, replaying inputs through a new model, and measuring live AUC once outcomes
    # are known. Table classes skip validation; the values come from an already validated BorrowerInput.
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)    # overrides PredictionOutput.id
    # A real timestamp column (filter by date range in SQL), always UTC. timezone=True makes Postgres store the
    # time zone; SQLite has no timestamp type and stores text.
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc),
                                sa_type=DateTime(timezone=True))
    model_version: str = Field(description="The registered model and version, e.g. workspace.default.credit_risk/1 "
                                           "(records before MLflow have 12 characters of the model file's SHA-256)")
    latency: float = Field(description="Time spent in predict_proba, in milliseconds")