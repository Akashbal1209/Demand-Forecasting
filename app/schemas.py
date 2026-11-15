"""Request and response models (Pydantic) for the forecast API."""

from datetime import date

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    stock_code: str = Field(..., min_length=5, max_length=12, examples=["84077"],
                            description="Product stock code, e.g. 84077")
    weeks: int = Field(8, ge=1, le=8, description="How many weeks ahead to return (1-8)")


class ForecastWeek(BaseModel):
    week: date
    h: int = Field(..., description="Weeks ahead (1 = next week)")
    p10: float
    p50: float = Field(..., description="Most likely demand (median)")
    p90: float = Field(..., description="Demand exceeded in only ~10% of weeks")
    safety_stock: float = Field(..., description="p90 - p50")
    stock_target: float = Field(..., description="Recommended stock = p90 (~90% service)")


class HistoryWeek(BaseModel):
    week: date
    units: int


class ForecastResponse(BaseModel):
    stock_code: str
    name: str
    trained_until: date
    forecast: list[ForecastWeek]
    history: list[HistoryWeek]


class Product(BaseModel):
    stock_code: str
    name: str


class Health(BaseModel):
    status: str
    trained_until: date
    n_products: int
    test_metrics: dict