"""
FastAPI app: weekly demand forecasts (p10 / p50 / p90) for 100 products.

Run locally from the project root:
    uvicorn app.main:app --reload
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app.schemas import ForecastRequest, ForecastResponse, Health, Product
from src.forecast import forecast

BASE_DIR = Path(__file__).resolve().parent.parent          # project root
BUNDLE_PATH = BASE_DIR / "models" / "forecast_bundle.pkl"
INDEX_HTML = BASE_DIR / "app" / "static" / "index.html"
HISTORY_WEEKS = 26                                         # weeks of history sent for the chart

# ---------------------------------------------------------------- load once at startup
bundle = joblib.load(BUNDLE_PATH)
history = bundle["history"]
names = {str(code): name for code, name in bundle["product_names"].items()}

# the saved history is fixed, so all forecasts can be computed once
all_forecasts = forecast(bundle)
all_forecasts["safety_stock"] = (all_forecasts["p90"] - all_forecasts["p50"]).round(1)
all_forecasts["stock_target"] = all_forecasts["p90"]

app = FastAPI(
    title="Weekly Demand Forecast API",
    description="8-week demand forecasts (p10 / p50 / p90) and stock targets "
                "for 100 products of a UK online gift retailer.",
    version="1.0.0",
)


# ---------------------------------------------------------------- endpoints
@app.get("/", include_in_schema=False)
def home():
    """The Stock Planner web page."""
    return FileResponse(INDEX_HTML)


@app.get("/health", response_model=Health)
def health():
    return Health(
        status="ok",
        trained_until=bundle["trained_until"],
        n_products=len(names),
        test_metrics=bundle["test_metrics"],
    )


@app.get("/products", response_model=list[Product])
def products():
    return [Product(stock_code=c, name=n) for c, n in names.items()]


@app.post("/forecast", response_model=ForecastResponse)
def make_forecast(req: ForecastRequest):
    code = req.stock_code.strip().upper()
    if code not in names:
        raise HTTPException(status_code=404, detail=f"Unknown stock_code '{req.stock_code}'. "
                                                    "See GET /products for valid codes.")

    fc = all_forecasts[(all_forecasts["StockCode"] == code) & (all_forecasts["h"] <= req.weeks)]
    past = history[code].iloc[-HISTORY_WEEKS:]

    return ForecastResponse(
        stock_code=code,
        name=names[code],
        trained_until=bundle["trained_until"],
        forecast=fc[["week", "h", "p10", "p50", "p90", "safety_stock", "stock_target"]]
                 .to_dict(orient="records"),
        history=[{"week": w, "units": int(u)} for w, u in past.items()],
    )