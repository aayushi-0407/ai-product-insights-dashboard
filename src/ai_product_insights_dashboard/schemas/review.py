"""Review record schema definitions."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Review:
    review_id: str
    asin: str
    review_text: str
    rating: int
    review_date: datetime
