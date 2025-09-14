from pydantic import BaseModel, Field


class FilterPage(BaseModel):
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, gt=0, le=1000, description="Maximum number of items to return")
