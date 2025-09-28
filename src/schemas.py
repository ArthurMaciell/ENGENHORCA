from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated

class QueryRequest(BaseModel):
    question: Annotated[str, Field(strip_whitespace=True, min_length=3, max_length=2000)]

class QueryResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: Optional[List[str]] = None
    route: Optional[Literal["tech", "product", "both"]] = None
    reason: Optional[str] = None
