import os
from typing import List, Optional
from enum import Enum
import dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.tabular_synthetic_data.openai import (
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_community.chat_models import ChatOpenAI

dotenv.load_dotenv()

class OpinionEnum(str, Enum):
    strongly_disagree = "strongly-disagree"
    disagree = "disagree"
    neutral = "neutral"
    agree = "agree"
    strongly_agree = "strongly-agree"

class Opinion(BaseModel):
    type: str = Field(default="opinion")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the opinion")
    created: str = Field(description="Creation date of the opinion entry")
    modified: str = Field(description="Last modification date of the opinion entry")
    explanation: Optional[str] = Field(default=None, description="An explanation of why the producer has this Opinion")
    authors: Optional[List[str]] = Field(default=None, description="The name of the author(s) of this Opinion")
    opinion: OpinionEnum = Field(description="The opinion about the STIX Object(s)")
    object_refs: List[str] = Field(description="The STIX Objects that the Opinion is being applied to")

    # Optional Common Properties
    created_by_ref: Optional[str] = Field(default=None, description="ID of the creator of this opinion")
    revoked: Optional[bool] = Field(default=None, description="Indicates whether the opinion has been revoked")
    labels: Optional[List[str]] = Field(default=None, description="List of terms describing this opinion")
    confidence: Optional[int] = Field(default=None, description="Confidence in the correctness of this opinion")
    lang: Optional[str] = Field(default=None, description="Language of the text content in this opinion")
    external_references: Optional[List[dict]] = Field(default=None, description="List of external references for this opinion")
    object_marking_refs: Optional[List[str]] = Field(default=None, description="List of marking definitions to be applied to this opinion")
    granular_markings: Optional[List[dict]] = Field(default=None, description="List of granular markings applied to this opinion")

examples = [
    {"example": """Type: opinion, Opinion: strongly-disagree, Object Refs: ["relationship--16d2358f-3b0d-4c88-b047-0da2f7ed4471"], Explanation: This doesn't seem like it is feasible. We've seen how PandaCat has attacked Spanish infrastructure over the last 3 years, so this change in targeting seems too great to be viable. The methods used are more commonly associated with the FlameDragonCrew., Authors: ["Alice Johnson"]"""},
    {"example": """Type: opinion, Opinion: agree, Object Refs: ["indicator--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f"], Explanation: Based on our internal telemetry, we've observed similar patterns of behavior that align with this indicator. The TTPs described are consistent with what we've seen in recent attacks., Authors: ["Bob Smith", "Charlie Davis"]"""},
    {"example": """Type: opinion, Opinion: neutral, Object Refs: ["malware--c7d1e135-8b34-549a-bb47-302f5cf998ed"], Explanation: While the malware capabilities described seem plausible, we haven't encountered this specific variant in our environment. More analysis is needed to confirm or refute these claims., Authors: ["Eva Green"]"""},
    {"example": """Type: opinion, Opinion: strongly-agree, Object Refs: ["attack-pattern--7e33a43e-e34b-40ec-89da-36c9bb2cacd5"], Explanation: This attack pattern perfectly matches what we've observed in recent incidents. The described techniques and procedures are spot-on, and we've seen this exact sequence of actions in multiple compromises., Authors: ["David Lee"]"""},
    {"example": """Type: opinion, Opinion: disagree, Object Refs: ["threat-actor--56f3f0db-b5d5-431c-ae56-c18f02caf500"], Explanation: The motivations attributed to this threat actor don't align with our intelligence. While some TTPs match, the overall profile seems inconsistent with our observations of their past behaviors and targeting preferences., Authors: ["Fiona White", "George Brown"]"""}
]

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

# Create a LangChain data generator
synthetic_data_generator = create_openai_data_generator(
    output_schema=Opinion,
    llm=ChatOpenAI(temperature=1, model='gpt-4'),
    prompt=prompt_template,
)

def generate_opinions(count: int) -> List[Opinion]:
    """
    Generate synthetic opinions.
    
    Args:
        count (int): Number of opinions to generate.
    
    Returns:
        List[Opinion]: List of generated opinions.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="opinion",
        extra="Create diverse opinions with realistic explanations, authors, and object references. Ensure that the object references are plausible STIX object identifiers. The opinions should cover a range of agreement levels and pertain to various types of STIX objects such as indicators, malware, threat actors, etc.",
        runs=count,
    )
    return synthetic_results
