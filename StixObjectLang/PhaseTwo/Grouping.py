import os
from typing import List, Optional
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

class Grouping(BaseModel):
    type: str = Field(default="grouping")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the grouping")
    created: str = Field(description="Creation date of the grouping entry")
    modified: str = Field(description="Last modification date of the grouping entry")
    name: Optional[str] = Field(default=None, description="Name used to identify the Grouping")
    description: Optional[str] = Field(default=None, description="More details and context about the Grouping, potentially including its purpose and key characteristics")
    context: str = Field(description="Short descriptor of the particular context shared by the content referenced by the Grouping")
    object_refs: List[str] = Field(description="List of STIX Object identifiers that this grouping contains")

    # Optional Common Properties
    created_by_ref: Optional[str] = Field(default=None, description="ID of the creator of this grouping")
    revoked: Optional[bool] = Field(default=None, description="Indicates whether the grouping has been revoked")
    labels: Optional[List[str]] = Field(default=None, description="List of terms describing this grouping")
    confidence: Optional[int] = Field(default=None, description="Confidence in the correctness of this grouping")
    lang: Optional[str] = Field(default=None, description="Language of the text content in this grouping")
    external_references: Optional[List[dict]] = Field(default=None, description="List of external references for this grouping")
    object_marking_refs: Optional[List[str]] = Field(default=None, description="List of marking definitions to be applied to this grouping")
    granular_markings: Optional[List[dict]] = Field(default=None, description="List of granular markings applied to this grouping")

examples = [
    {"example": """Type: grouping, Name: APT29 Campaign Analysis, Context: campaign, Description: A collection of STIX objects related to the APT29 campaign targeting government entities, Object Refs: ["indicator--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f", "malware--31b940d4-6f7f-459a-80ea-9c1f17b5891b", "attack-pattern--7e33a43e-e34b-40ec-89da-36c9bb2cacd5"]"""},
    {"example": """Type: grouping, Name: Ransomware Incident Response, Context: incident, Description: Grouping of artifacts and observables related to a recent ransomware attack on a financial institution, Object Refs: ["indicator--a932fcc6-e032-476c-826f-cb970a5a1ade", "observed-data--b67d30ff-02ac-498a-92f9-32f845f448cf", "course-of-action--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f"]"""},
    {"example": """Type: grouping, Name: Threat Intel Report: Emerging Cyber Espionage Group, Context: suspicious-activity, Description: A compilation of threat intelligence related to a newly identified cyber espionage group targeting the energy sector, Object Refs: ["threat-actor--56f3f0db-b5d5-431c-ae56-c18f02caf500", "indicator--f81f319c-f26c-4ec0-b81f-1c4df743f03f", "relationship--57b56a43-b8b0-4cba-9deb-34e3e1faed9e"]"""},
    {"example": """Type: grouping, Name: Malware Analysis: ZeuS Variant, Context: malware-analysis, Description: Detailed analysis of a new ZeuS banking trojan variant, including indicators and observed behaviors, Object Refs: ["malware--162d917e-766f-4611-b5d6-652791454fca", "indicator--e73b3dfd-9b8d-45f0-8456-da3d85ef5db7", "observed-data--b67d30ff-02ac-498a-92f9-32f845f448cf"]"""},
    {"example": """Type: grouping, Name: Phishing Campaign Detection, Context: suspicious-activity, Description: Collection of indicators and observables related to an ongoing phishing campaign targeting healthcare organizations, Object Refs: ["indicator--26ffb872-1dd9-446e-b6f5-d58527e5b5d2", "campaign--83422c77-904c-4dc1-aff5-5c38f3a2c55c", "infrastructure--38c47d93-d984-4fd9-b87b-d69d0841628d"]"""}
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
    output_schema=Grouping,
    llm=ChatOpenAI(temperature=1, model='gpt-4'),
    prompt=prompt_template,
)

def generate_grouping(count: int) -> List[Grouping]:
    """
    Generate synthetic groupings.
    
    Args:
        count (int): Number of groupings to generate.
    
    Returns:
        List[Grouping]: List of generated groupings.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="grouping",
        extra="Create diverse groupings with realistic contexts and object references. Ensure that the object references are plausible STIX object identifiers.",
        runs=count,
    )
    return synthetic_results