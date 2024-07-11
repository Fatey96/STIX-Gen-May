import os
from typing import List, Optional, Dict
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

class ObservedData(BaseModel):
    type: str = Field(default="observed-data")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the observed data")
    created: str = Field(description="Creation date of the observed data entry")
    modified: str = Field(description="Last modification date of the observed data entry")
    first_observed: str = Field(description="The beginning of the time window during which the data was seen")
    last_observed: str = Field(description="The end of the time window during which the data was seen")
    number_observed: int = Field(description="The number of times the data was observed", ge=1, le=999999999)
    object_refs: List[str] = Field(description="A list of SCOs and SROs representing the observation")

    # Optional Common Properties
    created_by_ref: Optional[str] = Field(default=None, description="ID of the creator of this observed data")
    revoked: Optional[bool] = Field(default=None, description="Indicates whether the observed data has been revoked")
    labels: Optional[List[str]] = Field(default=None, description="List of terms describing this observed data")
    confidence: Optional[int] = Field(default=None, description="Confidence in the correctness of this observed data")
    lang: Optional[str] = Field(default=None, description="Language of the text content in this observed data")
    external_references: Optional[List[dict]] = Field(default=None, description="List of external references for this observed data")
    object_marking_refs: Optional[List[str]] = Field(default=None, description="List of marking definitions to be applied to this observed data")
    granular_markings: Optional[List[dict]] = Field(default=None, description="List of granular markings applied to this observed data")

examples = [
    {"example": """Type: observed-data, First Observed: 2023-05-01T08:00:00Z, Last Observed: 2023-05-01T09:00:00Z, Number Observed: 100, Object Refs: ["ipv4-address--ff26c055-6336-5bc5-b98d-13d6226742dd", "network-traffic--2568d22a-8998-58eb-99ec-3c8ca74f527d"], Created By Ref: "identity--f431f809-377b-45e0-aa1c-6a4751cae5ff" """},
    {"example": """Type: observed-data, First Observed: 2023-05-02T14:30:00Z, Last Observed: 2023-05-02T15:30:00Z, Number Observed: 50, Object Refs: ["file--44014d21-35b9-5a38-98bd-cb6c5b30db51", "process--7966d0c8-505a-509b-941e-94a57b4b4bb0"], Labels: ["malware", "ransomware"]"""},
    {"example": """Type: observed-data, First Observed: 2023-05-03T10:00:00Z, Last Observed: 2023-05-03T11:00:00Z, Number Observed: 25, Object Refs: ["email-message--72b7698f-10c2-565a-a2a6-b4996a2f2265", "email-addr--89f52ea8-d6ef-51e9-8fce-6a29236436ed"], Confidence: 85"""},
    {"example": """Type: observed-data, First Observed: 2023-05-04T18:00:00Z, Last Observed: 2023-05-04T19:00:00Z, Number Observed: 75, Object Refs: ["domain-name--ecb120bf-2694-4902-a737-62b74539a41b", "url--c1477287-23ac-5971-a010-5c287877fa60"], External References: [{"source_name": "security blog", "url": "https://example.com/blog/suspicious-domains"}]"""},
    {"example": """Type: observed-data, First Observed: 2023-05-05T12:00:00Z, Last Observed: 2023-05-05T13:00:00Z, Number Observed: 30, Object Refs: ["windows-registry-key--2ba37ae7-2745-5082-9dfd-9486dad41016"], Object Marking Refs: ["marking-definition--fa42a846-8d90-4e51-bc29-71d5b4802168"]"""}
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
    output_schema=ObservedData,
    llm=ChatOpenAI(temperature=1, model='gpt-4'),
    prompt=prompt_template,
)

def generate_observed_data(count: int) -> List[ObservedData]:
    """
    Generate synthetic observed data.
    
    Args:
        count (int): Number of observed data entries to generate.
    
    Returns:
        List[ObservedData]: List of generated observed data entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="observed-data",
        extra="Create diverse observed data entries with realistic timestamps, number of observations, and object references. Ensure that the object references are plausible STIX Cyber-observable Objects (SCOs) identifiers. The observed data should represent various types of cyber security related entities such as IP addresses, files, network traffic, etc.",
        runs=count,
    )
    return synthetic_results

