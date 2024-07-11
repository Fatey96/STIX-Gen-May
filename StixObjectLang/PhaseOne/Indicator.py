import os
from typing import List, Optional
import dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_community.chat_models import ChatOpenAI

dotenv.load_dotenv()

# Define the Indicator schema
class Indicator(BaseModel):
    type: str = Field(default="indicator")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the indicator")
    created: str = Field(description="Creation date of the indicator entry")
    modified: str = Field(description="Last modification date of the indicator entry")
    name: Optional[str] = Field(default=None, description="Name of the indicator")
    description: Optional[str] = Field(default=None, description="Description of the indicator")
    pattern: List[str] = Field(description="Pattern for detecting the indicator")
    pattern_type: str = Field(description="Type of pattern (e.g., 'stix')")
    pattern_version: Optional[str] = Field(default=None, description="Version of the pattern")
    valid_from: str = Field(description="Start of validity period for the indicator")
    valid_until: Optional[str] = Field(default=None, description="End of validity period for the indicator")
    indicator_types: List[str] = Field(default=None, description="Types of the indicator")
    kill_chain_phases: Optional[List[dict]] = Field(default=None, description="Kill chain phases associated with the indicator")


examples = [
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f, created: 2016-04-06T20:03:48.000Z, modified: 2016-04-06T20:03:48.000Z, name: Poison Ivy malware, description: This file is part of Poision Ivy, pattern: file:hashes.'SHA-256' = '4bac27393bdd9777ce02453256c5577cd02275510b2227f473d03f533924f877', pattern_type: stix, valid_from: 2016-01-01T00:00:00Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--e8094b09-7df4-4b13-b207-1e27af3c4bde, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: ipv4-addr:value = '219.76.208.163', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--329ae6e9-25bd-49e8-89d1-aae4ca52e4a7, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'www.webserver.dynssl.com' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '75.126.95.138' OR ipv4-addr:value = '219.90.112.197' OR ipv4-addr:value = '202.65.222.45', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id:indicator--54e1e351-fec0-41a4-b62c-d7f86101e241, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'www.webserver.freetcp.com' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '202.65.220.64' OR ipv4-addr:value = '75.126.95.138' OR ipv4-addr:value = '219.90.112.197' OR ipv4-addr:value = '202.65.222.45', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--2e59f00b-0986-437e-9ebd-e0d61900d688, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'www.webserver.fartit.com' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '202.65.220.64' OR ipv4-addr:value = '75.126.95.138', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--8da68996-f175-4ae0-bd74-aad4913873b8, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'microsofta.byinter.net' OR domain-name:value = 'microsoftb.byinter.net' OR domain-name:value = 'microsoftc.byinter.net' OR domain-name:value = 'microsofte.byinter.net' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '202.65.220.64' OR ipv4-addr:value = '75.126.95.138' OR ipv4-addr:value = '219.90.112.197' OR ipv4-addr:value = '202.65.222.45' OR ipv4-addr:value = '98.126.148.114', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--4e11b23f-732b-418e-b786-4dbf65459d50, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z", pattern: domain-name:value = 'nkr.iphone.qpoe.com' OR ipv4-addr:value = '180.210.206.96' OR ipv4-addr:value = '101.78.151.179', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""}
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
    output_schema=Indicator,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_indicator(count: int) -> List[Indicator]:
    """
    Generate synthetic indicator entries.
    
    Args:
        count (int): Number of indicator entries to generate.
    
    Returns:
        List[Indicator]: List of generated indicator entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="indicator",
        extra="Create diverse and realistic indicators with unique patterns, types, and associated kill chain phases. Ensure a mix of malicious, anomalous, and benign indicators.",
        runs=count,
    )
    return synthetic_results