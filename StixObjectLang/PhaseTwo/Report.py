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

class ReportTypeEnum(str, Enum):
    threat_report = "threat-report"
    attack_pattern = "attack-pattern"
    campaign = "campaign"
    incident = "incident"
    malware = "malware"
    threat_actor = "threat-actor"
    tool = "tool"
    vulnerability = "vulnerability"
    situation_report = "situation-report"

class Report(BaseModel):
    type: str = Field(default="report")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the report")
    created: str = Field(description="Creation date of the report")
    modified: str = Field(description="Last modification date of the report")
    name: str = Field(description="A name used to identify the Report")
    description: Optional[str] = Field(default=None, description="A description that provides more details and context about the Report")
    report_types: List[ReportTypeEnum] = Field(description="The primary subject(s) of this report")
    published: str = Field(description="The date that this Report object was officially published")
    object_refs: List[str] = Field(description="Specifies the STIX Objects that are referred to by this Report")

    # Optional Common Properties
    created_by_ref: Optional[str] = Field(default=None, description="ID of the creator of this report")
    revoked: Optional[bool] = Field(default=None, description="Indicates whether the report has been revoked")
    labels: Optional[List[str]] = Field(default=None, description="List of terms describing this report")
    confidence: Optional[int] = Field(default=None, description="Confidence in the correctness of this report")
    lang: Optional[str] = Field(default=None, description="Language of the text content in this report")
    external_references: Optional[List[dict]] = Field(default=None, description="List of external references for this report")
    object_marking_refs: Optional[List[str]] = Field(default=None, description="List of marking definitions to be applied to this report")
    granular_markings: Optional[List[dict]] = Field(default=None, description="List of granular markings applied to this report")

examples = [
    {"example": """Type: report, Name: The Black Vine Cyberespionage Group, Description: A comprehensive analysis of the Black Vine threat actor group, their TTPs, and recent campaigns, Report Types: ["threat-actor", "campaign"], Published: 2023-05-15T10:00:00Z, Object Refs: ["threat-actor--56f3f0db-b5d5-431c-ae56-c18f02caf500", "campaign--83422c77-904c-4dc1-aff5-5c38f3a2c55c", "malware--c7d1e135-8b34-549a-bb47-302f5cf998ed"]"""},
    {"example": """Type: report, Name: Analysis of CVE-2023-1234 Exploitation in the Wild, Description: Detailed report on the exploitation of a critical vulnerability (CVE-2023-1234) by multiple threat actors, Report Types: ["vulnerability", "threat-report"], Published: 2023-06-01T14:30:00Z, Object Refs: ["vulnerability--f81f319c-f26c-4ec0-b81f-1c4df743f03f", "attack-pattern--7e33a43e-e34b-40ec-89da-36c9bb2cacd5", "indicator--26ffb872-1dd9-446e-b6f5-d58527e5b5d2"]"""},
    {"example": """Type: report, Name: Q2 2023 Cybersecurity Landscape Overview, Description: A comprehensive review of major cyber incidents, emerging threats, and trend analysis for Q2 2023, Report Types: ["situation-report"], Published: 2023-07-05T09:00:00Z, Object Refs: ["incident--26ffb872-1dd9-446e-b6f5-d58527e5b5d2", "indicator--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f", "observed-data--b67d30ff-02ac-498a-92f9-32f845f448cf"]"""},
    {"example": """Type: report, Name: Emerging Ransomware Variant: CryptoLock Analysis, Description: Technical deep-dive into the CryptoLock ransomware, including its infection chain, encryption methods, and mitigation strategies, Report Types: ["malware", "attack-pattern"], Published: 2023-05-20T16:45:00Z, Object Refs: ["malware--c7d1e135-8b34-549a-bb47-302f5cf998ed", "attack-pattern--7e33a43e-e34b-40ec-89da-36c9bb2cacd5", "course-of-action--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f"]"""},
    {"example": """Type: report, Name: APT42: Tactics, Techniques, and Procedures, Description: Comprehensive analysis of APT42, including their typical targets, preferred malware, and recent campaigns, Report Types: ["threat-actor", "campaign", "attack-pattern"], Published: 2023-06-10T11:15:00Z, Object Refs: ["threat-actor--56f3f0db-b5d5-431c-ae56-c18f02caf500", "malware--31b940d4-6f7f-459a-80ea-9c1f17b5891b", "campaign--83422c77-904c-4dc1-aff5-5c38f3a2c55c"]"""}
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
    output_schema=Report,
    llm=ChatOpenAI(temperature=1, model='gpt-4'),
    prompt=prompt_template,
)

def generate_reports(count: int) -> List[Report]:
    """
    Generate synthetic reports.
    
    Args:
        count (int): Number of reports to generate.
    
    Returns:
        List[Report]: List of generated reports.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="report",
        extra="Create diverse reports covering various cybersecurity topics. Include realistic names, descriptions, report types, and object references. Ensure that the object references are plausible STIX object identifiers. The reports should cover a range of subjects such as threat actors, campaigns, vulnerabilities, malware analysis, and situational overviews.",
        runs=count,
    )
    return synthetic_results