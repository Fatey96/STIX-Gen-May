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

# Define the Campaign schema
class Campaign(BaseModel):
    type: str = Field(default="campaign")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the campaign")
    created: str = Field(description="Creation date of the campaign entry")
    modified: str = Field(description="Last modification date of the campaign entry")
    name: str = Field(description="Name used to identify the Campaign")
    description: Optional[str] = Field(default=None, description="More details and context about the Campaign")
    aliases: Optional[List[str]] = Field(default=None, description="Alternative names for the Campaign")
    first_seen: Optional[str] = Field(default=None, description="Time that this Campaign was first seen")
    last_seen: Optional[str] = Field(default=None, description="Time that this Campaign was last seen")
    objective: Optional[str] = Field(default=None, description="Campaign's primary goal or desired outcome")
    
# Sample data as examples
examples = [
    {"example": """Type: campaign, Name: Green Group Attacks Against Finance, Description: Campaign by Green Group against a series of targets in the financial services sector., First Seen: 2016-01-01T00:00:00Z, Last Seen: 2016-06-30T00:00:00Z, Objective: Steal financial data and disrupt operations, Aliases: [Operation Money Grab, Finance Sector Assault]"""},
    {"example": """Type: campaign, Name: Operation Clipboard, Description: A series of spear-phishing attacks targeting government agencies to steal classified information., First Seen: 2018-03-15T00:00:00Z, Last Seen: 2019-12-31T00:00:00Z, Objective: Espionage and data exfiltration, Aliases: [ClipperCrew Campaign, Govt-Spear]"""},
    {"example": """Type: campaign, Name: Ransomware Wave Alpha, Description: Widespread ransomware attacks against healthcare institutions using a new strain of malware., First Seen: 2020-09-01T00:00:00Z, Last Seen: 2021-03-31T00:00:00Z, Objective: Extort money from healthcare providers, Aliases: [MediLock Campaign, Healthcare Ransom Spree]"""},
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
    output_schema=Campaign,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_campaigns(count: int) -> List[Campaign]:
    """
    Generate synthetic campaign entries.
    
    Args:
        count (int): Number of campaign entries to generate.
    
    Returns:
        List[Campaign]: List of generated campaign entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="campaign",
        extra="Create unique and diverse campaigns with realistic objectives, timeframes, and aliases. Ensure a mix of different sectors targeted and varied campaign durations.",
        runs=count,
    )
    return synthetic_results