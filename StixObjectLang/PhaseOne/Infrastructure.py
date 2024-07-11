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

# Define the Infrastructure schema
class Infrastructure(BaseModel):
    type: str = Field(default="infrastructure")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the infrastructure")
    created: str = Field(description="Creation date of the infrastructure entry")
    modified: str = Field(description="Last modification date of the infrastructure entry")
    name: str = Field(description="Name or characterizing text used to identify the Infrastructure")
    description: Optional[str] = Field(default=None, description="More details and context about the Infrastructure")
    infrastructure_types: List[str] = Field(description="The type of infrastructure being described")
    aliases: Optional[List[str]] = Field(default=None, description="Alternative names used to identify this Infrastructure")
    kill_chain_phases: Optional[List[dict]] = Field(default=None, description="List of Kill Chain Phases for which this Infrastructure is used")
    first_seen: Optional[str] = Field(default=None, description="Time that this Infrastructure was first seen performing malicious activities")
    last_seen: Optional[str] = Field(default=None, description="Time that this Infrastructure was last seen performing malicious activities")

# Sample data as examples
examples = [
    {"example": """Type: infrastructure, Name: Poison Ivy C2, Infrastructure Types: [command-and-control], Description: Command and control infrastructure for Poison Ivy malware, First Seen: 2016-01-01T00:00:00Z, Last Seen: 2016-12-31T23:59:59Z, Kill Chain Phases: [{kill_chain_name: lockheed-martin-cyber-kill-chain, phase_name: command-and-control}]"""},
    {"example": """Type: infrastructure, Name: Botnet Distribution Network, Infrastructure Types: [botnet, hosting-malware], Description: A network of compromised machines used to distribute malware, Aliases: [MalNet, DistroBot], First Seen: 2018-03-15T00:00:00Z, Last Seen: 2019-06-30T00:00:00Z, Kill Chain Phases: [{kill_chain_name: lockheed-martin-cyber-kill-chain, phase_name: delivery}, {kill_chain_name: lockheed-martin-cyber-kill-chain, phase_name: installation}]"""},
    {"example": """Type: infrastructure, Name: Phishing Campaign Servers, Infrastructure Types: [hosting-target-lists, hosting-malicious-content], Description: Servers used to host phishing pages and store stolen credentials, First Seen: 2020-09-01T00:00:00Z, Last Seen: 2021-03-31T00:00:00Z, Kill Chain Phases: [{kill_chain_name: lockheed-martin-cyber-kill-chain, phase_name: weaponization}, {kill_chain_name: lockheed-martin-cyber-kill-chain, phase_name: delivery}]"""},
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
    output_schema=Infrastructure,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_infrastructures(count: int) -> List[Infrastructure]:
    """
    Generate synthetic infrastructure entries.
    
    Args:
        count (int): Number of infrastructure entries to generate.
    
    Returns:
        List[Infrastructure]: List of generated infrastructure entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="infrastructure",
        extra="Create diverse infrastructure entries with various types, purposes, and associated kill chain phases. Ensure realistic timeframes for first_seen and last_seen dates.",
        runs=count,
    )
    return synthetic_results