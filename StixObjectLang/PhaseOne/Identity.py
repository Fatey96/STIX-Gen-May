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

# Define the Identity schema
class Identity(BaseModel):
    type: str = Field(default="identity")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the identity")
    created: str = Field(description="Creation date of the identity entry")
    modified: str = Field(description="Last modification date of the identity entry")
    name: str = Field(description="Name of this Identity")
    description: Optional[str] = Field(default=None, description="More details and context about the Identity")
    roles: Optional[List[str]] = Field(default=None, description="List of roles that this Identity performs")
    identity_class: str = Field(description="Type of entity that this Identity describes")
    sectors: Optional[List[str]] = Field(default=None, description="List of industry sectors that this Identity belongs to")
    contact_information: Optional[str] = Field(default=None, description="Contact information for this Identity")

# Sample data as examples
examples = [
    {"example": """Type: identity, Name: John Smith, Identity Class: individual, Roles: [employee, developer], Sectors: [technology], Contact Information: john.smith@example.com"""},
    {"example": """Type: identity, Name: ACME Widget, Inc., Identity Class: organization, Sectors: [manufacturing, retail], Contact Information: info@acmewidget.com, Description: A leading manufacturer of innovative widgets"""},
    {"example": """Type: identity, Name: Healthcare Professionals Association, Identity Class: group, Roles: [medical professionals], Sectors: [healthcare], Contact Information: contact@healthcareprofessionals.org, Description: An association representing healthcare professionals across various disciplines"""},
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
    output_schema=Identity,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_identities(count: int) -> List[Identity]:
    """
    Generate synthetic identity entries.
    
    Args:
        count (int): Number of identity entries to generate.
    
    Returns:
        List[Identity]: List of generated identity entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="identity",
        extra="Create diverse identities including individuals, organizations, and groups. Ensure a mix of different sectors, roles, and identity classes. Use realistic but fictional names and contact information. Always include ID, Created, and Modified fields. Do not use John Doe or Jane Doe as names.",
        runs=count,
    )
    return synthetic_results