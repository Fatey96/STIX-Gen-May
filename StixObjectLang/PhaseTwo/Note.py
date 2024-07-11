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

class Note(BaseModel):
    type: str = Field(default="note")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the note")
    created: str = Field(description="Creation date of the note entry")
    modified: str = Field(description="Last modification date of the note entry")
    abstract: Optional[str] = Field(default=None, description="A brief summary of the note content")
    content: str = Field(description="The content of the note")
    authors: Optional[List[str]] = Field(default=None, description="The name of the author(s) of this note")
    object_refs: List[str] = Field(description="The STIX Objects that the note is being applied to")

examples = [
    {"example": """Type: note, Abstract: APT29 Campaign Update, Content: Recent analysis shows APT29 has shifted tactics, now using spear-phishing emails with malicious attachments instead of their previous watering hole attacks. This change in approach suggests they're adapting to improved defensive measures., Authors: ["Jane Smith"], Object Refs: ["campaign--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f", "intrusion-set--4e78f46f-a023-4e5f-bc24-71b3ca22ec29"]"""},
    {"example": """Type: note, Abstract: Ransomware Incident Analysis, Content: During the investigation of the recent ransomware attack, we discovered that the initial point of entry was a vulnerable RDP server. The attackers then used lateral movement techniques to spread through the network before deploying the ransomware. This highlights the critical importance of properly securing remote access points., Authors: ["John Doe", "Alice Johnson"], Object Refs: ["malware--31b940d4-6f7f-459a-80ea-9c1f17b5891b", "indicator--a932fcc6-e032-476c-826f-cb970a5a1ade"]"""},
    {"example": """Type: note, Abstract: Potential False Positive in IDS Alert, Content: After investigating the IDS alert triggered by traffic to IP 203.0.113.100, we've concluded it's likely a false positive. The IP belongs to a legitimate CDN used by several of our vendors. Recommend whitelisting this IP to reduce alert noise., Authors: ["Bob Wilson"], Object Refs: ["indicator--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f"]"""},
    {"example": """Type: note, Abstract: Zero-Day Vulnerability Exploitation Attempt, Content: We've observed attempts to exploit a previously unknown vulnerability in our web application framework. The attacks seem to be coming from a known threat actor group. We've implemented a temporary mitigation and are working on a permanent fix. All web-facing systems should be closely monitored for unusual activity., Authors: ["Eva Chen", "Michael Brown"], Object Refs: ["vulnerability--f81f319c-f26c-4ec0-b81f-1c4df743f03f", "threat-actor--56f3f0db-b5d5-431c-ae56-c18f02caf500"]"""},
    {"example": """Type: note, Abstract: Phishing Campaign Target Expansion, Content: The ongoing phishing campaign targeting healthcare organizations has expanded its scope. We're now seeing similar tactics being used against financial institutions. The phishing emails are using COVID-19 themed lures to trick users into clicking malicious links., Authors: ["Sarah Lee"], Object Refs: ["campaign--83422c77-904c-4dc1-aff5-5c38f3a2c55c", "indicator--26ffb872-1dd9-446e-b6f5-d58527e5b5d2"]"""}
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
    output_schema=Note,
    llm=ChatOpenAI(temperature=1, model='gpt-4'),
    prompt=prompt_template,
)

def generate_notes(count: int) -> List[Note]:
    """
    Generate synthetic notes.
    
    Args:
        count (int): Number of notes to generate.
    
    Returns:
        List[Note]: List of generated notes.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="note",
        extra="Create diverse notes with realistic content, abstracts, and object references. Ensure that the object references are plausible STIX object identifiers. The notes should provide valuable context or analysis related to various cybersecurity scenarios.",
        runs=count,
    )
    return synthetic_results