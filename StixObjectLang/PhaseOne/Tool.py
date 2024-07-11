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

# Define the Tool schema
class Tool(BaseModel):
    type: str = Field(default="tool")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the tool")
    created: str = Field(description="Creation date of the tool entry")
    modified: str = Field(description="Last modification date of the tool entry")
    name: str = Field(description="Name of the tool")
    tool_types: Optional[List[str]] = Field(default=None, description="Types of the tool")
    description: Optional[str] = Field(default=None, description="Description of the tool")
    aliases: Optional[List[str]] = Field(default=None, description="Alternative names for the tool")
    kill_chain_phases: Optional[List[str]] = Field(default=None, description="Kill chain phases associated with the tool")
    tool_version: Optional[str] = Field(default=None, description="Version of the tool")


examples = [
    {"example": """Type: tool, Name: Metasploit Framework, Tool Types: exploitation, Description: A popular open-source penetration testing framework used for developing and executing exploit code against remote targets, Aliases: None, Kill Chain Phases: exploitation, Tool Version: 6.0.1"""},
    {"example": """Type: tool, Name: Wireshark, Tool Types: network-capture, Description: A popular network protocol analyzer used for troubleshooting, analysis, software, and protocol development, Aliases: Ethereal, Kill Chain Phases: reconnaissance, Tool Version: 3.4.5"""},
    {"example": """Type: tool, Name: Nmap, Tool Types: information-gathering, Description: A powerful network scanning tool used for network discovery and security auditing, Aliases: Network Mapper, Kill Chain Phases: reconnaissance, Tool Version: 7.91"""},
    {"example": """Type: tool, Name: Nessus, Tool Types: vulnerability-scanner, Description: A proprietary vulnerability scanner developed by Tenable, Inc. It is used to detect potential vulnerabilities in networks, configurations, and systems, Aliases: None, Kill Chain Phases: reconnaissance, Tool Version: 8.16.0"""},
    {"example": """Type: tool, Name: VNC , Tool Types: remote-access, Description: VNC is a graphical desktop-sharing system that uses the Remote Frame Buffer protocol to remotely control another computer. It allows users to view and interact with the desktop environment of a remote machine over a network connection, Aliases: None, Kill Chain Phases: exploitation, Tool Version: 6.7.2"""},
    {"example": """Type: tool, Name: John the Ripper, Tool Types: credential-exploitation, Description: A fast password cracker for various Unix-based systems, Aliases: None, Kill Chain Phases: exploitation, Tool Version: 1.9.0"""},
    {"example": """Type: tool, Name: Nikto, Tool Types: vulnerability-scanning, Description: A web server scanner that performs comprehensive tests against web servers for multiple items, Kill Chain Phases: reconnaissance, Tool Version: 2.1.6"""},
    {"example": """Type: tool, Name: LOIC (Low Orbit Ion Cannon), Tool Types: denial-of-service, Description: An open-source network stress testing and denial-of-service attack application, written in C#, Aliases: None, Kill Chain Phases: impact, Tool Version: 2.0.0"""},
    {"example": """Type: tool, Name: TeamViewer, Tool Types: remote-access, Description: A proprietary software application for remote control, desktop sharing, online meetings, web conferencing, and file transfer between computers, Aliases: None, Kill Chain Phases: reconnaissance, Tool Version: 15.21.5"""},
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
    output_schema=Tool,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_tool(count: int) -> List[Tool]:
    """
    Generate synthetic tool entries.
    
    Args:
        count (int): Number of tool entries to generate.
    
    Returns:
        List[Tool]: List of generated tool entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="tool",
        extra="Create unique and unconventional tools with diverse capabilities. Tool Types should be from this list: denial-of-service, exploitation, information-gathering, network-capture, credential-exploitation, remote-access, vulnerability-scanning, unknown. Ensure a mix of different tool types and associated kill chain phases.",
        runs=count,
    )
    return synthetic_results

