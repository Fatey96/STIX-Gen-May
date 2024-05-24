import os
os.environ["OPENAI_API_KEY"] = "sk-proj-6K2dANij8t7gG7FohQxBT3BlbkFJtkFeltl7Ua7K6JuRF6th"

"""# Imports"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from datetime import datetime
from typing import List, Optional
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator, OPENAI_TEMPLATE
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX

"""# Schema for generating Tool

"""

class Tool(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    name: str
    tool_types: Optional[List[str]]
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    kill_chain_phases: Optional[List[str]] = None
    tool_version: Optional[str] = None

"""# Sample Data as example"""

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

"""# Prompt Template for GPT-4"""

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

"""# Data Generator"""

synthetic_data_generator = create_openai_data_generator(
    output_schema=Tool,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="tool",
    extra="Choose a unique and unconventional name for each tool. Avoid common or typical names. Tool Types can be from this list: denial-of-service, exploitation, information-gathering, network-capture, credential-exploitation, remote-access, vulnerability-scanning, unknown",
    runs=55,
)

len(synthetic_results)

"""# Display Data"""

synthetic_results

"""# Display as a DataFrame"""

import pandas as pd

# Create a list of dictionaries from the objects
synthetic_data = []
for item in synthetic_results:
    synthetic_data.append({
        'type': item.type,
        'name': item.name,
        'tool_types': item.tool_types,
        'description': item.description,
        'aliases': item.aliases,
        'kill_chain_phases': item.kill_chain_phases,
        'tool_version': item.tool_version
    })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('tool_data.csv', index=False)  # index=False prevents adding an extra index column
print("Tool data saved to 'tool_data.csv'")

