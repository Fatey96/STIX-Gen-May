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

# Define the AttackPattern schema
class AttackPattern(BaseModel):
    type: str = Field(default="attack-pattern")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the attack pattern")
    created: str = Field(description="Creation date of the attack pattern entry")
    modified: str = Field(description="Last modification date of the attack pattern entry")
    name: str = Field(description="Name of the attack pattern")
    description: Optional[str] = Field(default=None, description="Description of the attack pattern")
    external_references: Optional[List[dict]] = Field(default=None, description="External references for the attack pattern")
    aliases: Optional[List[str]] = Field(default=None, description="Alternative names for the attack pattern")
    kill_chain_phases: Optional[List[dict]] = Field(default=None, description="Kill chain phases associated with the attack pattern")


"""# Sample Data as example"""

examples = [
    {"example": """Type: attack-pattern, spec_version: 2.1, id: attack-pattern--7e33a43e-e34b-40ec-89da-36c9bb2cacd5, created: 2016-05-12T08:17:27.000Z, modified: 2016-05-12T08:17:27.000Z, name: Spear Phishing as Practiced by Adversary X, description: A particular form of spear phishing where the attacker claims that the target had won a contest, including personal details, to get them to click on a link., external_references: [{ source_name: capec, external_id: CAPEC-163}]"""},
    {"example": """Type: attack-pattern, spec_version: 2.1, id: attack-pattern--19da6e1c-71ab-4c2f-886d-d620d09d3b5a, created: 2016-08-08T15:50:10.983Z, modified: 2017-01-30T21:15:04.127Z, name: Content Spoofing, external_references:[{ source_name: capec, url: https://capec.mitre.org/data/definitions/148.html, external_id: CAPEC-148}]"""},
    {"example": """Type: attack-pattern, spec_version: 2.1, id: attack-pattern--f6050ea6-a9a3-4524-93ed-c27858d6cb3c, created: 2016-08-08T15:50:10.983Z, modified: 2017-01-30T21:15:04.127Z, name: HTTP Flood, external_references: [{ source_name: capec, url: "https://capec.mitre.org/data/definitions/488.html, external_id: CAPEC-488}]"""},
    {"example": """Type: attack-pattern, spec-version: 2.1, id: attack-pattern--8ac90ff3-ecf8-4835-95b8-6aea6a623df5, created: 2015-05-07T14:22:14.760Z, modified: 2015-05-07T14:22:14.760Z, name: Phishing, description: Spear phishing used as a delivery mechanism for malware., kill_chain_phases: [{ kill_chain_name: mandiant-attack-lifecycle-model, phase_name: initial-compromise}], external_references: [{ source_name: capec, description: phishing, url: https://capec.mitre.org/data/definitions/98.html, external_id: CAPEC-98}]"""},
    {"example": """Type: attack-pattern, spec-version: 2.1, id: attack-pattern--3098c57b-d623-4c11-92f4-5905da66658b, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: Initial Compromise, description: As with most other APT groups, spear phishing is APT1’s most commonly used technique. The spear phishing emails contain either a malicious attachment or a hyperlink to a malicious file. The subject line and the text in the email body are usually relevant to the recipient. APT1 also creates webmail accounts using real peoples’ names — names that are familiar to the recipient, such as a colleague, a company executive, an IT department employee, or company counsel. The files they use contain malicious executables that install a custom APT1 backdoor that we call WEBC2-TABLE., external_references: [{ source_name: capec, description: spear phishing, external_id: CAPEC-163}], kill_chain_phases:[{kill_chain_name: mandiant-attack-lifecycle-model, phase_name: initial-compromise}]"""},
    {"example": """Type: attack-pattern, spec-version: 2.1, id: attack-pattern--1e2c4237-d469-4144-9c0b-9e5c0c513c49, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: Establishing a Foothold, description: APT1 establishes a foothold once email recipients open a malicious file and a backdoor is subsequently installed.  In almost every case, APT backdoors initiate outbound connections to the intruder’s 'command and control' (C2) server. While APT1 intruders occasionally use publicly available backdoors such as Poison Ivy and Gh0st RAT, the vast majority of the time they use what appear to be their own custom backdoors. APT1’s backdoors are in two categories: 'Beachhead Backdoors' and 'Standard Backdoors.' Beachhead Backdoors offer the attacker a toe-hold to perform simple tasks like retrieve files, gather basic system information and trigger the execution of other more significant capabilities such as a standard backdoor. APT1’s beachhead backdoors are usually what we call WEBC2 backdoors. WEBC2 backdoors are probably the most well-known kind of APT1 backdoor, and are the reason why some security companies refer to APT1 as the Comment Crew. A WEBC2 backdoor is designed to retrieve a webpage from a C2 server. It expects the webpage to contain special HTML tags; the backdoor will attempt to interpret the data between the tags as commands. WEBC2 backdoors are often packaged with spear phishing emails. Once installed, APT1 intruders have the option to tell victim systems to download and execute additional malicious software of their choice. The standard, non-WEBC2 APT1 backdoor typically communicates using the HTTP protocol (to blend in with legitimate web traffic) or a custom protocol that the malware authors designed themselves. The BISCUIT backdoor (so named for the command “bdkzt”) is an illustrative example of the range of commands that APT1 has built into its “standard” backdoors. APT1 has used and steadily modified BISCUIT since as early as 2007 and continues to use it presently. Some APT backdoors attempt to mimic legitimate Internet traffic other than the HTTP protocol. When network defenders see the communications between these backdoors and their C2 servers, they might easily dismiss them as legitimate network traffic. Additionally, many of APT1’s backdoors use SSL encryption so that communications are hidden in an encrypted SSL tunnel., kill_chain_phases: [{kill_chain_name: mandiant-attack-lifecycle-model,phase_name: establish-foothold}]"""},
    {"example": """Type: attack-pattern, spec-version: 2.1, id: attack-pattern--e13f3e6d-4f9c-4265-b1cf-f997a1bf782, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: Privilege Escalation, description: Escalating privileges involves acquiring items (most often usernames and passwords) that will allow access to more resources within the network. APT1 predominantly uses publicly available tools to dump password hashes from victim systems in order to obtain legitimate user credentials., kill_chain_phases: [{ kill_chain_name: mandiant-attack-lifecycle-model, phase_name: escalate-privileges}]"""},
    {"example": """Type: attack-pattern, spec-version: 2.1, id: attack-pattern--5728f45b-2eca-4942-a7f6-bc4267c1ab8d, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: Internal Reconnaisance, description: In the Internal Reconnaissance stage, the intruder collects information about the victim environment. Like most APT (and non-APT) intruders, APT1 primarily uses built-in operating system commands to explore a compromised system and its networked environment. Although they usually simply type these commands into a command shell, sometimes intruders may use batch scripts to speed up the process., kill_chain_phases: [{ kill_chain_name: mandiant-attack-lifecycle-model, phase_name: internal-recon}]"""},
    {"example": """Type: attack-pattern, spec-version: 2.1, id: attack-pattern--0bea2358-c244-4905-a664-a5cdce7bb767, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: Lateral Movement, description: Once an APT intruder has a foothold inside the network and a set of legitimate credentials, it is simple for the intruder to move around the network undetected. They can connect to shared resources on other systems. They can execute commands on other systems using the publicly available 'psexec' tool from Microsoft Sysinternals or the built-in Windows Task Scheduler ('at.exe')., kill_chain_phases: [{kill_chain_name: mandiant-attack-lifecycle-model,phase_name: move-laterally}]"""}

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
    output_schema=AttackPattern,
    llm=ChatOpenAI(temperature=1, model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

def generate_attack_pattern(count: int) -> List[AttackPattern]:
    """
    Generate synthetic attack pattern entries.
    
    Args:
        count (int): Number of attack pattern entries to generate.
    
    Returns:
        List[AttackPattern]: List of generated attack pattern entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="attack_pattern",
        extra="Create unique and realistic attack pattern names and characteristics. Ensure diversity in techniques, targets, and associated kill chain phases.",
        runs=count,
    )
    return synthetic_results
