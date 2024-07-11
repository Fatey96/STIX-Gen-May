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

# Define the CourseOfAction schema
class CourseOfAction(BaseModel):
    type: str = Field(default="course-of-action")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the course of action")
    created: str = Field(description="Creation date of the course of action entry")
    modified: str = Field(description="Last modification date of the course of action entry")
    name: str = Field(description="Name of the course of action")
    description: str = Field(description="Description of the course of action")
examples = [
{"example": """Type: course-of-action, Name: Implement Multi-Factor Authentication, Description: This action mandates the adoption of multi-factor authentication for all user accounts to significantly decrease the likelihood of unauthorized access. It involves using two or more verification methods for user authentication, adding an extra layer of security beyond just passwords."""},
{"example": """Type: course-of-action, Name: Regular Software Patching, Description: A systematic approach to regularly update and patch operating systems and software applications. This course of action aims to fix vulnerabilities that could be exploited by attackers, keeping the organization's digital assets secure."""},
{"example": """Type: course-of-action, Name: Cybersecurity Awareness Training, Description: An educational initiative aimed at all organizational members to heighten awareness about cyber threats such as phishing, social engineering, and how to prevent them. This course of action promotes a culture of security and vigilance."""},
{"example": """Type: course-of-action, Name: Data Encryption Strategy, Description: A security measure that involves encrypting sensitive data, both at rest and in transit. This course of action ensures that even in the event of a data breach, the information remains secure and inaccessible to unauthorized parties."""},
{"example": """Type: course-of-action, Name: Incident Response Plan, Description: The establishment of a formalized plan to respond to cybersecurity incidents. This includes identifying, containing, eradicating, and recovering from incidents to minimize impact and prevent future occurrences."""},
{"example": """Type: course-of-action, Name: Secure Configuration Standards, Description: The development and enforcement of secure configurations for all IT systems and applications to reduce vulnerabilities and safeguard against attacks. This includes disabling unnecessary services and applying the principle of least privilege."""},
{"example": """Type: course-of-action, Name: Network Segmentation, Description: Dividing the network into distinct zones to improve security and control. This course of action limits the spread of breaches by segregating sensitive areas and applying strict access controls."""},
{"example": """Type: course-of-action, Name: Continuous Monitoring, Description: The implementation of tools and practices for the ongoing surveillance of IT systems to detect and respond to threats in real time. This course of action helps in identifying suspicious activities and mitigating threats before they can cause significant damage."""},
{"example": """Type: course-of-action, Name: Disaster Recovery Planning, Description: The process of creating a comprehensive plan for the continuation or recovery of systems in the event of a catastrophic failure. This course of action includes strategies for data backup, system restoration, and maintaining business operations under adverse conditions."""}
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
    output_schema=CourseOfAction,
    llm=ChatOpenAI(temperature=1, model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

def generate_course_of_action(count: int) -> List[CourseOfAction]:
    """
    Generate synthetic course of action entries.
    
    Args:
        count (int): Number of course of action entries to generate.
    
    Returns:
        List[CourseOfAction]: List of generated course of action entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="course_of_action",
        extra="Create unique and practical courses of action for cybersecurity. Ensure diversity in approaches, targeting various aspects of information security.",
        runs=count,
    )
    return synthetic_results