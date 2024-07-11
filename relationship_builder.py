from stix2 import Relationship
from typing import List, Dict, Any
import re
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableSequence
import random
import dotenv
import json
import logging

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expanded relationship map (you can further expand this based on STIX guidelines)
relationship_map = {
    "threat-actor": {
        "identity": ["attributed-to", "impersonates", "compromises", "targets"],
        "attack-pattern": ["uses", "creates", "modifies"],
        "malware": ["uses", "creates", "modifies"],
        "tool": ["uses", "creates", "modifies"],
        "vulnerability": ["targets", "discovers", "exploits"],
        "infrastructure": ["uses", "controls", "maintains"],
        "campaign": ["initiates", "leads", "participates-in"],
        "intrusion-set": ["associated-with", "part-of"]
    },
    "identity": {
        "threat-actor": ["targets", "mitigates"],
        "campaign": ["targets", "mitigates"],
        "vulnerability": ["owns", "mitigates"],
        "infrastructure": ["owns", "uses"]
    },
    "malware": {
        "vulnerability": ["exploits", "targets"],
        "tool": ["uses", "drops"],
        "attack-pattern": ["uses", "implements"],
        "campaign": ["used-by", "associated-with"],
        "threat-actor": ["created-by", "used-by"],
        "infrastructure": ["uses", "hosted-on", "communicates-with"]
    },
    "indicator": {
        "campaign": ["indicates", "attributed-to"],
        "malware": ["indicates", "attributed-to"],
        "threat-actor": ["indicates", "attributed-to"],
        "tool": ["indicates", "attributed-to"],
        "intrusion-set": ["indicates", "attributed-to"],
        "attack-pattern": ["indicates"]
    },
    "campaign": {
        "threat-actor": ["attributed-to", "targets"],
        "intrusion-set": ["attributed-to", "uses"],
        "identity": ["targets"],
        "vulnerability": ["targets", "exploits"],
        "tool": ["uses"],
        "malware": ["uses", "delivers"]
    },
    "intrusion-set": {
        "campaign": ["consists-of", "attributed-to"],
        "threat-actor": ["attributed-to", "compromises"],
        "attack-pattern": ["uses"],
        "malware": ["uses"],
        "tool": ["uses"],
        "infrastructure": ["uses", "compromises"]
    },
    "attack-pattern": {
        "malware": ["delivers", "uses"],
        "identity": ["targets"],
        "vulnerability": ["exploits", "targets"],
        "tool": ["uses"]
    },
    "tool": {
        "threat-actor": ["used-by"],
        "malware": ["delivers", "drops"],
        "vulnerability": ["exploits", "targets"]
    },
    "course-of-action": {
        "indicator": ["investigates", "mitigates"],
        "observed-data": ["based-on"],
        "attack-pattern": ["mitigates"],
        "malware": ["remediates", "prevents"],
        "vulnerability": ["remediates", "mitigates"],
        "tool": ["mitigates"]
    },
    "location": {
        "identity": ["located-at"],
        "threat-actor": ["located-at"],
        "campaign": ["originates-from"],
        "malware": ["originates-from"],
        "intrusion-set": ["originates-from"],
        "attack-pattern": ["targets"],
        "tool": ["targets"],
        "infrastructure": ["located-at"]
    },
    "malware-analysis": {
        "malware": ["characterizes", "analysis-of", "static-analysis-of", "dynamic-analysis-of"]
    },
    "vulnerability": {
        "malware": ["targeted-by"],
        "tool": ["targeted-by"],
        "attack-pattern": ["targeted-by"],
        "campaign": ["targeted-by"],
        "intrusion-set": ["targeted-by"],
        "threat-actor": ["targeted-by"]
    },
    "infrastructure": {
        "threat-actor": ["used-by", "compromised-by"],
        "campaign": ["used-by"],
        "intrusion-set": ["used-by"],
        "malware": ["hosts", "communicates-with"],
        "tool": ["hosts"],
        "vulnerability": ["has"]
    },
    "note": {
        "malware": ["related-to"],
        "indicator": ["related-to"],
        "threat-actor": ["related-to"],
        "tool": ["related-to"],
        "intrusion-set": ["related-to"],
        "attack-pattern": ["related-to"]
    },
    "observed-data": {
        "indicator": ["based-on"],
        "threat-actor": ["based-on"],
        "tool": ["based-on"],
        "malware": ["based-on"],
        "attack-pattern": ["based-on"]
    },
    "report": {
        "threat-actor": ["reports-on"],
        "campaign": ["reports-on"],
        "vulnerability": ["reports-on"],
        "malware": ["reports-on"],
        "tool": ["reports-on"],
        "incident": ["reports-on"]
    },
    "grouping": {
        "threat-actor": ["groups"],
        "campaign": ["groups"],
        "vulnerability": ["groups"],
        "malware": ["groups"],
        "tool": ["groups"],
        "incident": ["groups"]
    },
    "opinion": {
        "threat-actor": ["opinion-on"],
        "campaign": ["opinion-on"],
        "vulnerability": ["opinion-on"],
        "malware": ["opinion-on"],
        "tool": ["opinion-on"],
        "incident": ["opinion-on"]
    }
    
    
}

def create_relationship_prompt():
    template = """You are an expert in STIX (Structured Threat Information Expression) relationships and cyber threat intelligence storytelling.

Given the following information:
- Source object: {source_obj}
- Target object: {target_obj}
- Valid relationship types: {valid_relationships}

Task:
1. Analyze the source and target objects, considering their types and attributes.
2. Choose the most appropriate relationship type from the valid relationships list that fits the narrative.
3. If none of the relationships seem appropriate, respond with "NO_RELATIONSHIP".
4. Provide a brief justification for your choice (1-2 sentences).

Provide your answer in the following JSON format:
{{
    "relationship_type": "chosen_relationship_or_NO_RELATIONSHIP",
    "justification": "Your brief justification here"
}}

IMPORTANT: Your response must be valid JSON. Do not include any text before or after the JSON object, and do not wrap it in code blocks.
"""
    return PromptTemplate(
        input_variables=["source_obj", "target_obj", "valid_relationships"],
        template=template
    )

def object_to_dict(obj):
    return {
        "id": obj.id,
        "type": obj.type,
        "name": getattr(obj, 'name', 'Unknown'),
        "description": getattr(obj, 'description', 'No description available')
    }

def clean_llm_response(response):
    # Remove Markdown code block syntax if present
    cleaned = re.sub(r'```json\s*|\s*```', '', response).strip()
    return cleaned

class STIXRelationshipAgent:
    def __init__(self, stix_objects: List[Any], max_relationships_per_object: int = 5):
        self.stix_objects = stix_objects
        self.max_relationships_per_object = max_relationships_per_object
        self.llm = ChatOpenAI(temperature=0.7, model_name='gpt-4o')
        self.relationships = []
        self.story = ""
        self.evaluation = {}

    def generate_relationships(self):
        prompt = create_relationship_prompt()
        chain = prompt | self.llm

        valid_stix_objects = [obj for obj in self.stix_objects if hasattr(obj, 'id') and hasattr(obj, 'type')]

        for source_obj in valid_stix_objects:
            potential_targets = [obj for obj in valid_stix_objects if obj.id != source_obj.id]
            
            for target_obj in potential_targets[:self.max_relationships_per_object]:
                valid_relationships = relationship_map.get(source_obj.type, {}).get(target_obj.type, [])
                if not valid_relationships:
                    continue

                try:
                    result = chain.invoke({
                        "source_obj": json.dumps(object_to_dict(source_obj)),
                        "target_obj": json.dumps(object_to_dict(target_obj)),
                        "valid_relationships": ", ".join(valid_relationships)
                    })
                    
                    logger.info(f"LLM Response for relationship: {result.content}")
                    
                    cleaned_content = clean_llm_response(result.content)
                    response = json.loads(cleaned_content)
                    relationship_type = response.get("relationship_type")
                    justification = response.get("justification")
                    
                    logger.info(f"Relationship Type: {relationship_type}")
                    logger.info(f"Justification: {justification}")
                    
                    if relationship_type and relationship_type.upper() != "NO_RELATIONSHIP":
                        relationship = Relationship(relationship_type=relationship_type, 
                                                    source_ref=source_obj.id, 
                                                    target_ref=target_obj.id, 
                                                    description=justification)
                        self.relationships.append(relationship)
                except Exception as e:
                    logger.error(f"Error generating relationship with LLM: {e}")

    def analyze_story(self):
        story_prompt = PromptTemplate(
            input_variables=["relationships"],
            template="""Given the following STIX relationships:

{relationships}

Provide a brief analysis of the cyber threat story these relationships tell. 
Focus on key actors, their motivations, and the progression of their activities.
Limit your response to 3-5 paragraphs."""
        )
        
        story_chain = story_prompt | self.llm
        
        relationship_descriptions = [
            f"{r.source_ref} {r.relationship_type} {r.target_ref}: {getattr(r, 'description', 'No description')}"
            for r in self.relationships
        ]
        
        try:
            result = story_chain.invoke({"relationships": "\n".join(relationship_descriptions)})
            self.story = result.content
            logger.info(f"Generated Story: {self.story}")
        except Exception as e:
            logger.error(f"Error generating story: {e}")
            self.story = "Unable to generate story due to an error."

    def evaluate_performance(self):
        evaluation_prompt = PromptTemplate(
            input_variables=["relationships", "story"],
            template="""You are an expert in evaluating STIX relationships and cyber threat intelligence narratives.

Given the following generated relationships and story:

Relationships:
{relationships}

Story:
{story}

Task:
1. Evaluate the coherence and plausibility of the relationships.
2. Assess how well the story captures the essence of the relationships.
3. Consider the diversity of relationship types and object types involved.
4. Provide a score from 0 to 10, where 0 is completely implausible or incoherent, and 10 is highly coherent and compelling.
5. Provide a brief justification for your score (2-3 sentences).

Provide your answer in the following JSON format:
{{
    "score": your_score_here,
    "justification": "Your justification here"
}}

IMPORTANT: Your response must be valid JSON. Do not include any text before or after the JSON object, and do not wrap it in code blocks.
"""
        )
        
        evaluation_chain = evaluation_prompt | self.llm
        
        relationship_descriptions = [
            f"{r.source_ref} {r.relationship_type} {r.target_ref}: {getattr(r, 'description', 'No description')}"
            for r in self.relationships
        ]
        
        try:
            result = evaluation_chain.invoke({
                "relationships": "\n".join(relationship_descriptions),
                "story": self.story
            })
            
            cleaned_content = clean_llm_response(result.content)
            self.evaluation = json.loads(cleaned_content)
            logger.info(f"Self-Evaluation: {self.evaluation}")
        except Exception as e:
            logger.error(f"Error during self-evaluation: {e}")
            self.evaluation = {"score": 0, "justification": "Unable to evaluate due to an error."}

    def run(self):
        logger.info("Starting STIX Relationship Agent")
        self.generate_relationships()
        self.analyze_story()
        self.evaluate_performance()
        logger.info("STIX Relationship Agent completed its run")
        return {
            "relationships": self.relationships,
            "story": self.story,
            "evaluation": self.evaluation
        }

def create_stix_story(stix_objects: List):
    agent = STIXRelationshipAgent(stix_objects)
    return agent.run()