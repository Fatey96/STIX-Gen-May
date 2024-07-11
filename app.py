from flask import Flask, render_template, request, jsonify
from stix_object_builder import (
    create_threat_actors, create_identities, create_malwares,
    create_indicators, create_tools,
    create_intrusion_sets, create_course_of_actions, create_locations,
    create_malware_analysis, create_attack_patterns, create_campaigns, create_vulnerabilities
)
from relationship_builder import create_stix_story
from stix_bundler import create_bundle
from langchain_community.chat_models import ChatOpenAI
import dotenv
import os
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__, static_folder='static', template_folder='templates')

special_cases = {
    'identity': 'create_identities',
    'vulnerability': 'create_vulnerabilities',
    'malware-analysis': 'create_malware_analysis'
}

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select-objects', methods=['GET'])
def select_objects():
    return render_template('select_objects.html')

def create_objects(object_type, count):
    if object_type in special_cases:
        function_name = special_cases[object_type]
    else:
        function_name = f"create_{object_type.replace('-', '_')}s"
    
    creation_function = globals()[function_name]
    objects = creation_function(count)
    logger.debug(f"Created {len(objects)} {object_type} objects")
    return objects

@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    try:
        data = request.form
        logger.debug(f"Received form data: {data}")

        stix_objects = []
        object_types = ['threat-actor', 'identity', 'malware', 'indicator', 'attack-pattern', 'tool', 'campaign', 'intrusion-set', 'vulnerability', 'course-of-action', 'location', 'malware-analysis']
        
        # Use ThreadPoolExecutor for parallel object generation
        with ThreadPoolExecutor(max_workers=min(len(object_types), os.cpu_count() or 1)) as executor:
            future_to_object_type = {
                executor.submit(create_objects, object_type, int(data.get(f'{object_type}-count', 0))): object_type
                for object_type in object_types
                if int(data.get(f'{object_type}-count', 0)) > 0
            }

            for future in as_completed(future_to_object_type):
                object_type = future_to_object_type[future]
                try:
                    objects = future.result()
                    stix_objects.extend(objects)
                except Exception as exc:
                    logger.error(f'{object_type} generated an exception: {exc}')

        logger.info(f"Total STIX objects created: {len(stix_objects)}")

        # Use the STIXRelationshipAgent to generate relationships, story, and evaluation
        agent_result = create_stix_story(stix_objects)

        # Create the STIX bundle
        stix_bundle = create_bundle(stix_objects, agent_result['relationships'])
        logger.debug("STIX bundle created successfully")
        
        return jsonify({
            "stix_bundle": stix_bundle,
            "relationships": [{"source": r.source_ref, "type": r.relationship_type, "target": r.target_ref, "description": r.description} for r in agent_result['relationships']],
            "story": agent_result['story'],
            "evaluation": agent_result['evaluation']
        })
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
