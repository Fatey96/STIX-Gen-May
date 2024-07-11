from faker import Faker
from stix2 import ThreatActor, Identity, Malware, Tool,Infrastructure, Indicator, AttackPattern, Campaign, IntrusionSet, Vulnerability, Location, CourseOfAction, MalwareAnalysis, Note, Opinion, ObservedData, Report, Grouping

from StixObjectLang.PhaseOne.Campaign import generate_campaigns
from StixObjectLang.PhaseOne.Identity import generate_identities
from StixObjectLang.PhaseOne.Indicator import generate_indicator
from StixObjectLang.PhaseOne.Infrastructure import generate_infrastructures
from StixObjectLang.PhaseOne.IntrusionSet import generate_intrusion_set
from StixObjectLang.PhaseOne.Location import generate_locations
from StixObjectLang.PhaseOne.Malware import generate_malware
from StixObjectLang.PhaseOne.ThreatActor import generate_threat_actor
from StixObjectLang.PhaseOne.Tool import generate_tool

from StixObjectLang.PhaseTwo.AttackPattern import generate_attack_pattern
from StixObjectLang.PhaseTwo.CourseOfAction import generate_course_of_action
from StixObjectLang.PhaseTwo.Grouping import generate_grouping
from StixObjectLang.PhaseTwo.MalwareAnalysis import generate_malware_analysis
from StixObjectLang.PhaseTwo.Note import generate_notes
from StixObjectLang.PhaseTwo.Opinion import generate_opinions
from StixObjectLang.PhaseTwo.ObservedData import generate_observed_data
from StixObjectLang.PhaseTwo.Report import generate_reports
from StixObjectLang.PhaseTwo.Vulnerability import generate_vulnerabilities

from stix2 import KillChainPhase
from datetime import datetime, timedelta
from dateutil.parser import parse
from typing import List, Optional
import random
import uuid

def convert_to_iso_format(date_str: Optional[str]) -> Optional[str]:
    """
    Convert a date string to ISO format with UTC timezone.
    If the input is None or not a valid date string, return None.
    """
    if date_str is None:
        return None
    try:
        date_obj = datetime.fromisoformat(date_str.rstrip('Z'))
        return date_obj.isoformat() + "Z"
    except ValueError:
        return datetime.utcnow().isoformat() + "Z"

def format_kill_chain_phases(phases):
    if not phases:
        return None
    return [KillChainPhase(kill_chain_name=phase.get('kill_chain_name'), phase_name=phase.get('phase_name')) 
            for phase in phases if 'kill_chain_name' in phase and 'phase_name' in phase]

def format_stix_pattern(pattern: str) -> str:
    pattern = pattern.strip()
    if not pattern.startswith('[') and not pattern.endswith(']'):
        pattern = f"[{pattern}]"
    pattern = pattern.replace('email:message-subject', 'email-message:subject')
    return pattern

def get_random_refs(phase_1_objects, count=2):
    return random.sample([obj.id for obj in phase_1_objects], min(count, len(phase_1_objects)))

def create_attack_patterns(count, phase_1_objects):
    synthetic_results = generate_attack_pattern(count)
    fake_attack_patterns = []
    for item in synthetic_results:
        fake_attack_pattern = AttackPattern(
            type=item.type,
            spec_version="2.1",
            id=f"attack-pattern--{str(uuid.uuid4())}",
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            external_references=get_random_refs(phase_1_objects),
            aliases=item.aliases,
            kill_chain_phases=format_kill_chain_phases(item.kill_chain_phases) if item.kill_chain_phases else None,
        )
        fake_attack_patterns.append(fake_attack_pattern)
    return fake_attack_patterns

def create_campaigns(count):
    synthetic_results = generate_campaigns(count)
    fake_campaigns = []
    for item in synthetic_results:
        try:
            # Remove the 'Z' and '+00:00' from the timestamp string
            first_seen = item.first_seen.rstrip('Z').split('+')[0] if item.first_seen else None
            last_seen = item.last_seen.rstrip('Z').split('+')[0] if item.last_seen else None
            
            fake_campaign = Campaign(
                type=item.type,
                spec_version=item.spec_version,
                id=f"campaign--{str(uuid.uuid4())}",
                created=datetime.utcnow(),
                modified=datetime.utcnow(),
                name=item.name,
                description=item.description,
                aliases=item.aliases,
                first_seen=datetime.fromisoformat(first_seen) if first_seen else None,
                last_seen=datetime.fromisoformat(last_seen) if last_seen else None,
                objective=item.objective
            )
            fake_campaigns.append(fake_campaign)
        except Exception as e:
            print(f"Error creating campaign: {str(e)}")
            print(f"Problematic item: {item}")
    return fake_campaigns

def create_notes(count, phase_1_objects):
    synthetic_results = generate_notes(count)
    fake_notes = []
    for item in synthetic_results:
        fake_note = Note(
            type=item.type,
            spec_version="2.1",
            id="note--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            abstract=item.abstract,
            content=item.content,
            authors=item.authors,
            object_refs=get_random_refs(phase_1_objects)
        )
        fake_notes.append(fake_note)
    return fake_notes

def create_observed_datas(count, phase_1_objects):
    synthetic_results = generate_observed_data(count)
    fake_observed_data = []
    for item in synthetic_results:
        fake_observed_datum = ObservedData(
            type=item.type,
            spec_version="2.1",
            id="observed-data--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            first_observed=datetime.fromisoformat(item.first_observed.rstrip('Z')),
            last_observed=datetime.fromisoformat(item.last_observed.rstrip('Z')),
            number_observed=item.number_observed,
            objects=get_random_refs(phase_1_objects)
        )
        fake_observed_data.append(fake_observed_datum)
    return fake_observed_data

def create_reports(count, phase_1_objects):
    synthetic_results = generate_reports(count)
    fake_reports = []
    for item in synthetic_results:
        fake_report = Report(
            type=item.type,
            spec_version="2.1",
            id="report--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            published=datetime.fromisoformat(item.published.rstrip('Z')),
            object_refs=get_random_refs(phase_1_objects)
        )
        fake_reports.append(fake_report)
    return fake_reports

def create_course_of_actions(count, phase_1_objects):
    synthetic_results = generate_course_of_action(count)
    fake_courses_of_action = []
    for item in synthetic_results:
        fake_course_of_action = CourseOfAction(
            type=item.type,
            spec_version="2.1",
            id="course-of-action--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description
            
        )
        fake_courses_of_action.append(fake_course_of_action)

    return fake_courses_of_action

def create_identities(count):
    synthetic_results = generate_identities(count)
    fake_identities = []
    for item in synthetic_results:
        try:
            fake_identity = Identity(
                type=item.type,
                spec_version=item.spec_version,
                id="identity--" + str(uuid.uuid4()),
                created=datetime.utcnow().isoformat() + "Z",
                modified=datetime.utcnow().isoformat() + "Z",
                name=item.name,
                description=item.description,
                roles=item.roles,
                identity_class=item.identity_class,
                sectors=item.sectors,
                contact_information=item.contact_information
            )
            fake_identities.append(fake_identity)
        except Exception as e:
            print(f"Error creating identity: {str(e)}")
            print(f"Problematic item: {item}")
    return fake_identities

def create_groupings(count, phase_1_objects):
    synthetic_results = generate_grouping(count)
    fake_groupings = []
    for item in synthetic_results:
        fake_grouping = Grouping(
            type=item.type,
            spec_version="2.1",
            id="grouping--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            context=item.context,
            object_refs=get_random_refs(phase_1_objects)
        )
        fake_groupings.append(fake_grouping)
    return fake_groupings

def create_opinions(count, phase_1_objects):
    synthetic_results = generate_opinions(count)
    fake_opinions = []
    for item in synthetic_results:
        fake_opinion = Opinion(
            type=item.type,
            spec_version="2.1",
            id="opinion--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            opinion=item.opinion,
            explanation=item.explanation,
            authors=item.authors,
            object_refs=get_random_refs(phase_1_objects)
        )
        fake_opinions.append(fake_opinion)

def create_indicators(count):
    synthetic_results = generate_indicator(count)
    fake_indicators = []
    for item in synthetic_results:
        try:
            fake_indicator = Indicator(
                type=item.type,
                spec_version=item.spec_version,
                id=f"indicator--{str(uuid.uuid4())}",
                created=datetime.utcnow(),
                modified=datetime.utcnow(),
                name=item.name,
                description=item.description,
                pattern=item.pattern,
                pattern_type=item.pattern_type,
                valid_from=datetime.fromisoformat(item.valid_from.rstrip('Z')),
                valid_until=datetime.fromisoformat(item.valid_until.rstrip('Z')) if item.valid_until else None,
                indicator_types=item.indicator_types
            )
            fake_indicators.append(fake_indicator)
        except Exception as e:
            print(f"Error creating indicator: {str(e)}")
            print(f"Problematic item: {item}")
    return fake_indicators

def create_infrastructures(count):
    synthetic_results = generate_infrastructures(count)
    stix_infrastructures = []
    for item in synthetic_results:
        stix_infrastructure = Infrastructure(
            type=item.type,
            spec_version="2.1",
            id=f"infrastructure--{str(uuid.uuid4())}",
            created=datetime.utcnow(),
            modified=datetime.utcnow(),
            name=item.name,
            description=item.description,
            infrastructure_types=item.infrastructure_types,
            aliases=item.aliases,
            kill_chain_phases=format_kill_chain_phases(item.kill_chain_phases),
            first_seen=convert_to_iso_format(item.first_seen) if item.first_seen else None,
            last_seen=convert_to_iso_format(item.last_seen) if item.last_seen else None
        )
        stix_infrastructures.append(stix_infrastructure)
    return stix_infrastructures

def create_intrusion_sets(count):
    synthetic_results = generate_intrusion_set(count)
    fake_intrusion_sets = []
    for item in synthetic_results:
        try:
            fake_intrusion_set = IntrusionSet(
                type=item.type,
                spec_version="2.1",
                id=f"intrusion-set--{str(uuid.uuid4())}",
                created=datetime.utcnow(),
                modified=datetime.utcnow(),
                name=item.name,
                description=item.description,
                aliases=item.aliases,
                first_seen=convert_to_iso_format(item.first_seen),
                last_seen=convert_to_iso_format(item.last_seen),
                goals=item.goals,
                resource_level=item.resource_level,
                primary_motivation=item.primary_motivation,
                secondary_motivations=item.secondary_motivations
            )
            fake_intrusion_sets.append(fake_intrusion_set)
        except Exception as e:
            print(f"Error creating intrusion set: {str(e)}")
            print(f"Problematic item: {item}")
    return fake_intrusion_sets

def create_locations(count):
    synthetic_results = generate_locations(count)
    fake_locations = []
    for item in synthetic_results:
        fake_location = Location(
            type=item.type,
            spec_version="2.1",
            id="location--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            latitude=item.latitude,
            longitude=item.longitude,
            precision=item.precision,
            region=item.region,
            country=item.country,
            administrative_area=item.administrative_area,
            city=item.city,
            street_address=item.street_address,
            postal_code=item.postal_code
        )
        fake_locations.append(fake_location)

    return fake_locations

def create_malwares(count):
    synthetic_results = generate_malware(count)
    fake_malwares = []
    for item in synthetic_results:
        fake_malware = Malware(
            type=item.type,
            spec_version="2.1",
            id="malware--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            malware_types=item.malware_types,
            is_family=bool(item.is_family),
            aliases=item.aliases,
            first_seen=convert_to_iso_format(item.first_seen),
            last_seen=convert_to_iso_format(item.last_seen),
        )
        fake_malwares.append(fake_malware)
    return fake_malwares

def create_malware_analysis(count, phase_1_objects):
    synthetic_results = generate_malware_analysis(count)
    fake_malware_analyses = []
    for item in synthetic_results:
        fake_malware_analysis = MalwareAnalysis(
            type=item.type,
            spec_version="2.1",
            id="malware-analysis--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            product=item.product,
            version=item.version,
            configuration_version=item.configuration_version,
            modules=item.modules,
            analysis_engine_version=item.analysis_engine_version,
            analysis_definition_version=item.analysis_definition_version,
            submitted=item.submitted,
            analysis_started=item.analysis_started,
            analysis_ended=item.analysis_ended,
            analysis_sco_refs=get_random_refs(phase_1_objects)
        )
        fake_malware_analyses.append(fake_malware_analysis)
    return fake_malware_analyses

def create_threat_actors(count):
    synthetic_results = generate_threat_actor(count)

    # Convert the synthetic results to the format expected by STIX
    fake_threat_actors = []
    for item in synthetic_results:
        fake_threat_actor = ThreatActor(
            type=item.type,
            spec_version="2.1",
            id="threat-actor--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            threat_actor_types=item.threat_actor_types,
            aliases=item.aliases,
            first_seen=convert_to_iso_format(item.first_seen),
            last_seen=convert_to_iso_format(item.last_seen),
            roles=item.roles,
            goals=item.goals,
            sophistication=item.sophistication,
            resource_level=item.resource_level,
            primary_motivation=item.primary_motivation,
            secondary_motivations=item.secondary_motivations,
            personal_motivations=item.personal_motivations
        )
        fake_threat_actors.append(fake_threat_actor)

    return fake_threat_actors

def create_tools(count):
    synthetic_results = generate_tool(count)
    fake_tools = []
    for item in synthetic_results:
        fake_tool = Tool(
            type=item.type,
            spec_version="2.1",
            id="tool--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            tool_version=item.tool_version,
            tool_types=item.tool_types,
            kill_chain_phases=format_kill_chain_phases(item.kill_chain_phases),
            aliases=item.aliases
        )
        fake_tools.append(fake_tool)
    return fake_tools

def create_vulnerabilities(count, phase_1_objects):
    synthetic_results = generate_vulnerabilities(count)
    fake_vulnerabilities = []
    for item in synthetic_results:
        # Extract potential sources from phase 1 objects
        potential_sources = [obj for obj in phase_1_objects if obj.type in ['identity', 'tool', 'malware']]
        source = random.choice(potential_sources) if potential_sources else None
        
        fake_vulnerability = Vulnerability(
            type=item.type,
            spec_version="2.1",
            id="vulnerability--" + str(uuid.uuid4()),
            created=datetime.utcnow().isoformat() + "Z",
            modified=datetime.utcnow().isoformat() + "Z",
            name=item.name,
            description=item.description,
            vulnerability_types=item.vulnerability_types,
            first_seen=parse(item.first_seen).isoformat() + "Z",
            last_seen=parse(item.last_seen).isoformat() + "Z",
            severity=item.severity,
            impact=item.impact,
            likelihood=item.likelihood
        )
        
        if source:
            fake_vulnerability.source_name = source.name
        
        fake_vulnerabilities.append(fake_vulnerability)

    return fake_vulnerabilities
