# Expose nodes via modular agents package to maintain backwards compatibility
from .agents import (
    router_node, 
    research_node, 
    orchestrator_node, 
    worker_node, 
    fanout, 
    merge_content, 
    decide_images, 
    generate_and_place_images,
    fact_checker_node,
    revision_node,
    campaign_generator_node, 
    evaluator_node
)
from .agents.utils import _safe_slug

__all__ = [
    "router_node",
    "research_node",
    "orchestrator_node",
    "fanout",
    "worker_node",
    "merge_content",
    "decide_images",
    "generate_and_place_images",
    "fact_checker_node",
    "revision_node",
    "evaluator_node",
    "campaign_generator_node",
    "_safe_slug"
]