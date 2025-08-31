"""
Attribute Verbalization Module

Converts predicted attribute vectors into structured text descriptions
and JSON representations for LLM consumption.
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict


class AttributeVerbalizer:
    """Converts attribute predictions to human-readable text and structured JSON."""
    
    def __init__(self, attribute_mapping_path: Optional[str] = None):
        """
        Initialize verbalizer with attribute mappings.
        
        Args:
            attribute_mapping_path: Path to YAML file with attribute mappings
        """
        if attribute_mapping_path and Path(attribute_mapping_path).exists():
            with open(attribute_mapping_path, 'r') as f:
                self.attribute_map = yaml.safe_load(f)
        else:
            # Create default mapping based on CUB-200-2011 attributes
            self.attribute_map = self._create_default_attribute_mapping()
    
    def _create_default_attribute_mapping(self) -> Dict:
        """
        Create default attribute mapping for CUB-200-2011 dataset.
        
        This is a simplified version - in practice you'd want the full
        attribute descriptions from the CUB dataset.
        """
        # This is a subset of actual CUB attributes organized by facets
        # In practice, you'd load the complete mapping from CUB data
        mapping = {
            'bill_shape': {
                'attributes': list(range(1, 16)),  # Bill shape attributes
                'values': {
                    1: 'curved_upper_edge', 2: 'curved_lower_edge', 3: 'straight',
                    4: 'hooked_upper_edge', 5: 'pointed', 6: 'spatulate',
                    7: 'all_purpose', 8: 'conical', 9: 'specialized',
                    10: 'needle', 11: 'hooked', 12: 'spatulate',
                    13: 'curved', 14: 'dagger', 15: 'cone'
                }
            },
            'wing_color': {
                'attributes': list(range(16, 31)),  # Wing color attributes
                'values': {
                    16: 'blue', 17: 'brown', 18: 'buff', 19: 'gray',
                    20: 'green', 21: 'pink', 22: 'purple', 23: 'red',
                    24: 'rufous', 25: 'white', 26: 'yellow', 27: 'black',
                    28: 'orange', 29: 'iridescent', 30: 'multi_colored'
                }
            },
            'upperparts_color': {
                'attributes': list(range(31, 46)),
                'values': {
                    31: 'blue', 32: 'brown', 33: 'buff', 34: 'gray',
                    35: 'green', 36: 'pink', 37: 'purple', 38: 'red',
                    39: 'rufous', 40: 'white', 41: 'yellow', 42: 'black',
                    43: 'orange', 44: 'iridescent', 45: 'multi_colored'
                }
            },
            'underparts_color': {
                'attributes': list(range(46, 61)),
                'values': {
                    46: 'blue', 47: 'brown', 48: 'buff', 49: 'gray',
                    50: 'green', 51: 'pink', 52: 'purple', 53: 'red',
                    54: 'rufous', 55: 'white', 56: 'yellow', 57: 'black',
                    58: 'orange', 59: 'iridescent', 60: 'multi_colored'
                }
            },
            'breast_pattern': {
                'attributes': list(range(61, 71)),
                'values': {
                    61: 'solid', 62: 'spotted', 63: 'striped', 64: 'multi_colored',
                    65: 'streaked', 66: 'barred', 67: 'mottled', 68: 'scaled',
                    69: 'underparts_color', 70: 'breast_color'
                }
            },
            'back_pattern': {
                'attributes': list(range(71, 81)),
                'values': {
                    71: 'solid', 72: 'spotted', 73: 'striped', 74: 'multi_colored',
                    75: 'streaked', 76: 'barred', 77: 'mottled', 78: 'scaled',
                    79: 'back_color', 80: 'upperparts_color'
                }
            },
            'tail_pattern': {
                'attributes': list(range(81, 91)),
                'values': {
                    81: 'solid', 82: 'spotted', 83: 'striped', 84: 'multi_colored',
                    85: 'streaked', 86: 'barred', 87: 'mottled', 88: 'scaled',
                    89: 'tail_color', 90: 'forked'
                }
            },
            'wing_pattern': {
                'attributes': list(range(91, 101)),
                'values': {
                    91: 'solid', 92: 'spotted', 93: 'striped', 94: 'multi_colored',
                    95: 'streaked', 96: 'barred', 97: 'mottled', 98: 'scaled',
                    99: 'wing_color', 100: 'wing_shape'
                }
            },
            'head_pattern': {
                'attributes': list(range(101, 111)),
                'values': {
                    101: 'eyebrow', 102: 'eyering', 103: 'eyeline', 104: 'malar',
                    105: 'crest', 106: 'crown', 107: 'forehead', 108: 'nape',
                    109: 'throat', 110: 'head_color'
                }
            },
            'leg_color': {
                'attributes': list(range(111, 126)),
                'values': {
                    111: 'blue', 112: 'brown', 113: 'buff', 114: 'gray',
                    115: 'green', 116: 'pink', 117: 'purple', 118: 'red',
                    119: 'rufous', 120: 'white', 121: 'yellow', 122: 'black',
                    123: 'orange', 124: 'dark', 125: 'pale'
                }
            },
            'bill_color': {
                'attributes': list(range(126, 141)),
                'values': {
                    126: 'blue', 127: 'brown', 128: 'buff', 129: 'gray',
                    130: 'green', 131: 'pink', 132: 'purple', 133: 'red',
                    134: 'rufous', 135: 'white', 136: 'yellow', 137: 'black',
                    138: 'orange', 139: 'dark', 140: 'pale'
                }
            },
            'eye_color': {
                'attributes': list(range(141, 156)),
                'values': {
                    141: 'blue', 142: 'brown', 143: 'buff', 144: 'gray',
                    145: 'green', 146: 'pink', 147: 'purple', 148: 'red',
                    149: 'rufous', 150: 'white', 151: 'yellow', 152: 'black',
                    153: 'orange', 154: 'dark', 155: 'pale'
                }
            },
            'size': {
                'attributes': list(range(156, 166)),
                'values': {
                    156: 'large', 157: 'medium', 158: 'small', 159: 'very_large',
                    160: 'very_small', 161: 'perching_like', 162: 'upright_like',
                    163: 'crow_like', 164: 'long_legged_like', 165: 'duck_like'
                }
            },
            'shape': {
                'attributes': list(range(166, 176)),
                'values': {
                    166: 'perching_like', 167: 'upright_like', 168: 'crow_like',
                    169: 'long_legged_like', 170: 'duck_like', 171: 'owl_like',
                    172: 'gull_like', 173: 'hawk_like', 174: 'pigeon_like',
                    175: 'tree_clinging_like'
                }
            },
            'behavior': {
                'attributes': list(range(176, 201)),
                'values': {
                    i: f'behavior_{i-175}' for i in range(176, 201)
                }
            },
            'habitat': {
                'attributes': list(range(201, 251)),
                'values': {
                    i: f'habitat_{i-200}' for i in range(201, 251)
                }
            },
            'additional': {
                'attributes': list(range(251, 313)),
                'values': {
                    i: f'feature_{i-250}' for i in range(251, 313)
                }
            }
        }
        
        return mapping
    
    def get_active_attributes(self, 
                            attribute_probs: np.ndarray,
                            thresholds: Optional[Dict[str, float]] = None) -> List[int]:
        """
        Get list of active attribute indices based on probabilities and thresholds.
        
        Args:
            attribute_probs: Array of 312 attribute probabilities
            thresholds: Per-attribute thresholds (if None, use 0.5)
            
        Returns:
            List of active attribute indices (1-312)
        """
        active_attrs = []
        
        for i in range(312):
            attr_key = f'attr_{i+1}'
            threshold = thresholds.get(attr_key, 0.5) if thresholds else 0.5
            
            if attribute_probs[i] >= threshold:
                active_attrs.append(i + 1)  # 1-indexed
        
        return active_attrs
    
    def group_attributes_by_facet(self, active_attributes: List[int]) -> Dict[str, List[str]]:
        """
        Group active attributes by semantic facets.
        
        Args:
            active_attributes: List of active attribute indices
            
        Returns:
            Dictionary mapping facet names to lists of attribute descriptions
        """
        facet_attributes = defaultdict(list)
        
        for attr_idx in active_attributes:
            # Find which facet this attribute belongs to
            for facet_name, facet_info in self.attribute_map.items():
                if attr_idx in facet_info['attributes']:
                    attr_description = facet_info['values'].get(
                        attr_idx, f'attr_{attr_idx}'
                    )
                    facet_attributes[facet_name].append(attr_description)
                    break
        
        return dict(facet_attributes)
    
    def create_compact_text(self, facet_attributes: Dict[str, List[str]]) -> str:
        """
        Create compact text description from faceted attributes.
        
        Args:
            facet_attributes: Dictionary of facet -> attribute list
            
        Returns:
            Compact text description
        """
        text_parts = []
        
        # Define preferred order of facets for readability
        facet_order = [
            'size', 'shape', 'bill_shape', 'bill_color', 
            'upperparts_color', 'underparts_color', 'wing_color',
            'wing_pattern', 'breast_pattern', 'back_pattern', 
            'tail_pattern', 'head_pattern', 'leg_color', 'eye_color',
            'behavior', 'habitat'
        ]
        
        # Add attributes in preferred order
        for facet in facet_order:
            if facet in facet_attributes and facet_attributes[facet]:
                attrs = facet_attributes[facet]
                # Remove duplicates and clean up
                attrs = list(set(attrs))
                if len(attrs) == 1:
                    text_parts.append(f"{facet}: {attrs[0]}")
                else:
                    text_parts.append(f"{facet}: {', '.join(attrs)}")
        
        # Add any remaining facets not in preferred order
        for facet, attrs in facet_attributes.items():
            if facet not in facet_order and attrs:
                attrs = list(set(attrs))
                if len(attrs) == 1:
                    text_parts.append(f"{facet}: {attrs[0]}")
                else:
                    text_parts.append(f"{facet}: {', '.join(attrs)}")
        
        # Join with semicolons and ensure under 300 characters
        compact_text = "; ".join(text_parts)
        
        # Truncate if too long
        if len(compact_text) > 300:
            compact_text = compact_text[:297] + "..."
        
        return compact_text
    
    def create_structured_json(self, facet_attributes: Dict[str, List[str]]) -> Dict:
        """
        Create structured JSON representation.
        
        Args:
            facet_attributes: Dictionary of facet -> attribute list
            
        Returns:
            Structured JSON dictionary
        """
        # Clean up attributes (remove duplicates, sort)
        structured_json = {}
        
        for facet, attrs in facet_attributes.items():
            if attrs:
                # Remove duplicates and sort
                unique_attrs = sorted(list(set(attrs)))
                structured_json[facet] = unique_attrs
        
        return structured_json
    
    def verbalize_attributes(self,
                           attribute_probs: np.ndarray,
                           thresholds: Optional[Dict[str, float]] = None) -> Tuple[str, Dict]:
        """
        Main verbalization function.
        
        Args:
            attribute_probs: Array of 312 attribute probabilities
            thresholds: Per-attribute thresholds
            
        Returns:
            Tuple of (compact_text, structured_json)
        """
        # Get active attributes
        active_attrs = self.get_active_attributes(attribute_probs, thresholds)
        
        # Group by facets
        facet_attrs = self.group_attributes_by_facet(active_attrs)
        
        # Create text and JSON representations
        compact_text = self.create_compact_text(facet_attrs)
        structured_json = self.create_structured_json(facet_attrs)
        
        return compact_text, structured_json
    
    def batch_verbalize(self,
                       attribute_probs_batch: np.ndarray,
                       thresholds: Optional[Dict[str, float]] = None) -> List[Tuple[str, Dict]]:
        """
        Verbalize a batch of attribute predictions.
        
        Args:
            attribute_probs_batch: Array of shape (N, 312)
            thresholds: Per-attribute thresholds
            
        Returns:
            List of (compact_text, structured_json) tuples
        """
        results = []
        
        for i in range(len(attribute_probs_batch)):
            text, json_data = self.verbalize_attributes(
                attribute_probs_batch[i], thresholds
            )
            results.append((text, json_data))
        
        return results
    
    def save_attribute_mapping(self, output_path: str):
        """Save the current attribute mapping to a YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.attribute_map, f, default_flow_style=False, indent=2)
        
        print(f"✓ Attribute mapping saved to {output_path}")


def create_cub_attribute_mapping(output_path: str):
    """
    Create and save a complete CUB-200-2011 attribute mapping.
    
    In practice, this would parse the actual CUB attribute files to create
    a comprehensive mapping of all 312 attributes to their descriptions.
    """
    verbalizer = AttributeVerbalizer()
    verbalizer.save_attribute_mapping(output_path)


def main():
    """Example usage of attribute verbalizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test attribute verbalization")
    parser.add_argument("--create_mapping", action="store_true",
                       help="Create default attribute mapping file")
    parser.add_argument("--output", type=str, default="configs/attribute_mapping.yaml",
                       help="Output path for attribute mapping")
    parser.add_argument("--test", action="store_true",
                       help="Run test verbalization")
    
    args = parser.parse_args()
    
    if args.create_mapping:
        create_cub_attribute_mapping(args.output)
    
    if args.test:
        # Test with random attribute probabilities
        verbalizer = AttributeVerbalizer()
        
        # Generate random probabilities
        np.random.seed(42)
        test_probs = np.random.rand(312)
        
        # Test verbalization
        text, json_data = verbalizer.verbalize_attributes(test_probs)
        
        print("Test Verbalization:")
        print(f"Compact text: {text}")
        print(f"Structured JSON: {json.dumps(json_data, indent=2)}")
        print(f"Text length: {len(text)} characters")


if __name__ == "__main__":
    main()
