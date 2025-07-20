"""
Mock Data Generator for TPMS Pipeline Testing
Creates realistic test data that matches ICLR submission and author profile formats
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockDataGenerator:
    """Generates mock data for testing the TPMS pipeline"""
    
    def __init__(self):
        self.output_dir = Path("mock_data")
    
    def create_mock_submissions(self) -> List[Dict]:
        """Create mock ICLR submissions"""
        return [
            {
                "id": "sub_001",
                "content": {
                    "title": {"value": "Deep Learning for Natural Language Processing"},
                    "abstract": {"value": "This paper presents a novel approach to NLP using deep learning techniques including transformers and attention mechanisms. We demonstrate significant improvements on standard benchmarks including GLUE and SuperGLUE. Our method combines self-attention with novel positional encodings to achieve better understanding of long-range dependencies in text."},
                    "authors": {"value": ["Alice Smith", "Bob Johnson"]},
                    "authorids": {"value": ["~Alice_Smith1", "~Bob_Johnson1"]},
                    "keywords": {"value": ["deep learning", "NLP", "transformers"]}
                }
            },
            {
                "id": "sub_002", 
                "content": {
                    "title": {"value": "Computer Vision with Convolutional Neural Networks"},
                    "abstract": {"value": "We explore advanced CNN architectures for image classification and object detection tasks. Our method achieves state-of-the-art results on ImageNet and COCO datasets. We introduce a novel attention mechanism for convolutional layers that improves feature extraction and reduces computational overhead."},
                    "authors": {"value": ["Charlie Brown", "Diana Prince"]},
                    "authorids": {"value": ["~Charlie_Brown1", "~Diana_Prince1"]},
                    "keywords": {"value": ["computer vision", "CNN", "image classification"]}
                }
            },
            {
                "id": "sub_003",
                "content": {
                    "title": {"value": "Reinforcement Learning for Robotics"},
                    "abstract": {"value": "This work presents a novel reinforcement learning framework for robotic control tasks. We develop a policy gradient method that learns from minimal human demonstrations and achieves superior performance on manipulation tasks. Our approach combines imitation learning with exploration strategies to handle sparse reward environments."},
                    "authors": {"value": ["Eve Wilson", "Frank Miller"]},
                    "authorids": {"value": ["~Eve_Wilson1", "~Frank_Miller1"]},
                    "keywords": {"value": ["reinforcement learning", "robotics", "policy gradient"]}
                }
            },
            {
                "id": "sub_004",
                "content": {
                    "title": {"value": "Graph Neural Networks for Social Network Analysis"},
                    "abstract": {"value": "We propose a new graph neural network architecture for analyzing social networks and predicting user behavior. Our model incorporates temporal dynamics and multi-modal user features to achieve better prediction accuracy. Experiments on real-world social media datasets demonstrate the effectiveness of our approach."},
                    "authors": {"value": ["Grace Lee", "Henry Davis"]},
                    "authorids": {"value": ["~Grace_Lee1", "~Henry_Davis1"]},
                    "keywords": {"value": ["graph neural networks", "social networks", "behavior prediction"]}
                }
            }
        ]
    
    def create_mock_author_profiles(self) -> List[Dict]:
        """Create mock author profiles with publication history"""
        return [
            {
                "id": "~Reviewer_One1",
                "content": {
                    "publications": [
                        {
                            "id": "paper_r1_1",
                            "content": {
                                "title": "Attention Mechanisms in Neural Networks",
                                "abstract": "A comprehensive study of attention mechanisms and their applications in deep learning. We analyze various attention architectures including self-attention, cross-attention, and multi-head attention. Our analysis provides insights into when and why different attention mechanisms work best for specific tasks.",
                                "authors": ["Reviewer One"],
                                "authorids": ["~Reviewer_One1"],
                                "venue": "ICLR 2023"
                            }
                        },
                        {
                            "id": "paper_r1_2", 
                            "content": {
                                "title": "Language Models and Text Generation",
                                "abstract": "Exploring large language models for text generation tasks. We present novel training techniques and evaluation metrics for generative models. Our work focuses on improving coherence and factual accuracy in generated text through better training objectives and architectural improvements.",
                                "authors": ["Reviewer One", "Collaborator A"],
                                "authorids": ["~Reviewer_One1", "~Collaborator_A1"],
                                "venue": "NeurIPS 2022"
                            }
                        },
                        {
                            "id": "paper_r1_3",
                            "content": {
                                "title": "Transformer Architectures for Natural Language Understanding",
                                "abstract": "We investigate different transformer architectures for natural language understanding tasks. Our study compares various positional encoding schemes and attention patterns. Results show that task-specific architectural choices can significantly improve performance on downstream NLP tasks.",
                                "authors": ["Reviewer One", "Co-author B", "Co-author C"],
                                "authorids": ["~Reviewer_One1", "~Co_author_B1", "~Co_author_C1"],
                                "venue": "ACL 2022"
                            }
                        }
                    ]
                }
            },
            {
                "id": "~Reviewer_Two1",
                "content": {
                    "publications": [
                        {
                            "id": "paper_r2_1",
                            "content": {
                                "title": "Object Detection in Images",
                                "abstract": "Novel approaches to object detection using deep convolutional networks. We introduce a new architecture that improves detection accuracy and speed. Our method combines feature pyramid networks with attention mechanisms to better detect objects at different scales and in cluttered scenes.",
                                "authors": ["Reviewer Two"],
                                "authorids": ["~Reviewer_Two1"],
                                "venue": "CVPR 2023"
                            }
                        },
                        {
                            "id": "paper_r2_2",
                            "content": {
                                "title": "Image Segmentation with Deep Learning",
                                "abstract": "Advanced techniques for semantic segmentation using neural networks. Our method achieves superior performance on standard benchmarks including Cityscapes and PASCAL VOC. We propose a novel loss function that better handles class imbalance and boundary detection in segmentation tasks.",
                                "authors": ["Reviewer Two", "Co-author B"],
                                "authorids": ["~Reviewer_Two1", "~Co_author_B1"],
                                "venue": "ICCV 2022"
                            }
                        },
                        {
                            "id": "paper_r2_3",
                            "content": {
                                "title": "Convolutional Neural Networks for Medical Image Analysis",
                                "abstract": "Application of CNNs to medical image analysis tasks including disease detection and organ segmentation. We develop specialized architectures that handle the unique challenges of medical imaging such as limited data and high precision requirements.",
                                "authors": ["Reviewer Two", "Medical Expert A", "Medical Expert B"],
                                "authorids": ["~Reviewer_Two1", "~Medical_Expert_A1", "~Medical_Expert_B1"],
                                "venue": "MICCAI 2021"
                            }
                        }
                    ]
                }
            },
            {
                "id": "~Reviewer_Three1",
                "content": {
                    "publications": [
                        {
                            "id": "paper_r3_1",
                            "content": {
                                "title": "Policy Gradient Methods in Reinforcement Learning",
                                "abstract": "We study policy gradient methods for reinforcement learning and propose improvements to existing algorithms. Our work focuses on reducing variance in gradient estimates and improving sample efficiency. Experiments on robotic control tasks demonstrate the effectiveness of our approach.",
                                "authors": ["Reviewer Three"],
                                "authorids": ["~Reviewer_Three1"],
                                "venue": "ICML 2023"
                            }
                        },
                        {
                            "id": "paper_r3_2",
                            "content": {
                                "title": "Deep Reinforcement Learning for Game Playing",
                                "abstract": "Application of deep reinforcement learning to complex game environments. We develop new exploration strategies and network architectures that achieve superhuman performance on various games. Our method combines model-free and model-based approaches for better sample efficiency.",
                                "authors": ["Reviewer Three", "Game AI Expert"],
                                "authorids": ["~Reviewer_Three1", "~Game_AI_Expert1"],
                                "venue": "NeurIPS 2022"
                            }
                        }
                    ]
                }
            },
            {
                "id": "~Reviewer_Four1",
                "content": {
                    "publications": [
                        {
                            "id": "paper_r4_1",
                            "content": {
                                "title": "Graph Convolutional Networks for Node Classification",
                                "abstract": "We propose improved graph convolutional networks for node classification tasks. Our method handles heterogeneous graphs and incorporates node features more effectively. Experiments on citation networks and social graphs show significant improvements over baseline methods.",
                                "authors": ["Reviewer Four"],
                                "authorids": ["~Reviewer_Four1"],
                                "venue": "ICLR 2023"
                            }
                        },
                        {
                            "id": "paper_r4_2",
                            "content": {
                                "title": "Social Network Analysis with Machine Learning",
                                "abstract": "Machine learning approaches for analyzing social networks and predicting user behavior. We develop models that capture both structural and temporal patterns in social interactions. Our work has applications in recommendation systems and influence analysis.",
                                "authors": ["Reviewer Four", "Social Science Expert"],
                                "authorids": ["~Reviewer_Four1", "~Social_Science_Expert1"],
                                "venue": "WWW 2022"
                            }
                        },
                        {
                            "id": "paper_r4_3",
                            "content": {
                                "title": "Graph Neural Networks for Molecular Property Prediction",
                                "abstract": "Application of graph neural networks to molecular property prediction tasks. We design specialized architectures that capture chemical bond information and molecular symmetries. Our method achieves state-of-the-art results on quantum chemistry benchmarks.",
                                "authors": ["Reviewer Four", "Chemistry Expert A", "Chemistry Expert B"],
                                "authorids": ["~Reviewer_Four1", "~Chemistry_Expert_A1", "~Chemistry_Expert_B1"],
                                "venue": "ICML 2021"
                            }
                        }
                    ]
                }
            }
        ]
    
    def generate_mock_data(self) -> None:
        """Generate and save mock data files"""
        logger.info("Generating mock data...")
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Generate submissions
        submissions = self.create_mock_submissions()
        submissions_file = self.output_dir / "mock_submissions.json"
        with open(submissions_file, "w", encoding='utf-8') as f:
            json.dump(submissions, f, indent=2, ensure_ascii=False)
        
        # Generate author profiles
        profiles = self.create_mock_author_profiles()
        profiles_file = self.output_dir / "mock_profiles.json"
        with open(profiles_file, "w", encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Mock data generated:")
        logger.info(f"  - {len(submissions)} submissions saved to {submissions_file}")
        logger.info(f"  - {len(profiles)} author profiles saved to {profiles_file}")
        
        # Generate statistics
        total_publications = sum(len(profile['content']['publications']) for profile in profiles)
        logger.info(f"  - Total publications in profiles: {total_publications}")

def main():
    """Main function for generating mock data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate mock data for TPMS pipeline testing')
    parser.add_argument('--output-dir', default='mock_data',
                       help='Output directory for mock data files')
    
    args = parser.parse_args()
    
    generator = MockDataGenerator()
    generator.output_dir = Path(args.output_dir)
    generator.generate_mock_data()
    
    print("\nMock data generated successfully!")
    print(f"Files created in {generator.output_dir}/:")
    print("  - mock_submissions.json")
    print("  - mock_profiles.json")
    print("\nYou can now test the pipeline with:")
    print(f"python tpms_pipeline.py --submissions {generator.output_dir}/mock_submissions.json --profiles {generator.output_dir}/mock_profiles.json")

if __name__ == "__main__":
    main()