"""
Submission-Reviewer Pipeline with TPMS (Toronto Paper Matching System) Score Calculation
Supports both local testing with mock data and GPU server deployment
"""

import json
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Paper:
    """Paper data structure"""
    id: str
    title: str
    abstract: str
    authors: List[str]
    author_ids: List[str]
    keywords: Optional[List[str]] = None
    venue: Optional[str] = None
    
    def get_text_content(self) -> str:
        """Get combined text content for similarity calculation (title + abstract only)"""
        return f"{self.title} {self.abstract}".strip()

@dataclass
class AuthorProfile:
    """Author profile data structure"""
    id: str
    papers: List[Paper]
    
    def get_expertise_text(self) -> str:
        """Get combined text representing author's expertise (title + abstract from all papers)"""
        all_content = []
        for paper in self.papers:
            paper_content = f"{paper.title} {paper.abstract}".strip()
            if paper_content:  # Only add non-empty content
                all_content.append(paper_content)
        return " ".join(all_content)

@dataclass
class TPMSScore:
    """TPMS score result"""
    submission_id: str
    reviewer_id: str
    score: float

class DataLoader:
    """Handles loading and parsing of submission and author profile data"""
    
    def __init__(self):
        self.submissions: Dict[str, Paper] = {}
        self.author_profiles: Dict[str, AuthorProfile] = {}
    
    def load_submissions(self, file_path: str) -> Dict[str, Paper]:
        """Load submissions from JSON file"""
        logger.info(f"Loading submissions from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        submissions = {}
        
        # Handle both single paper and list of papers
        if isinstance(data, list):
            papers_data = data
        else:
            papers_data = [data]
        
        for paper_data in papers_data:
            try:
                # Extract paper information
                paper_id = paper_data.get('id', '')
                
                # Handle different data structures
                if 'content' in paper_data:
                    content = paper_data['content']
                    if isinstance(content, dict):
                        # ICLR 2025 format
                        if 'title' in content and isinstance(content['title'], dict):
                            title = content['title'].get('value', '')
                            abstract = content.get('abstract', {}).get('value', '')
                            authors = content.get('authors', {}).get('value', [])
                            author_ids = content.get('authorids', {}).get('value', [])
                            keywords = content.get('keywords', {}).get('value', [])
                        else:
                            # DBLP format
                            title = content.get('title', '')
                            abstract = content.get('abstract', '')
                            authors = content.get('authors', [])
                            author_ids = content.get('authorids', [])
                            keywords = content.get('keywords', [])
                    else:
                        # Simple format
                        title = content
                        abstract = ''
                        authors = []
                        author_ids = []
                        keywords = []
                else:
                    # Direct format
                    title = paper_data.get('title', '')
                    abstract = paper_data.get('abstract', '')
                    authors = paper_data.get('authors', [])
                    author_ids = paper_data.get('authorids', [])
                    keywords = paper_data.get('keywords', [])
                
                paper = Paper(
                    id=paper_id,
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    author_ids=author_ids,
                    keywords=keywords
                )
                
                submissions[paper_id] = paper
                
            except Exception as e:
                logger.warning(f"Error processing paper {paper_data.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Loaded {len(submissions)} submissions")
        self.submissions = submissions
        return submissions
    
    def load_author_profiles(self, file_path: str) -> Dict[str, AuthorProfile]:
        """Load author profiles from JSON file with publications"""
        logger.info(f"Loading author profiles from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        profiles = {}
        
        # Handle both single profile and list of profiles
        if isinstance(data, list):
            profiles_data = data
        else:
            profiles_data = [data]
        
        for profile_data in profiles_data:
            try:
                author_id = profile_data.get('id', '')
                
                # Extract publications from the nested structure
                publications_data = []
                if 'content' in profile_data and 'publications' in profile_data['content']:
                    publications_data = profile_data['content']['publications']
                elif 'publications' in profile_data:
                    publications_data = profile_data['publications']
                
                papers = []
                for pub_data in publications_data:
                    try:
                        # Extract publication information
                        paper_id = pub_data.get('id', '')
                        
                        # Handle nested content structure
                        if 'content' in pub_data:
                            content = pub_data['content']
                        else:
                            content = pub_data
                        
                        title = content.get('title', '')
                        abstract = content.get('abstract', '')
                        authors = content.get('authors', [])
                        author_ids = content.get('authorids', [])
                        venue = content.get('venue', '')
                        
                        # Ensure we have at least title or abstract
                        if not title and not abstract:
                            continue
                            
                        paper = Paper(
                            id=paper_id,
                            title=title,
                            abstract=abstract,
                            authors=authors,
                            author_ids=author_ids,
                            venue=venue
                        )
                        papers.append(paper)
                        
                    except Exception as e:
                        logger.warning(f"Error processing publication in profile {author_id}: {e}")
                        continue
                
                if papers:  # Only create profile if we have valid papers
                    profile = AuthorProfile(id=author_id, papers=papers)
                    profiles[author_id] = profile
                    logger.debug(f"Loaded profile {author_id} with {len(papers)} papers")
                
            except Exception as e:
                logger.warning(f"Error processing profile {profile_data.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Loaded {len(profiles)} author profiles")
        self.author_profiles = profiles
        return profiles



class SemanticMatcher:
    """Handles semantic similarity calculation using various methods"""
    
    def __init__(self, method='tfidf', device='cpu'):
        self.method = method
        self.device = device
        self.vectorizer = None
        self.model = None
        self.tokenizer = None
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
        elif method == 'transformer':
            self._load_transformer_model()
    
    def _load_transformer_model(self):
        """Load transformer model for semantic similarity"""
        try:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            if torch.cuda.is_available() and self.device == 'cuda':
                self.model = self.model.cuda()
                logger.info("Using GPU for transformer model")
            else:
                logger.info("Using CPU for transformer model")
                
        except Exception as e:
            logger.warning(f"Failed to load transformer model: {e}")
            logger.info("Falling back to TF-IDF method")
            self.method = 'tfidf'
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    def _get_transformer_embedding(self, text: str) -> np.ndarray:
        """Get embedding using transformer model"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        
        if torch.cuda.is_available() and self.device == 'cuda':
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.method == 'tfidf':
            # Fit vectorizer on both texts
            corpus = [text1, text2]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
        elif self.method == 'transformer':
            emb1 = self._get_transformer_embedding(text1)
            emb2 = self._get_transformer_embedding(text2)
            similarity = cosine_similarity(emb1, emb2)[0][0]
        
        return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

class TPMSCalculator:
    """Calculates TPMS scores for submission-reviewer pairs based on semantic similarity"""
    
    def __init__(self, device='cpu', similarity_method='tfidf'):
        self.semantic_matcher = SemanticMatcher(method=similarity_method, device=device)
    
    def calculate_tpms_score(self, submission: Paper, reviewer_profile: AuthorProfile) -> TPMSScore:
        """Calculate TPMS score as semantic similarity between submission and reviewer's papers"""
        
        # Get submission text (title + abstract)
        submission_text = submission.get_text_content()
        
        # Get reviewer expertise text (all paper titles + abstracts)
        reviewer_expertise = reviewer_profile.get_expertise_text()
        
        # Handle empty texts
        if not submission_text.strip() or not reviewer_expertise.strip():
            score = 0.0
        else:
            # Calculate semantic similarity as TPMS score
            score = self.semantic_matcher.calculate_similarity(
                submission_text, reviewer_expertise
            )
        
        return TPMSScore(
            submission_id=submission.id,
            reviewer_id=reviewer_profile.id,
            score=score
        )

class SubmissionReviewerPipeline:
    """Main pipeline orchestrating the submission-reviewer matching process"""
    
    def __init__(self, device='cpu', similarity_method='tfidf'):
        self.device = device
        self.data_loader = DataLoader()
        self.tpms_calculator = TPMSCalculator(device=device, similarity_method=similarity_method)
        
        self.submissions = {}
        self.author_profiles = {}
        self.scores = {}
    
    def load_data(self, submissions_path: str, profiles_path: str):
        """Load submission and author profile data"""
        logger.info("Loading data...")
        self.submissions = self.data_loader.load_submissions(submissions_path)
        self.author_profiles = self.data_loader.load_author_profiles(profiles_path)
    
    def calculate_scores(self):
        """Calculate TPMS scores for all submission-reviewer pairs"""
        logger.info("Calculating TPMS scores...")
        
        total_calculations = len(self.submissions) * len(self.author_profiles)
        logger.info(f"Calculating scores for {total_calculations} submission-reviewer pairs")
        
        for submission_id, submission in self.submissions.items():
            submission_scores = []
            
            for reviewer_id, reviewer_profile in self.author_profiles.items():
                score = self.tpms_calculator.calculate_tpms_score(submission, reviewer_profile)
                submission_scores.append(score)
            
            # Store all scores without top-k filtering
            self.scores[submission_id] = submission_scores
    
    def export_results(self, output_path: str):
        """Export results summary to JSON file"""
        logger.info(f"Exporting results summary to {output_path}")
        
        summary = {
            'num_submissions': len(self.submissions),
            'num_reviewers': len(self.author_profiles),
            'total_scores_calculated': sum(len(scores) for scores in self.scores.values()),
            'created_at': time.time()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def export_tpms_matrix(self, output_path: str, format: str = 'torch'):
        """Export full TPMS similarity matrix in deep learning formats
        
        Args:
            output_path (str): Path to save the matrix
            format (str): Output format - 'torch', 'numpy', 'h5', or 'pickle'
        """
        logger.info(f"Exporting TPMS matrix to {output_path} in {format} format")
        
        if not self.scores:
            logger.error("No scores calculated. Run calculate_scores() first.")
            return
        
        # Get all submission and reviewer IDs
        submission_ids = list(self.submissions.keys())
        reviewer_ids = list(self.author_profiles.keys())
        
        # Create matrix
        import numpy as np
        tpms_matrix = np.zeros((len(submission_ids), len(reviewer_ids)), dtype=np.float32)
        
        # Fill matrix
        for i, submission_id in enumerate(submission_ids):
            reviewer_scores = {}
            if submission_id in self.scores:
                for score in self.scores[submission_id]:
                    reviewer_scores[score.reviewer_id] = score.score
            
            for j, reviewer_id in enumerate(reviewer_ids):
                if reviewer_id in reviewer_scores:
                    tpms_matrix[i, j] = reviewer_scores[reviewer_id]
                else:
                    tpms_matrix[i, j] = 0.0
        
        # Metadata
        metadata = {
            'submission_ids': submission_ids,
            'reviewer_ids': reviewer_ids,
            'shape': tpms_matrix.shape,
            'created_at': time.time(),
            'num_submissions': len(submission_ids),
            'num_reviewers': len(reviewer_ids)
        }
        
        if format == 'torch':
            import torch
            
            # Convert to PyTorch tensor
            tpms_tensor = torch.from_numpy(tpms_matrix)
            
            # Save as PyTorch data
            torch_data = {
                'tpms_matrix': tpms_tensor,
                'submission_ids': submission_ids,
                'reviewer_ids': reviewer_ids,
                'metadata': metadata
            }
            
            scores_path = output_path.replace('.pt', '_data.pt')
            torch.save(torch_data, scores_path)
            logger.info(f"Saved PyTorch tensor to {scores_path}")
            
        elif format == 'numpy':
            # Save as compressed NumPy array
            scores_path = output_path.replace('.npz', '_data.npz')
            np.savez_compressed(
                scores_path,
                tpms_matrix=tpms_matrix,
                submission_ids=np.array(submission_ids, dtype=object),
                reviewer_ids=np.array(reviewer_ids, dtype=object),
                **metadata
            )
            logger.info(f"Saved NumPy array to {scores_path}")
            
        elif format == 'h5':
            try:
                import h5py
                
                scores_path = output_path.replace('.h5', '_data.h5')
                with h5py.File(scores_path, 'w') as f:
                    # Create dataset
                    f.create_dataset('tpms_matrix', data=tpms_matrix, compression='gzip')
                    
                    # Save string arrays
                    dt = h5py.string_dtype(encoding='utf-8')
                    f.create_dataset('submission_ids', data=submission_ids, dtype=dt)
                    f.create_dataset('reviewer_ids', data=reviewer_ids, dtype=dt)
                    
                    # Save metadata as attributes
                    for key, value in metadata.items():
                        if isinstance(value, (int, float, str)):
                            f.attrs[key] = value
                        elif isinstance(value, tuple):
                            f.attrs[key] = list(value)
                
                logger.info(f"Saved HDF5 file to {scores_path}")
                
            except ImportError:
                logger.error("h5py not installed. Install with: pip install h5py")
                return
                
        elif format == 'pickle':
            import pickle
            
            # Save as pickle
            pickle_data = {
                'tpms_matrix': tpms_matrix,
                'submission_ids': submission_ids,
                'reviewer_ids': reviewer_ids,
                'metadata': metadata,
                'raw_scores': self.scores  # Include raw score objects
            }
            
            scores_path = output_path.replace('.pkl', '_data.pkl')
            with open(scores_path, 'wb') as f:
                pickle.dump(pickle_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved pickle file to {scores_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'torch', 'numpy', 'h5', or 'pickle'")
        
        logger.info(f"Matrix shape: {tpms_matrix.shape[0]} submissions × {tpms_matrix.shape[1]} reviewers")
        logger.info(f"Score range: [{tpms_matrix.min():.3f}, {tpms_matrix.max():.3f}]")
        logger.info(f"Non-zero scores: {np.count_nonzero(tpms_matrix)}/{tpms_matrix.size}")
        
        return {
            'path': scores_path if 'scores_path' in locals() else output_path,
            'shape': tpms_matrix.shape,
            'format': format,
            'metadata': metadata
        }
    
    def run_pipeline(self, submissions_path: str, profiles_path: str, 
                     output_path: str = None, matrix_path: str = None, 
                     matrix_format: str = 'torch'):
        """Run the complete pipeline
        
        Args:
            submissions_path (str): Path to submissions JSON file
            profiles_path (str): Path to author profiles JSON file
            output_path (str): Path to save summary (optional)
            matrix_path (str): Path to save full TPMS matrix (optional)
            matrix_format (str): Format for matrix export ('torch', 'numpy', 'h5', 'pickle')
        """
        logger.info("Starting submission-reviewer pipeline")
        
        # Load data
        self.load_data(submissions_path, profiles_path)
        
        # Calculate scores for all pairs
        self.calculate_scores()
        
        # Export summary if path provided
        if output_path:
            self.export_results(output_path)
        
        # Export full TPMS matrix if path provided
        if matrix_path:
            matrix_info = self.export_tpms_matrix(matrix_path, format=matrix_format)
            logger.info(f"Matrix exported with {matrix_info['shape'][0]}×{matrix_info['shape'][1]} dimensions")
        
        logger.info("Pipeline completed successfully")
        return self.scores



def main():
    """Main function for running the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run submission-reviewer pipeline')
    parser.add_argument('--submissions', required=True,
                       help='Path to submissions JSON file')
    parser.add_argument('--profiles', required=True, 
                       help='Path to author profiles JSON file')
    parser.add_argument('--output', default=None,
                       help='Output file for summary (optional)')
    parser.add_argument('--matrix', required=True,
                       help='Output file for full TPMS matrix')
    parser.add_argument('--matrix-format', choices=['torch', 'numpy', 'h5', 'pickle'], default='torch',
                       help='Format for matrix export')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use (cpu or cuda)')
    parser.add_argument('--method', choices=['tfidf', 'transformer'], default='tfidf',
                       help='Similarity calculation method')
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not Path(args.submissions).exists() or not Path(args.profiles).exists():
        logger.error("Data files not found. Please provide valid file paths.")
        return
    
    # Initialize and run pipeline
    pipeline = SubmissionReviewerPipeline(device=args.device, similarity_method=args.method)
    
    try:
        results = pipeline.run_pipeline(
            submissions_path=args.submissions,
            profiles_path=args.profiles,
            output_path=args.output,
            matrix_path=args.matrix,
            matrix_format=args.matrix_format
        )
        
        # Print summary statistics
        logger.info(f"\nPipeline Summary:")
        logger.info(f"Submissions processed: {len(pipeline.submissions)}")
        logger.info(f"Reviewers processed: {len(pipeline.author_profiles)}")
        logger.info(f"Total scores calculated: {sum(len(scores) for scores in results.values())}")
        logger.info(f"Matrix saved to: {args.matrix}")
        logger.info(f"Matrix format: {args.matrix_format}")
        
        # Print sample scores
        if results:
            sample_scores = []
            for scores in results.values():
                sample_scores.extend([s.score for s in scores])
            
            if sample_scores:
                logger.info(f"Score statistics:")
                logger.info(f"  Min: {min(sample_scores):.3f}")
                logger.info(f"  Max: {max(sample_scores):.3f}")
                logger.info(f"  Mean: {sum(sample_scores)/len(sample_scores):.3f}")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()