"""
Test Script for TPMS Pipeline
Provides comprehensive testing functionality for the submission-reviewer matching pipeline
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List
import sys

# Import the main pipeline
from tpms_pipeline import SubmissionReviewerPipeline, TPMSScore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PipelineTester:
    """Comprehensive testing class for TPMS pipeline"""
    
    def __init__(self):
        self.results = {}
        self.test_summary = {}
    
    def test_data_loading(self, submissions_path: str, profiles_path: str) -> bool:
        """Test data loading functionality"""
        logger.info("Testing data loading...")
        
        try:
            pipeline = SubmissionReviewerPipeline()
            pipeline.load_data(submissions_path, profiles_path)
            
            submissions_count = len(pipeline.submissions)
            profiles_count = len(pipeline.author_profiles)
            
            logger.info(f"✓ Successfully loaded {submissions_count} submissions")
            logger.info(f"✓ Successfully loaded {profiles_count} author profiles")
            
            # Validate data structure
            if submissions_count == 0:
                logger.error("✗ No submissions loaded")
                return False
            
            if profiles_count == 0:
                logger.error("✗ No author profiles loaded")
                return False
            
            # Check first submission
            first_submission = next(iter(pipeline.submissions.values()))
            if not first_submission.title or not first_submission.abstract:
                logger.warning("⚠ First submission missing title or abstract")
            
            # Check first profile
            first_profile = next(iter(pipeline.author_profiles.values()))
            if not first_profile.papers:
                logger.warning("⚠ First profile has no papers")
            
            self.test_summary['data_loading'] = {
                'passed': True,
                'submissions_count': submissions_count,
                'profiles_count': profiles_count
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Data loading failed: {e}")
            self.test_summary['data_loading'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_similarity_calculation(self, submissions_path: str, profiles_path: str, 
                                  methods: List[str] = ['tfidf']) -> bool:
        """Test similarity calculation with different methods"""
        logger.info("Testing similarity calculation...")
        
        all_passed = True
        
        for method in methods:
            logger.info(f"Testing {method} method...")
            
            try:
                pipeline = SubmissionReviewerPipeline(similarity_method=method)
                pipeline.load_data(submissions_path, profiles_path)
                
                # Test on first submission and first reviewer
                first_submission = next(iter(pipeline.submissions.values()))
                first_profile = next(iter(pipeline.author_profiles.values()))
                
                score = pipeline.tpms_calculator.calculate_tpms_score(first_submission, first_profile)
                
                # Validate score
                if not isinstance(score, TPMSScore):
                    logger.error(f"✗ {method}: Invalid score type returned")
                    all_passed = False
                    continue
                
                if not (0 <= score.score <= 1):
                    logger.error(f"✗ {method}: Score {score.score} not in range [0, 1]")
                    all_passed = False
                    continue
                
                logger.info(f"✓ {method}: Score = {score.score:.3f}")
                
                self.test_summary[f'similarity_{method}'] = {
                    'passed': True,
                    'sample_score': score.score
                }
                
            except Exception as e:
                logger.error(f"✗ {method} similarity calculation failed: {e}")
                self.test_summary[f'similarity_{method}'] = {'passed': False, 'error': str(e)}
                all_passed = False
        
        return all_passed
    
    def test_full_pipeline(self, submissions_path: str, profiles_path: str,
                          device: str = 'cpu', method: str = 'tfidf') -> bool:
        """Test the complete pipeline end-to-end"""
        logger.info("Testing full pipeline...")
        
        try:
            start_time = time.time()
            
            pipeline = SubmissionReviewerPipeline(device=device, similarity_method=method)
            results = pipeline.run_pipeline(
                submissions_path=submissions_path,
                profiles_path=profiles_path
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Validate results
            if not results:
                logger.error("✗ No results returned")
                return False
            
            total_scores = sum(len(scores) for scores in results.values())
            logger.info(f"✓ Generated {total_scores} scores in {processing_time:.2f} seconds")
            
            # Check result structure
            for submission_id, scores in results.items():
                if not scores:
                    logger.warning(f"⚠ No scores for submission {submission_id}")
                    continue
                
                # Check that we have scores for all reviewers
                expected_scores = len(pipeline.author_profiles)
                if len(scores) != expected_scores:
                    logger.error(f"✗ Expected {expected_scores} scores but got {len(scores)} for submission {submission_id}")
                    return False
            
            self.results = results
            self.test_summary['full_pipeline'] = {
                'passed': True,
                'processing_time': processing_time,
                'total_scores': total_scores,
                'submissions_processed': len(results)
            }
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Full pipeline test failed: {e}")
            self.test_summary['full_pipeline'] = {'passed': False, 'error': str(e)}
            return False
    
    def test_performance(self, submissions_path: str, profiles_path: str) -> bool:
        """Test pipeline performance with different configurations"""
        logger.info("Testing performance...")
        
        configurations = [
            {'device': 'cpu', 'method': 'tfidf'},
        ]
        
        # Add GPU test if available
        try:
            import torch
            if torch.cuda.is_available():
                configurations.append({'device': 'cuda', 'method': 'transformer'})
        except ImportError:
            pass
        
        performance_results = {}
        
        for config in configurations:
            config_name = f"{config['device']}_{config['method']}"
            logger.info(f"Testing configuration: {config_name}")
            
            try:
                start_time = time.time()
                
                pipeline = SubmissionReviewerPipeline(
                    device=config['device'], 
                    similarity_method=config['method']
                )
                results = pipeline.run_pipeline(
                    submissions_path=submissions_path,
                    profiles_path=profiles_path
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                performance_results[config_name] = {
                    'time': processing_time,
                    'scores_generated': sum(len(scores) for scores in results.values())
                }
                
                logger.info(f"✓ {config_name}: {processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"✗ {config_name} failed: {e}")
                performance_results[config_name] = {'error': str(e)}
        
        self.test_summary['performance'] = performance_results
        return len(performance_results) > 0
    
    def print_sample_results(self, num_samples: int = 2):
        """Print sample results for inspection"""
        if not self.results:
            logger.info("No results to display")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info("SAMPLE RESULTS")
        logger.info(f"{'='*60}")
        
        sample_submissions = list(self.results.items())[:num_samples]
        
        for submission_id, scores in sample_submissions:
            logger.info(f"\nSubmission: {submission_id}")
            logger.info("-" * 40)
            
            if not scores:
                logger.info("  No matching reviewers found")
                continue
            
            # Sort scores for display
            sorted_scores = sorted(scores, key=lambda x: x.score, reverse=True)
            
            for i, score in enumerate(sorted_scores[:5], 1):  # Show top 5
                logger.info(f"  {i}. Reviewer: {score.reviewer_id}")
                logger.info(f"     TPMS Score: {score.score:.4f}")
                if i < len(sorted_scores[:5]):
                    logger.info("")
    
    def export_test_results(self, output_path: str = "test_results.json"):
        """Export test results to JSON file"""
        test_data = {
            'test_summary': self.test_summary,
            'timestamp': time.time(),
            'sample_results': {}
        }
        
        # Add sample results
        if self.results:
            sample_submissions = list(self.results.items())[:3]
            for submission_id, scores in sample_submissions:
                test_data['sample_results'][submission_id] = [
                    {
                        'reviewer_id': score.reviewer_id,
                        'score': score.score
                    }
                    for score in scores[:5]
                ]
        
        with open(output_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Test results exported to {output_path}")
    
    def run_comprehensive_test(self, submissions_path: str, profiles_path: str,
                             device: str = 'cpu', method: str = 'tfidf') -> bool:
        """Run all tests comprehensively"""
        logger.info(f"\n{'='*60}")
        logger.info("COMPREHENSIVE TPMS PIPELINE TEST")
        logger.info(f"{'='*60}")
        logger.info(f"Submissions: {submissions_path}")
        logger.info(f"Profiles: {profiles_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Method: {method}")
        logger.info(f"{'='*60}")
        
        all_tests_passed = True
        
        # Test 1: Data Loading
        if not self.test_data_loading(submissions_path, profiles_path):
            all_tests_passed = False
        
        # Test 2: Similarity Calculation
        test_methods = [method]
        if method == 'tfidf':
            test_methods = ['tfidf']  # Only test requested method
        
        if not self.test_similarity_calculation(submissions_path, profiles_path, test_methods):
            all_tests_passed = False
        
        # Test 3: Full Pipeline
        if not self.test_full_pipeline(submissions_path, profiles_path, device, method):
            all_tests_passed = False
        
        # Test 4: Performance (optional)
        self.test_performance(submissions_path, profiles_path)
        
        # Print results
        self.print_test_summary()
        self.print_sample_results()
        
        return all_tests_passed
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_summary.items():
            total_tests += 1
            status = "✓ PASS" if result.get('passed', False) else "✗ FAIL"
            
            if result.get('passed', False):
                passed_tests += 1
            
            logger.info(f"{test_name.upper()}: {status}")
            
            if not result.get('passed', False) and 'error' in result:
                logger.info(f"  Error: {result['error']}")
            
            # Add specific details
            if test_name == 'data_loading' and result.get('passed'):
                logger.info(f"  Submissions: {result.get('submissions_count', 0)}")
                logger.info(f"  Profiles: {result.get('profiles_count', 0)}")
            
            elif test_name == 'full_pipeline' and result.get('passed'):
                logger.info(f"  Processing Time: {result.get('processing_time', 0):.2f}s")
                logger.info(f"  Total Scores: {result.get('total_scores', 0)}")
            
            elif test_name == 'performance':
                for config, perf in result.items():
                    if 'error' not in perf:
                        logger.info(f"  {config}: {perf.get('time', 0):.2f}s")
        
        logger.info(f"\nOVERALL: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            logger.info("🎉 All tests passed successfully!")
        else:
            logger.info("⚠️  Some tests failed. Check the details above.")

def main():
    """Main function for running tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test TPMS pipeline')
    parser.add_argument('--submissions', required=True,
                       help='Path to submissions JSON file')
    parser.add_argument('--profiles', required=True,
                       help='Path to author profiles JSON file')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use for testing')
    parser.add_argument('--method', choices=['tfidf', 'transformer'], default='tfidf',
                       help='Similarity calculation method')
    parser.add_argument('--export-results', action='store_true',
                       help='Export test results to JSON file')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run only basic tests (skip performance testing)')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.submissions).exists():
        logger.error(f"Submissions file not found: {args.submissions}")
        sys.exit(1)
    
    if not Path(args.profiles).exists():
        logger.error(f"Profiles file not found: {args.profiles}")
        sys.exit(1)
    
    # Run tests
    tester = PipelineTester()
    
    if args.quick_test:
        # Quick test: just data loading and basic pipeline
        logger.info("Running quick test...")
        success = (tester.test_data_loading(args.submissions, args.profiles) and
                  tester.test_full_pipeline(args.submissions, args.profiles, args.device, args.method))
        tester.print_sample_results()
    else:
        # Comprehensive test
        success = tester.run_comprehensive_test(
            args.submissions, args.profiles, args.device, args.method
        )
    
    # Export results if requested
    if args.export_results:
        tester.export_test_results()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()