"""
RAGAs evaluation module using native RAGAs without LangChain.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class RAGAsEvaluator:
    """RAGAs evaluator using native RAGAs without LangChain."""
    
    def __init__(self):
        """Initialize the RAGAs evaluator."""
        logger.info("RAGAs evaluator initialized")
        
        # Check if RAGAs is available
        self.ragas_available = self._check_ragas_availability()
        
        if self.ragas_available:
            self._initialize_ragas()
        else:
            logger.warning("⚠️ RAGAs not available, using fallback metrics")
            self._initialize_fallback_metrics()
        
        # Initialize evaluation history
        self.evaluation_history = []
        
        # Create evaluation data directory
        os.makedirs("data/evaluation", exist_ok=True)
    
    def _check_ragas_availability(self) -> bool:
        """Check if RAGAs is available."""
        try:
            import ragas
            logger.info("✅ RAGAs is available")
            return True
        except ImportError:
            logger.warning("⚠️ RAGAs not installed")
            return False
    
    def _initialize_ragas(self):
        """Initialize RAGAs components with enhanced metrics."""
        try:
            from ragas import evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_recall,
                answer_correctness,
                precision,
                recall
            )
            
            self.evaluate = evaluate
            self.metrics = [
                faithfulness,
                answer_relevancy,
                context_relevancy,
                context_recall,
                answer_correctness,
                precision,
                recall
            ]
            
            logger.info(f"✅ RAGAs initialized with {len(self.metrics)} enhanced metrics")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize RAGAs: {e}")
            self.ragas_available = False
            self._initialize_fallback_metrics()
    
    def _initialize_fallback_metrics(self):
        """Initialize fallback metrics when RAGAs is not available."""
        self.metrics = []
        logger.info("✅ Fallback metrics initialized")
    
    def evaluate_response(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a response using RAGAs or fallback metrics.
        
        Args:
            query (str): The user query
            response (str): The system response
            sources (List[Dict[str, Any]]): Source documents used
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            if self.ragas_available:
                return self._evaluate_with_ragas(query, response, sources)
            else:
                return self._evaluate_with_fallback(query, response, sources)
                
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            return self._evaluate_with_fallback(query, response, sources)
    
    def _evaluate_with_ragas(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate using native RAGAs."""
        try:
            # Prepare data for RAGAs
            from datasets import Dataset
            
            # Create context from sources
            context = "\n".join([source.get('chunk_text', '') for source in sources])
            
            # Create dataset
            data = {
                "question": [query],
                "answer": [response],
                "contexts": [[context]],
                "ground_truth": [""]  # No ground truth available
            }
            
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            results = self.evaluate(dataset, self.metrics)
            
            # Extract scores
            scores = {}
            for metric_name, score in results.items():
                if isinstance(score, dict) and 'score' in score:
                    scores[metric_name] = score['score']
                else:
                    scores[metric_name] = score
            
            # Store evaluation with enhanced data
            evaluation_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'sources_count': len(sources),
                'scores': scores,
                'method': 'ragas',
                'model_used': 'llama2:7b',
                'source_documents': str(sources),
                'metadata': {
                    'ragas_version': 'native',
                    'metrics_count': len(scores),
                    'evaluation_type': 'ragas_native',
                    'metrics_included': ['faithfulness', 'answer_relevancy', 'context_relevancy', 'context_recall', 'answer_correctness', 'precision', 'recall']
                }
            }
            self.evaluation_history.append(evaluation_data)
            
            logger.info(f"✅ RAGAs evaluation completed with {len(scores)} metrics")
            
            return {
                'method': 'ragas',
                'scores': scores,
                'metrics_used': list(scores.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ RAGAs evaluation failed: {e}")
            return self._evaluate_with_fallback(query, response, sources)
    
    def _evaluate_with_fallback(self, query: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate using fallback metrics including precision, recall, faithfulness, and answer correctness."""
        try:
            # Enhanced fallback metrics
            scores = {}
            
            # Response length score
            response_length = len(response)
            scores['response_length'] = min(response_length / 100, 1.0)  # Normalize to 0-1
            
            # Source utilization score
            source_count = len(sources)
            scores['source_utilization'] = min(source_count / 5, 1.0)  # Normalize to 0-1
            
            # Query-response relevance (simple keyword matching)
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            if query_words:
                relevance = len(query_words.intersection(response_words)) / len(query_words)
                scores['keyword_relevance'] = relevance
            else:
                scores['keyword_relevance'] = 0.0
            
            # Calculate Precision (how many relevant words in response)
            if response_words:
                precision = len(query_words.intersection(response_words)) / len(response_words)
                scores['precision'] = precision
            else:
                scores['precision'] = 0.0
            
            # Calculate Recall (how many query words are covered in response)
            if query_words:
                recall = len(query_words.intersection(response_words)) / len(query_words)
                scores['recall'] = recall
            else:
                scores['recall'] = 0.0
            
            # Calculate Faithfulness (how well response follows source content)
            if sources:
                source_text = " ".join([str(source.get('text', '')) for source in sources]).lower()
                source_words = set(source_text.split())
                response_words_set = set(response.lower().split())
                
                # Faithfulness: how many response words are in source
                if response_words_set:
                    faithfulness = len(response_words_set.intersection(source_words)) / len(response_words_set)
                    scores['faithfulness'] = faithfulness
                else:
                    scores['faithfulness'] = 0.0
            else:
                scores['faithfulness'] = 0.0
            
            # Calculate Answer Correctness (combination of precision, recall, and faithfulness)
            correctness_components = []
            if 'precision' in scores:
                correctness_components.append(scores['precision'])
            if 'recall' in scores:
                correctness_components.append(scores['recall'])
            if 'faithfulness' in scores:
                correctness_components.append(scores['faithfulness'])
            
            if correctness_components:
                scores['answer_correctness'] = sum(correctness_components) / len(correctness_components)
            else:
                scores['answer_correctness'] = 0.0
            
            # Overall score (average of all available metrics)
            available_scores = [v for v in scores.values() if v is not None]
            if available_scores:
                scores['overall_score'] = sum(available_scores) / len(available_scores)
            else:
                scores['overall_score'] = 0.0
            
            # Store evaluation with enhanced data
            evaluation_data = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'sources_count': len(sources),
                'scores': scores,
                'method': 'fallback',
                'model_used': 'llama2:7b',
                'source_documents': str(sources),
                'metadata': {
                    'fallback_metrics': 'enhanced_rag_metrics',
                    'metrics_count': len(scores),
                    'evaluation_type': 'fallback_enhanced',
                    'metrics_included': ['precision', 'recall', 'faithfulness', 'answer_correctness']
                }
            }
            self.evaluation_history.append(evaluation_data)
            
            logger.info(f"✅ Enhanced fallback evaluation completed with {len(scores)} metrics")
            
            return {
                'method': 'fallback',
                'scores': scores,
                'metrics_used': list(scores.keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Fallback evaluation failed: {e}")
            return {
                'method': 'error',
                'scores': {'error': 0.0},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def export_evaluation_data(self, filename: Optional[str] = None, simplified: bool = True) -> str:
        """
        Export evaluation data to CSV with simplified fields.
        
        Args:
            filename (Optional[str]): Custom filename
            simplified (bool): If True, export only essential fields
            
        Returns:
            str: Path to exported file
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/evaluation/ragas_evaluation_data_{timestamp}.csv"
            
            # Convert evaluation history to DataFrame
            data = []
            for eval_data in self.evaluation_history:
                if simplified:
                    # Simplified version with only essential fields
                    row = {
                        'timestamp': eval_data.get('timestamp', ''),
                        'query': eval_data.get('query', ''),
                        'response': eval_data.get('response', ''),
                        'sources_count': eval_data.get('sources_count', 0),
                        'processing_time': eval_data.get('processing_time', 0.0),
                        'evaluation_method': eval_data.get('method', 'unknown'),
                        'model_used': eval_data.get('model_used', 'llama2:7b')
                    }
                    
                    # Add only the main RAG evaluation scores
                    scores = eval_data.get('scores', {})
                    for metric in ['precision', 'recall', 'faithfulness', 'answer_correctness', 'overall_score']:
                        if metric in scores:
                            row[f'score_{metric}'] = scores[metric]
                        else:
                            row[f'score_{metric}'] = 0.0
                    
                else:
                    # Full version with all fields (original behavior)
                    row = {
                        'timestamp': eval_data.get('timestamp', ''),
                        'query': eval_data.get('query', ''),
                        'response': eval_data.get('response', ''),
                        'sources_count': eval_data.get('sources_count', 0),
                        'processing_time': eval_data.get('processing_time', 0.0),
                        'evaluation_method': eval_data.get('method', 'unknown'),
                        'model_used': eval_data.get('model_used', 'llama2:7b'),
                        'query_length': len(eval_data.get('query', '')),
                        'response_length': len(eval_data.get('response', '')),
                        'source_documents': eval_data.get('source_documents', ''),
                        'trace_id': eval_data.get('trace_id', ''),
                        'session_id': eval_data.get('session_id', ''),
                        'user_id': eval_data.get('user_id', ''),
                        'confidence_score': eval_data.get('confidence_score', 0.0),
                        'error_message': eval_data.get('error', ''),
                        'evaluation_status': 'success' if not eval_data.get('error') else 'error'
                    }
                    
                    # Add all scores
                    scores = eval_data.get('scores', {})
                    for metric, score in scores.items():
                        row[f'score_{metric}'] = score
                    
                    # Add metadata
                    metadata = eval_data.get('metadata', {})
                    for key, value in metadata.items():
                        row[f'metadata_{key}'] = value
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
            field_count = len(df.columns)
            logger.info(f"✅ Evaluation data exported to {filename} ({field_count} fields)")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to export evaluation data: {e}")
            return ""
    
    def export_evaluation_data_to_csv(self) -> bool:
        """
        Export evaluation data to CSV (alias for export_evaluation_data).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            filename = self.export_evaluation_data()
            return bool(filename)
        except Exception as e:
            logger.error(f"❌ Failed to export evaluation data to CSV: {e}")
            return False
    
    def generate_evaluation_summary_csv(self) -> bool:
        """
        Generate and export evaluation summary to CSV.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            summary = self.get_evaluation_summary()
            
            if summary['total_evaluations'] == 0:
                logger.warning("⚠️ No evaluation data to summarize")
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/evaluation/ragas_evaluation_summary_{timestamp}.csv"
            
            # Create enhanced summary data
            summary_data = {
                'metric': [],
                'average_score': [],
                'min_score': [],
                'max_score': [],
                'std_deviation': [],
                'total_evaluations': [],
                'method_used': [],
                'last_evaluation': [],
                'model_used': [],
                'total_processing_time': [],
                'avg_processing_time': [],
                'total_queries': [],
                'total_responses': [],
                'avg_query_length': [],
                'avg_response_length': [],
                'success_rate': [],
                'error_count': []
            }
            
            # Calculate enhanced statistics
            import statistics
            
            # Calculate processing time statistics
            processing_times = [eval_data.get('processing_time', 0) for eval_data in self.evaluation_history]
            total_processing_time = sum(processing_times)
            avg_processing_time = total_processing_time / len(processing_times) if processing_times else 0
            
            # Calculate query/response length statistics
            query_lengths = [eval_data.get('query_length', 0) for eval_data in self.evaluation_history]
            response_lengths = [eval_data.get('response_length', 0) for eval_data in self.evaluation_history]
            avg_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
            avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
            
            # Calculate success rate
            success_count = sum(1 for eval_data in self.evaluation_history if not eval_data.get('error'))
            success_rate = success_count / len(self.evaluation_history) if self.evaluation_history else 0
            error_count = len(self.evaluation_history) - success_count
            
            # Add enhanced average scores
            for metric, score in summary['average_scores'].items():
                # Calculate min, max, std for this metric
                metric_scores = [eval_data.get('scores', {}).get(metric, 0) for eval_data in self.evaluation_history]
                min_score = min(metric_scores) if metric_scores else 0
                max_score = max(metric_scores) if metric_scores else 0
                std_dev = statistics.stdev(metric_scores) if len(metric_scores) > 1 else 0
                
                summary_data['metric'].append(metric)
                summary_data['average_score'].append(score)
                summary_data['min_score'].append(min_score)
                summary_data['max_score'].append(max_score)
                summary_data['std_deviation'].append(std_dev)
                summary_data['total_evaluations'].append(summary['total_evaluations'])
                summary_data['method_used'].append(summary['method_used'])
                summary_data['last_evaluation'].append(summary['last_evaluation'])
                summary_data['model_used'].append('llama2:7b')
                summary_data['total_processing_time'].append(total_processing_time)
                summary_data['avg_processing_time'].append(avg_processing_time)
                summary_data['total_queries'].append(len(self.evaluation_history))
                summary_data['total_responses'].append(len(self.evaluation_history))
                summary_data['avg_query_length'].append(avg_query_length)
                summary_data['avg_response_length'].append(avg_response_length)
                summary_data['success_rate'].append(success_rate)
                summary_data['error_count'].append(error_count)
            
            # Add overall statistics
            overall_score = sum(summary['average_scores'].values()) / len(summary['average_scores']) if summary['average_scores'] else 0
            summary_data['metric'].append('overall')
            summary_data['average_score'].append(overall_score)
            summary_data['min_score'].append(0)  # Overall min would need different calculation
            summary_data['max_score'].append(1)  # Overall max would need different calculation
            summary_data['std_deviation'].append(0)  # Overall std would need different calculation
            summary_data['total_evaluations'].append(summary['total_evaluations'])
            summary_data['method_used'].append(summary['method_used'])
            summary_data['last_evaluation'].append(summary['last_evaluation'])
            summary_data['model_used'].append('llama2:7b')
            summary_data['total_processing_time'].append(total_processing_time)
            summary_data['avg_processing_time'].append(avg_processing_time)
            summary_data['total_queries'].append(len(self.evaluation_history))
            summary_data['total_responses'].append(len(self.evaluation_history))
            summary_data['avg_query_length'].append(avg_query_length)
            summary_data['avg_response_length'].append(avg_response_length)
            summary_data['success_rate'].append(success_rate)
            summary_data['error_count'].append(error_count)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(filename, index=False)
            
            logger.info(f"✅ Evaluation summary exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to generate evaluation summary CSV: {e}")
            return False
    
    def trace_query_execution(self, query: str, response: str, source_documents: List[Dict[str, Any]], processing_time: float = 0.0, trace_id: str = "", session_id: str = "", user_id: str = "") -> Dict[str, Any]:
        """
        Trace query execution and store evaluation data with enhanced information.
        
        Args:
            query (str): User query
            response (str): System response
            source_documents (List[Dict[str, Any]]): Source documents used
            processing_time (float): Processing time in seconds
            trace_id (str): Phoenix trace ID
            session_id (str): Session identifier
            user_id (str): User identifier
            
        Returns:
            Dict[str, Any]: Evaluation result
        """
        try:
            # Evaluate the response
            evaluation_result = self.evaluate_response(query, response, source_documents)
            
            # Add enhanced information
            evaluation_result['processing_time'] = processing_time
            evaluation_result['trace_id'] = trace_id
            evaluation_result['session_id'] = session_id
            evaluation_result['user_id'] = user_id
            evaluation_result['query_length'] = len(query)
            evaluation_result['response_length'] = len(response)
            evaluation_result['source_documents'] = str(source_documents)
            
            # Calculate confidence score (simple heuristic)
            scores = evaluation_result.get('scores', {})
            if scores:
                confidence_score = sum(scores.values()) / len(scores)
                evaluation_result['confidence_score'] = confidence_score
            else:
                evaluation_result['confidence_score'] = 0.0
            
            # Store in history
            self.evaluation_history.append(evaluation_result)
            
            logger.info(f"✅ Query execution traced and evaluated with enhanced data")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"❌ Failed to trace query execution: {e}")
            return {
                'method': 'error',
                'scores': {'error': 0.0},
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time': processing_time
            }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary statistics."""
        try:
            if not self.evaluation_history:
                return {
                    'total_evaluations': 0,
                    'average_scores': {},
                    'method_used': 'none'
                }
            
            # Calculate average scores
            all_scores = {}
            for eval_data in self.evaluation_history:
                for metric, score in eval_data['scores'].items():
                    if metric not in all_scores:
                        all_scores[metric] = []
                    all_scores[metric].append(score)
            
            average_scores = {}
            for metric, scores in all_scores.items():
                if scores:
                    average_scores[metric] = sum(scores) / len(scores)
            
            return {
                'total_evaluations': len(self.evaluation_history),
                'average_scores': average_scores,
                'method_used': self.evaluation_history[-1].get('method', 'unknown') if self.evaluation_history else 'none',
                'last_evaluation': self.evaluation_history[-1]['timestamp'] if self.evaluation_history else None
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get evaluation summary: {e}")
            return {
                'total_evaluations': 0,
                'average_scores': {},
                'method_used': 'error',
                'error': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get evaluator status."""
        return {
            'ragas_available': self.ragas_available,
            'metrics_configured': [metric.__name__ if hasattr(metric, '__name__') else str(metric) for metric in self.metrics],
            'evaluation_history_count': len(self.evaluation_history),
            'fallback_metrics_available': not self.ragas_available,
            'capabilities': [
                'native_ragas_evaluation' if self.ragas_available else 'fallback_metrics',
                'csv_export',
                'evaluation_summary'
            ]
        } 