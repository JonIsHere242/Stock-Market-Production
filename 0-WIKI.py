#!/usr/bin/env python3
"""
Trading Alpha Discovery System
Recursively searches Wikipedia for trading-related terms and ranks them by alpha potential.
"""

import requests
import time
import re
import argparse
import json
import csv
from collections import defaultdict
from urllib.parse import quote
import sys
from typing import List, Dict, Optional, Tuple

class ComplexityAnalyzer:
    """Analyzes text complexity using various linguistic metrics."""
    
    def __init__(self):
        # Common trading/finance terms for technical density calculation
        self.finance_terms = {
            'arbitrage', 'volatility', 'momentum', 'reversal', 'spread', 'delta', 'gamma', 
            'theta', 'vega', 'hedge', 'portfolio', 'correlation', 'regression', 'stochastic',
            'brownian', 'monte', 'carlo', 'scholes', 'binomial', 'derivatives', 'quantitative',
            'algorithmic', 'liquidity', 'execution', 'slippage', 'alpha', 'beta', 'sharpe',
            'sortino', 'drawdown', 'backtesting', 'optimization', 'cointegration', 'stationary'
        }
    
    def calculate_complexity(self, text: str) -> float:
        """Calculate complexity score based on multiple linguistic features."""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Basic text statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not sentences or not words:
            return 0.0
        
        # 1. Average sentence length (normalized)
        avg_sentence_length = len(words) / len(sentences)
        sentence_complexity = min(1.0, avg_sentence_length / 25)  # 25 words = max complexity
        
        # 2. Average word length (normalized)
        avg_word_length = sum(len(word) for word in words) / len(words)
        word_complexity = min(1.0, avg_word_length / 8)  # 8 chars = max complexity
        
        # 3. Technical term density
        finance_word_count = sum(1 for word in words if word in self.finance_terms)
        technical_density = finance_word_count / len(words)
        
        # 4. Long word density (8+ characters)
        long_words = sum(1 for word in words if len(word) >= 8)
        long_word_density = long_words / len(words)
        
        # 5. Unique word ratio (vocabulary richness)
        unique_ratio = len(set(words)) / len(words)
        
        # 6. Mathematical/scientific notation density
        math_patterns = len(re.findall(r'[α-ωΑ-Ω]|[∀-⋿]|\b\d+\.\d+\b|\b[a-z]\([a-z,\s]*\)', text))
        math_density = min(1.0, math_patterns / len(words) * 10)
        
        # Weighted combination of complexity factors
        complexity = (
            sentence_complexity * 0.15 +      # Sentence structure
            word_complexity * 0.15 +          # Word length
            technical_density * 0.25 +        # Finance-specific terms
            long_word_density * 0.20 +        # Academic vocabulary
            unique_ratio * 0.15 +             # Vocabulary richness
            math_density * 0.10               # Mathematical notation
        )
        
        return round(complexity, 3)


class WikipediaSearcher:
    """Handles Wikipedia API interactions."""
    
    def __init__(self, delay: float = 0.5):
        self.base_url = "https://en.wikipedia.org/w/api.php"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingAlphaDiscovery/1.0 (Educational Research)'
        })
    
    def search_articles(self, query: str, limit: int = 10) -> List[str]:
        """Search for articles matching the query."""
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': query,
            'srlimit': limit,
            'srprop': 'titlesnippet'
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'query' in data and 'search' in data['query']:
                return [result['title'] for result in data['query']['search']]
            return []
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    def get_article_content(self, title: str) -> Optional[Dict]:
        """Get full article content and metadata."""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': title,
            'prop': 'extracts|info',
            'explaintext': True,
            'inprop': 'url'
        }
        
        try:
            response = self.session.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            pages = data.get('query', {}).get('pages', {})
            page = next(iter(pages.values()))
            
            if 'missing' in page:
                return None
            
            extract = page.get('extract', '')
            word_count = len(extract.split())
            
            # Skip very short articles and disambiguation pages
            if word_count < 150:  # Increased minimum to filter out disambiguation pages
                return None
                
            # Skip disambiguation pages based on content patterns
            content_lower = extract.lower()
            if ('may refer to:' in content_lower or 
                'disambiguation page' in content_lower or
                extract.count('\n') > word_count / 10):  # Too many line breaks = list page
                return None
            
            return {
                'title': page.get('title', title),
                'content': extract,
                'url': page.get('fullurl', ''),
                'word_count': word_count
            }
        except Exception as e:
            print(f"Error fetching '{title}': {e}")
            return None
    
    def extract_related_terms(self, content: str) -> List[str]:
        """Extract potential trading-related terms from content."""
        if not content:
            return []
        
        # Look for terms that might lead to interesting articles
        patterns = [
            r'\b([A-Z][a-z]+ (?:theorem|model|equation|method|strategy|effect|paradox))\b',
            r'\b([A-Z][a-z]+-[A-Z][a-z]+ (?:model|theorem|test))\b',
            r'\b((?:Black-Scholes|Monte Carlo|Kelly|Sharpe|Sortino|Value at Risk)[^.]*?)\b',
            r'\b([a-z]+ (?:arbitrage|trading|hedging|pricing|optimization))\b',
            r'\b((?:algorithmic|quantitative|statistical|stochastic) [a-z]+)\b'
        ]
        
        terms = []
        content_lower = content.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            terms.extend(matches[:3])  # Limit per pattern
        
        # Also look for capitalized terms that might be proper nouns
        proper_nouns = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', content)
        finance_related = [
            term for term in proper_nouns[:10] 
            if any(keyword in term.lower() for keyword in 
                  ['market', 'option', 'bond', 'stock', 'fund', 'index', 'ratio', 'model'])
        ]
        
        terms.extend(finance_related)
        return list(set(terms))[:5]  # Return unique terms, limited


class AlphaDiscoverySystem:
    """Main system for discovering trading alpha through Wikipedia analysis."""
    
    def __init__(self, delay: float = 0.5):
        self.analyzer = ComplexityAnalyzer()
        self.searcher = WikipediaSearcher(delay)
        self.processed_articles = set()
        self.results = []
    
    def calculate_alpha_score(self, word_count: int, complexity: float) -> float:
        """Calculate alpha potential score based on word count and complexity."""
        # Inverse relationship with word count (shorter = better)
        if word_count <= 1000:
            length_factor = 1.0
        elif word_count <= 3000:
            length_factor = 0.8
        elif word_count <= 5000:
            length_factor = 0.6
        else:
            length_factor = max(0.1, (10000 - word_count) / 10000)
        
        # Direct relationship with complexity (more complex = better)
        complexity_factor = complexity
        
        # Weighted combination favoring complexity
        alpha_score = (complexity_factor * 0.75) + (length_factor * 0.25)
        return round(alpha_score, 3)
    
    def analyze_article(self, title: str, depth: int = 0) -> Optional[Dict]:
        """Analyze a single article and return metrics."""
        if title in self.processed_articles:
            return None
        
        print(f"{'  ' * depth}Analyzing: {title}")
        self.processed_articles.add(title)
        
        article = self.searcher.get_article_content(title)
        if not article:
            return None
        
        # Calculate metrics
        complexity = self.analyzer.calculate_complexity(article['content'])
        alpha_score = self.calculate_alpha_score(article['word_count'], complexity)
        
        result = {
            'title': article['title'],
            'word_count': article['word_count'],
            'complexity': complexity,
            'alpha_score': alpha_score,
            'url': article['url'],
            'depth': depth,
            'summary': article['content'][:300] + '...' if len(article['content']) > 300 else article['content']
        }
        
        return result
    
    def recursive_search(self, 
                        seed_terms: List[str], 
                        max_depth: int = 2,
                        min_complexity: float = 0.3,
                        max_word_count: int = 5000,
                        min_alpha_score: float = 0.0) -> List[Dict]:
        """Recursively search and analyze articles."""
        
        print(f"Starting recursive search with {len(seed_terms)} seed terms")
        print(f"Max depth: {max_depth}, Min complexity: {min_complexity}")
        print(f"Max word count: {max_word_count}, Min alpha score: {min_alpha_score}")
        print("-" * 60)
        
        queue = [(term, 0) for term in seed_terms]
        
        while queue:
            current_term, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Search for articles matching this term
            article_titles = self.searcher.search_articles(current_term, limit=3)
            
            for title in article_titles:
                result = self.analyze_article(title, depth)
                
                if result and self._meets_criteria(result, min_complexity, max_word_count, min_alpha_score):
                    self.results.append(result)
                    
                    # Extract related terms for next level
                    if depth < max_depth - 1:
                        article_content = self.searcher.get_article_content(title)
                        if article_content:
                            related_terms = self.searcher.extract_related_terms(article_content['content'])
                            for term in related_terms[:2]:  # Limit to prevent explosion
                                if term not in [item[0] for item in queue]:
                                    queue.append((term, depth + 1))
                
                time.sleep(self.searcher.delay)
        
        return self.results
    
    def _meets_criteria(self, result: Dict, min_complexity: float, max_word_count: int, min_alpha_score: float) -> bool:
        """Check if article meets filtering criteria."""
        return (result['complexity'] >= min_complexity and 
                result['word_count'] <= max_word_count and
                result['alpha_score'] >= min_alpha_score)
    
    def print_results(self, limit: int = 20):
        """Print results sorted by alpha score."""
        if not self.results:
            print("No results found.")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['alpha_score'], reverse=True)
        
        print(f"\n🔍 ALPHA DISCOVERY RESULTS (Top {min(limit, len(sorted_results))})")
        print("=" * 80)
        
        for i, result in enumerate(sorted_results[:limit], 1):
            print(f"\n{i}. {result['title']}")
            print(f"   Alpha Score: {result['alpha_score']:.3f} | "
                  f"Complexity: {result['complexity']:.3f} | "
                  f"Words: {result['word_count']:,} | "
                  f"Depth: {result['depth']}")
            print(f"   URL: {result['url']}")
            print(f"   Summary: {result['summary']}")
            print("-" * 80)
    
    def export_csv(self, filename: str):
        """Export results to CSV file."""
        if not self.results:
            print("No results to export.")
            return
        
        sorted_results = sorted(self.results, key=lambda x: x['alpha_score'], reverse=True)
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['rank', 'title', 'alpha_score', 'complexity', 'word_count', 'depth', 'url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i, result in enumerate(sorted_results, 1):
                writer.writerow({
                    'rank': i,
                    'title': result['title'],
                    'alpha_score': result['alpha_score'],
                    'complexity': result['complexity'],
                    'word_count': result['word_count'],
                    'depth': result['depth'],
                    'url': result['url']
                })
        
        print(f"Results exported to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Trading Alpha Discovery System')
    parser.add_argument('--terms', nargs='+',
                       default=['stochastic resonance', 'multiplicative cascades', '1_f_noise',
                                'fractional Brownian motion', 'nonlinear echo state networks',
                                'synaptic stochasticity', 'path-dependent volatility surfaces',
                                'Gibbs-Markov random fields', 'subordination of Lévy processes',
                                'Markov-switching multifractals', 'noise-induced phase transitions',
                                'perceptual quantization error', 'complex demodulation',
                                'phase vocoder artifacts', 'psychoacoustic masking thresholds',
                                'interaural time difference estimation', 'chaotic intermittency',
                                'wavelet modulus maxima', 'topological signal analysis',
                                'variational Bayesian changepoint detection'],
                       help='Seed terms to start search')

    parser.add_argument('--depth', type=int, default=2, 
                       help='Maximum search depth (default: 2)')
    parser.add_argument('--min-complexity', type=float, default=0.3,
                       help='Minimum complexity score (default: 0.3)')
    parser.add_argument('--max-words', type=int, default=5000,
                       help='Maximum word count (default: 5000)')
    parser.add_argument('--min-alpha', type=float, default=0.0,
                       help='Minimum alpha score (default: 0.0)')
    parser.add_argument('--limit', type=int, default=20,
                       help='Number of results to display (default: 20)')
    parser.add_argument('--export', type=str,
                       help='Export results to CSV file')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API calls in seconds (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize system
    system = AlphaDiscoverySystem(delay=args.delay)
    
    # Run analysis
    print("🚀 Trading Alpha Discovery System")
    print(f"Seed terms: {', '.join(args.terms)}")
    
    try:
        results = system.recursive_search(
            seed_terms=args.terms,
            max_depth=args.depth,
            min_complexity=args.min_complexity,
            max_word_count=args.max_words,
            min_alpha_score=args.min_alpha
        )
        
        # Display results
        system.print_results(limit=args.limit)
        
        # Export if requested
        if args.export:
            system.export_csv(args.export)
        
        print(f"\nTotal articles analyzed: {len(system.processed_articles)}")
        print(f"Articles meeting criteria: {len(results)}")
        
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
        system.print_results(limit=args.limit)
    except Exception as e:
        print(f"Error during analysis: {e}")



if __name__ == "__main__":
    main()


