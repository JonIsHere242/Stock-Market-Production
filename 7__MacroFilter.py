#!/usr/bin/env python
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import yfinance as yf
from typing import Dict, List, Optional, Tuple
import time

from Util import get_logger

logger = get_logger(script_name="7__MacroFilter")

# ============ CONFIGURATION ============
API_KEY_FILE = "Claud-API-KEY.txt"

# Try to load API key from file first, then fall back to environment variable
ANTHROPIC_API_KEY = None
if os.path.exists(API_KEY_FILE):
    try:
        with open(API_KEY_FILE, 'r') as f:
            ANTHROPIC_API_KEY = f.read().strip()
            if not ANTHROPIC_API_KEY:
                print(f"Warning: {API_KEY_FILE} exists but is empty")
    except Exception as e:
        print(f"Warning: Could not read API key file: {e}")

if not ANTHROPIC_API_KEY:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", None)

SIGNALS_PATH = "Data/0__signals.parquet"
FILTER_RESULTS_PATH = "Data/ModelData/filter_results.json"
CACHE_PATH = "Data/ModelData/filter_run_cache.json"
RUBRIC_PATH = "FilterRubric.txt"

# Hardcoded thresholds from the rubric
MIN_PRICE = 2.00
MIN_VOLUME = 206_000  # Very low volume threshold
MEDIAN_VOLUME = 608_701  # Where performance degrades
HIGH_VOLUME = 1_640_000  # High volume threshold

# Market cap boundaries (in millions)
MICRO_CAP_MAX = 952
SMALL_CAP_MIN = 952
SMALL_CAP_MAX = 2_200
MID_CAP_MIN = 2_200
MID_CAP_MAX = 7_900
LARGE_CAP_MIN = 7_900

# ============ ARGUMENT PARSER ============
parser = argparse.ArgumentParser(description="Macro Filter for Trading Signals")
parser.add_argument("--mock", action="store_true", help="Run in mock mode without Claude API")
parser.add_argument("--dry-run", action="store_true", help="Don't modify the signals file, just report")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
parser.add_argument("--skip-claude", action="store_true", help="Skip Claude API calls, use only hardcoded checks (fastest, cheapest)")
parser.add_argument("--resume", action="store_true", help="Resume a previous run, skipping already-completed tickers and retrying errored ones")
args = parser.parse_args()

# ============ CACHE FUNCTIONS ============

def load_run_cache() -> Dict:
    """Load the persistent run cache from disk."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load run cache: {e}")
        return {}


def save_ticker_to_cache(date_key: str, ticker: str, result: Dict, status: str = "completed"):
    """
    Persist a ticker result to the run cache.
    status: 'completed' (full result saved) or 'errored' (API/network failure)
    On resume, completed tickers are skipped; errored tickers are retried.
    """
    cache = load_run_cache()
    if date_key not in cache:
        cache[date_key] = {"completed": {}, "errored": [], "run_timestamp": datetime.now().isoformat()}

    if status == "errored":
        if ticker not in cache[date_key]["errored"]:
            cache[date_key]["errored"].append(ticker)
    else:
        cache[date_key]["completed"][ticker] = result
        # Remove from errored list if it was previously failed but now succeeded
        if ticker in cache[date_key].get("errored", []):
            cache[date_key]["errored"].remove(ticker)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2, default=str)


# ============ HELPER FUNCTIONS ============

def get_ticker_fundamentals(ticker: str) -> Dict:
    """
    Fetch fundamental data for a ticker using yfinance.
    Returns dict with price, volume, market_cap, exchange.
    """
    try:
        logger.info(f"Fetching fundamentals for {ticker}...")
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get recent trading data for volume
        hist = stock.history(period="30d")
        avg_volume = hist['Volume'].mean() if len(hist) > 0 else 0

        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        if current_price == 0 and len(hist) > 0:
            current_price = hist['Close'].iloc[-1]

        # Get market cap
        market_cap = info.get('marketCap', 0)
        market_cap_millions = market_cap / 1_000_000 if market_cap else 0

        # Get exchange
        exchange = info.get('exchange', 'UNKNOWN')

        return {
            'ticker': ticker,
            'price': float(current_price),
            'avg_volume_30d': int(avg_volume),
            'market_cap_millions': float(market_cap_millions),
            'exchange': exchange,
            'success': True
        }
    except Exception as e:
        logger.error(f"Error fetching fundamentals for {ticker}: {e}")
        return {
            'ticker': ticker,
            'price': 0,
            'avg_volume_30d': 0,
            'market_cap_millions': 0,
            'exchange': 'UNKNOWN',
            'success': False,
            'error': str(e)
        }


def classify_market_cap(market_cap_millions: float) -> str:
    """Classify market cap into categories based on rubric."""
    if market_cap_millions < MICRO_CAP_MAX:
        return "Micro"
    elif market_cap_millions < SMALL_CAP_MAX:
        return "Small"
    elif market_cap_millions < MID_CAP_MAX:
        return "Mid"
    else:
        return "Large"


def classify_volume_tier(avg_volume: float) -> str:
    """Classify volume into tiers based on rubric."""
    if avg_volume >= HIGH_VOLUME:
        return "High"
    elif avg_volume >= MEDIAN_VOLUME:
        return "Medium"
    elif avg_volume >= MIN_VOLUME:
        return "Low"
    else:
        return "VeryLow"


def calculate_strategy_fit_score(fundamentals: Dict) -> Dict:
    """
    Calculate strategy fit score based on volume and market cap.
    Returns dict with volume_score, cap_score, premium_bonus, and total.
    """
    volume = fundamentals['avg_volume_30d']
    market_cap = fundamentals['market_cap_millions']

    # Volume tier score
    volume_tier = classify_volume_tier(volume)
    volume_score_map = {
        "High": 3,
        "Medium": 1,
        "Low": 0,
        "VeryLow": -2
    }
    volume_score = volume_score_map[volume_tier]

    # Market cap tier score
    cap_tier = classify_market_cap(market_cap)
    cap_score_map = {
        "Small": 3,
        "Mid": 1,
        "Large": 1,
        "Micro": -3
    }
    cap_score = cap_score_map[cap_tier]

    # Premium stock bonus (High Volume + Large Cap)
    premium_bonus = 2 if (volume_tier == "High" and cap_tier == "Large") else 0

    # Raw total (before normalization)
    raw_total = volume_score + cap_score + premium_bonus

    # Normalize to 1-10 scale (range is -5 to 8, so shift and scale)
    # Map -5 to 1, 8 to 10
    normalized_score = ((raw_total + 5) / 13) * 9 + 1
    normalized_score = max(1, min(10, normalized_score))

    return {
        'volume_tier': volume_tier,
        'volume_score': volume_score,
        'cap_tier': cap_tier,
        'cap_score': cap_score,
        'premium_bonus': premium_bonus,
        'raw_total': raw_total,
        'normalized_score': round(normalized_score, 2)
    }


def calculate_risk_adjustments(fundamentals: Dict, claude_analysis: Dict) -> Dict:
    """
    Calculate risk penalties based on fundamentals and Claude analysis.
    Returns dict with risk_points and list of concerns.
    """
    risk_points = 0
    concerns = []

    price = fundamentals['price']
    market_cap = fundamentals['market_cap_millions']
    volume = fundamentals['avg_volume_30d']

    # Price < $2.00
    if price < MIN_PRICE:
        risk_points += 4
        concerns.append(f"Price below ${MIN_PRICE:.2f} (delisting risk)")

    # Micro-cap
    if classify_market_cap(market_cap) == "Micro":
        risk_points += 2
        concerns.append("Micro-cap (historical profit factor: 0.11)")

    # Very low volume
    if classify_volume_tier(volume) == "VeryLow":
        risk_points += 1
        concerns.append(f"Very low volume (<{MIN_VOLUME:,})")

    # Active M&A (from Claude analysis)
    if claude_analysis.get('has_active_ma', False):
        risk_points += 3
        concerns.append(f"Active M&A: {claude_analysis.get('ma_details', 'Pending merger/acquisition')}")

    # Current crisis event (from Claude analysis)
    if claude_analysis.get('has_crisis_event', False):
        risk_points += 3
        concerns.append(f"Crisis event: {claude_analysis.get('crisis_details', 'Material negative event')}")

    return {
        'risk_points': risk_points,
        'concerns': concerns
    }


def research_ticker_with_claude(ticker: str, fundamentals: Dict) -> Dict:
    """
    Use Claude API with web search to research M&A status and current events.
    Returns dict with has_active_ma, ma_details, has_crisis_event, crisis_details.
    """
    if args.mock or args.skip_claude or not ANTHROPIC_API_KEY:
        mode = "mock" if args.mock else "skip-claude" if args.skip_claude else "no API key"
        logger.info(f"Skipping Claude API for {ticker} ({mode})")
        return {
            'ticker': ticker,
            'has_active_ma': False,
            'ma_details': None,
            'has_crisis_event': False,
            'crisis_details': None,
            'news_summary': f'Skipped ({mode})',
            'mock': True
        }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Build the research prompt - short and focused on recent news
        prompt = f"""Search for recent news about {ticker}. Focus on the last 14 days, but check up to 60 days for M&A.

Check for:
1. M&A activity (pending/announced mergers, acquisitions, buyout rumors)
2. Crisis events (regulatory issues, lawsuits, bankruptcy signals, scandals, recalls, major layoffs)

Context: Price ${fundamentals['price']:.2f}, Market Cap ${fundamentals['market_cap_millions']:.0f}M

Return JSON:
{{
    "has_active_ma": true/false,
    "ma_details": "description or null",
    "has_crisis_event": true/false,
    "crisis_details": "description or null",
    "news_summary": "2-3 sentence summary"
}}

Prioritize recent news. Only flag material events."""

        # Retry loop for rate limit (429) errors
        max_retries = 3
        retry_delays = [45, 90, 180]  # seconds between retries
        last_exception = None

        for attempt in range(max_retries):
            try:
                message = client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=1024,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    tools=[
                        {
                            "type": "web_search_20250305",
                            "name": "web_search"
                        }
                    ]
                )
                break  # success — exit retry loop
            except anthropic.RateLimitError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = retry_delays[attempt]
                    logger.warning(
                        f"Rate limit hit for {ticker} (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait_time}s before retry..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"Rate limit persisted after {max_retries} attempts for {ticker}: {e}")
                    raise
        else:
            # Should not reach here normally, but guard against it
            raise last_exception

        # Handle tool use (web search) in the response
        response_text = ""
        for block in message.content:
            if block.type == "text":
                response_text += block.text

        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        analysis = json.loads(response_text)
        analysis['ticker'] = ticker
        analysis['mock'] = False

        logger.info(f"Claude analysis for {ticker}: {json.dumps(analysis, indent=2)}")
        return analysis

    except Exception as e:
        logger.error(f"Error calling Claude API for {ticker}: {e}")
        return {
            'ticker': ticker,
            'has_active_ma': False,
            'ma_details': None,
            'has_crisis_event': False,
            'crisis_details': None,
            'news_summary': f'Error: {str(e)}',
            'mock': False,
            'error': str(e)
        }


def evaluate_ticker(ticker: str) -> Dict:
    """
    Main evaluation function that combines all checks.
    Returns comprehensive evaluation dict.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating ticker: {ticker}")
    logger.info(f"{'='*60}")

    # Step 1: Get fundamentals
    fundamentals = get_ticker_fundamentals(ticker)
    if not fundamentals['success']:
        logger.error(f"Failed to fetch fundamentals for {ticker}")
        return {
            'ticker': ticker,
            'pass': False,
            'reason': f"Failed to fetch data: {fundamentals.get('error', 'Unknown error')}",
            'final_score': 10,
            'recommendation': 'Exclude'
        }

    # Step 2: Hardcoded checks (fast exit)
    if fundamentals['price'] < MIN_PRICE:
        logger.warning(f"{ticker} FAILED: Price ${fundamentals['price']:.2f} below ${MIN_PRICE:.2f}")
        return {
            'ticker': ticker,
            'pass': False,
            'reason': f"Price ${fundamentals['price']:.2f} below ${MIN_PRICE:.2f} (delisting risk)",
            'final_score': 10,
            'recommendation': 'Exclude',
            'fundamentals': fundamentals
        }

    # Step 3: Calculate strategy fit score
    strategy_fit = calculate_strategy_fit_score(fundamentals)
    logger.info(f"Strategy Fit Score: {strategy_fit['normalized_score']}/10")

    # Step 4: Research with Claude (M&A and crisis events) - NOW WITH WEB SEARCH
    claude_analysis = research_ticker_with_claude(ticker, fundamentals)

    # Step 5: Calculate risk adjustments
    risk_adjustments = calculate_risk_adjustments(fundamentals, claude_analysis)

    # Step 6: Calculate final score
    final_score = strategy_fit['normalized_score'] + risk_adjustments['risk_points']
    final_score = min(10, max(1, final_score))

    # Step 7: Make decision based on rubric
    # Exclude if Final Score >= 7 or price < $2 or active M&A or material crisis
    should_exclude = (
        final_score >= 7 or
        fundamentals['price'] < MIN_PRICE or
        claude_analysis.get('has_active_ma', False) or
        claude_analysis.get('has_crisis_event', False)
    )

    # Conditional if score 4-6
    is_conditional = 4 <= final_score <= 6

    if should_exclude:
        recommendation = "Exclude"
        position_sizing = "No Position"
    elif is_conditional:
        recommendation = "Conditional"
        position_sizing = "Reduced (50%)"
    else:
        recommendation = "Include"
        position_sizing = "Standard"

    # Build result
    result = {
        'ticker': ticker,
        'pass': not should_exclude,
        'price': fundamentals['price'],
        'exchange': fundamentals['exchange'],
        'avg_volume_30d': fundamentals['avg_volume_30d'],
        'volume_tier': strategy_fit['volume_tier'],
        'market_cap_millions': fundamentals['market_cap_millions'],
        'cap_tier': strategy_fit['cap_tier'],
        'has_active_ma': claude_analysis.get('has_active_ma', False),
        'ma_details': claude_analysis.get('ma_details'),
        'has_crisis_event': claude_analysis.get('has_crisis_event', False),
        'crisis_details': claude_analysis.get('crisis_details'),
        'news_summary': claude_analysis.get('news_summary'),
        'strategy_fit_score': strategy_fit['normalized_score'],
        'volume_score': strategy_fit['volume_score'],
        'cap_score': strategy_fit['cap_score'],
        'premium_bonus': strategy_fit['premium_bonus'],
        'risk_points': risk_adjustments['risk_points'],
        'risk_concerns': risk_adjustments['concerns'],
        'final_score': round(final_score, 2),
        'recommendation': recommendation,
        'position_sizing': position_sizing,
        'reason': '; '.join(risk_adjustments['concerns']) if risk_adjustments['concerns'] else 'Passed all checks',
        'mock_mode': claude_analysis.get('mock', False)
    }

    # Log result
    logger.info(f"\n--- EVALUATION RESULT for {ticker} ---")
    logger.info(f"Price: ${result['price']:.2f} | Volume: {result['avg_volume_30d']:,} ({result['volume_tier']})")
    logger.info(f"Market Cap: ${result['market_cap_millions']:.0f}M ({result['cap_tier']})")
    logger.info(f"M&A: {result['has_active_ma']} | Crisis: {result['has_crisis_event']}")
    logger.info(f"Strategy Fit: {result['strategy_fit_score']}/10 | Risk Points: {result['risk_points']}")
    logger.info(f"FINAL SCORE: {result['final_score']}/10")
    logger.info(f"RECOMMENDATION: {result['recommendation']} ({result['position_sizing']})")
    if result['risk_concerns']:
        logger.info(f"Concerns: {'; '.join(result['risk_concerns'])}")
    logger.info(f"---")

    return result


# ============ MAIN EXECUTION ============

def main():
    logger.info("="*70)
    logger.info("MACRO FILTER - SIGNAL SCREENING")
    logger.info("="*70)

    if args.mock:
        logger.warning("Running in MOCK mode - Claude API calls will be simulated")
    if args.skip_claude:
        logger.warning("SKIP-CLAUDE mode - Only hardcoded checks will be used (no AI research)")
    if args.dry_run:
        logger.warning("DRY RUN mode - signals file will NOT be modified")

    # Check if API key is set
    if not args.mock and not args.skip_claude and not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not found!")
        logger.error("Please either:")
        logger.error(f"  1. Create a file '{API_KEY_FILE}' with your API key")
        logger.error("  2. Set ANTHROPIC_API_KEY environment variable")
        logger.error("  3. Run with --mock or --skip-claude flag")
        sys.exit(1)

    if ANTHROPIC_API_KEY and not args.mock and not args.skip_claude:
        logger.info(f"API key loaded successfully (from {'file' if os.path.exists(API_KEY_FILE) else 'environment'})")
        logger.info(f"API key: {ANTHROPIC_API_KEY[:8]}...{ANTHROPIC_API_KEY[-4:]} (masked)")

    # Load signals
    if not os.path.exists(SIGNALS_PATH):
        logger.error(f"Signals file not found: {SIGNALS_PATH}")
        sys.exit(1)

    logger.info(f"Loading signals from: {SIGNALS_PATH}")
    df_signals = pd.read_parquet(SIGNALS_PATH)
    logger.info(f"Loaded {len(df_signals)} total signals")

    # Filter to pending signals only
    pending_signals = df_signals[df_signals['Status'] == 'Pending'].copy()
    logger.info(f"Found {len(pending_signals)} pending signals")

    if len(pending_signals) == 0:
        logger.info("No pending signals to filter. Exiting.")
        return

    # Calculate next trading day (skip weekends)
    today = datetime.now().date()
    next_trading_day = today + timedelta(days=1)

    # Skip Saturday (5) and Sunday (6)
    while next_trading_day.weekday() >= 5:
        next_trading_day += timedelta(days=1)

    logger.info(f"Today: {today} | Next trading day: {next_trading_day}")

    # Filter to signals with TargetDate matching next trading day
    pending_signals['TargetDate'] = pd.to_datetime(pending_signals['TargetDate']).dt.date
    next_day_signals = pending_signals[pending_signals['TargetDate'] == next_trading_day].copy()

    logger.info(f"Signals for next trading day ({next_trading_day}): {len(next_day_signals)}")
    logger.info(f"Signals with other dates (skipped): {len(pending_signals) - len(next_day_signals)}")

    if len(next_day_signals) == 0:
        logger.info(f"No signals for next trading day ({next_trading_day}). Exiting.")
        return

    # Get unique tickers from next trading day signals
    tickers = next_day_signals['Symbol'].unique().tolist()
    logger.info(f"Unique tickers to evaluate for {next_trading_day}: {tickers}")

    # Load cache for resume support
    date_key = str(next_trading_day)
    date_cache = load_run_cache().get(date_key, {"completed": {}, "errored": []})

    if args.resume:
        completed_set = set(date_cache.get("completed", {}).keys())
        errored_set = set(date_cache.get("errored", []))
        results = list(date_cache.get("completed", {}).values())
        logger.info(
            f"Resume mode active for {date_key}: "
            f"{len(completed_set)} completed (skipping), "
            f"{len(errored_set)} errored (retrying): {sorted(errored_set)}"
        )
    else:
        completed_set = set()
        results = []

    # Determine which tickers still need processing
    tickers_to_run = [t for t in tickers if t not in completed_set]
    logger.info(f"Tickers to process: {tickers_to_run}")

    # Evaluate each ticker
    for i, ticker in enumerate(tickers_to_run):
        result = evaluate_ticker(ticker)
        results.append(result)

        # Detect whether the Claude API call errored (result carries an 'error' field
        # if research_ticker_with_claude returned an error dict)
        claude_had_error = result.get('news_summary', '').startswith('Error:')
        cache_status = "errored" if claude_had_error else "completed"
        save_ticker_to_cache(date_key, ticker, result, status=cache_status)

        if claude_had_error:
            logger.warning(
                f"Ticker {ticker} saved as ERRORED in cache — run with --resume to retry it."
            )

        # Rate limiting between tickers only (skip after the last one)
        is_last = (i == len(tickers_to_run) - 1)
        if not args.mock and not args.skip_claude and not is_last:
            logger.info(f"Rate limiting: waiting 60 seconds before next ticker...")
            time.sleep(60)

    # Save results to JSON
    logger.info(f"\nSaving filter results to: {FILTER_RESULTS_PATH}")
    os.makedirs(os.path.dirname(FILTER_RESULTS_PATH), exist_ok=True)
    with open(FILTER_RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    passed = [r for r in results if r['pass']]
    failed = [r for r in results if not r['pass']]

    logger.info("\n" + "="*70)
    logger.info("FILTER SUMMARY")
    logger.info("="*70)
    logger.info(f"Total evaluated: {len(results)}")
    logger.info(f"Passed: {len(passed)}")
    logger.info(f"Failed: {len(failed)}")

    if failed:
        logger.info("\nFailed tickers:")
        for r in failed:
            logger.info(f"  {r['ticker']}: {r['reason']} (Score: {r['final_score']}/10)")

    if passed:
        logger.info("\nPassed tickers:")
        for r in passed:
            logger.info(f"  {r['ticker']}: {r['recommendation']} - {r['position_sizing']} (Score: {r['final_score']}/10)")

    # Update signals file (remove failed tickers)
    if not args.dry_run:
        failed_tickers = [r['ticker'] for r in failed]
        if failed_tickers:
            logger.info(f"\nRemoving {len(failed_tickers)} failed tickers from signals file...")

            # Remove rows where Symbol is in failed_tickers and Status is Pending
            original_count = len(df_signals)
            df_signals = df_signals[~((df_signals['Symbol'].isin(failed_tickers)) & (df_signals['Status'] == 'Pending'))]
            removed_count = original_count - len(df_signals)

            # Save updated signals
            df_signals.to_parquet(SIGNALS_PATH, index=False)
            logger.info(f"Removed {removed_count} rows from signals file")
            logger.info(f"Signals file updated: {SIGNALS_PATH}")
        else:
            logger.info("\nNo tickers to remove - all passed!")
    else:
        logger.info("\nDRY RUN - signals file NOT modified")

    logger.info("\n" + "="*70)
    logger.info("MACRO FILTER COMPLETE")
    logger.info("="*70)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


