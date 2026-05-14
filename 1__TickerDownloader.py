#!/root/root/miniconda4/envs/tf/bin/python
import datetime
import os
import re
import requests
import pandas as pd
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys
from Util import get_logger, LogPerformance, dprint

CONFIG = {
    "url": "https://www.sec.gov/files/company_tickers_exchange.json",
    "parquet_file_path": "Data/TickerCikData/TickerCIKs_{date}.parquet",
    "user_agent": "MarketAnalysis NotMyRealEmail@gmail.com"
}

def setup_args():
    parser = argparse.ArgumentParser(description="Download and filter Ticker CIK data.")
    parser.add_argument("--ImmediateDownload", action='store_true', help="Download the file immediately without waiting for the scheduled time.")
    parser.add_argument("-v", "--verbose", action='store_true', help="Increase output verbosity")
    parser.add_argument("--save-unfiltered", action='store_true', help="Also save unfiltered dataset with _raw suffix")
    return parser.parse_args()

def is_problematic_ticker(company_name):
    """
    Identify problematic tickers that should be filtered out
    """
    if not isinstance(company_name, str):
        return False
    
    name = company_name.lower().strip()
    
    # AGGRESSIVE FILTERING - if ANY of these words appear, reject immediately
    instant_reject_words = [
        'acquisition',
        'spac', 
        'etf',
        'fund',
        'reit',
        'warrant',
        'preferred',
        'holdings',
        'vehicle',
        'ventures',
        'income',
        'ai',
        'blockchain',
        'metaverse',
        'crypto',
        'merger',
        'capital',
        'bitcoin'
        
    ]
    
    for word in instant_reject_words:
        if word in name:
            return True
    
    # Additional patterns for problematic tickers
    problematic_patterns = [
        # Specific phrases
        r'special purpose',
        r'blank check',
        r'shell company',
        r'merger corp',
        r'exchange traded',
        r'real estate investment',
        r'mutual fund',
        r'index fund',
        r'investment trust',
        r'income fund',
        r'bond fund',
        r'growth fund',
        r'dividend fund',
        r'closed.*end',
        r'open.*end',
        
        # Popular fund providers
        r'vanguard',
        r'ishares',
        r'spdr',
        r'invesco',
        r'direxion',
        r'proshares',
        r'wisdomtree',
        r'blackrock',
        r'fidelity',
        
        # Capital patterns (be more specific to avoid false positives)
        r'capital.*corp\.?\s+(i{2,}|iv|v|vi{1,3}|ix|x|\d)',
        r'capital.*investment.*corp\.?\s+(i{2,}|iv|v|vi{1,3}|ix|x|\d)',
        
        # Roman numerals and numbers after corp/inc/llc (shells and SPACs)
        r'(corp|inc|ltd|llc|company|co)\.?\s+(i{2,}|iv|v|vi{1,3}|ix|x)',
        r'(corp|inc|ltd|llc|company|co)\.?\s+[2-9]',
        
        # Leveraged products
        r'\d+x\s',
        r'leveraged',
        r'inverse',
        r'bear.*etf',
        r'bull.*etf',
        r'ultra.*short',
        r'ultra.*long',
        r'volatility',
        r'\bvix\b',
        
        # Rights and units
        r'\bright[s]?\b',
        r'\bunit[s]?\b',
        r'when issued',
        r'\bstub[s]?\b',
        
        # Series and class designations
        r'series [a-z]',
        r'class [a-z]',
        
        # Distressed
        r'bankruptcy',
        r'liquidat',
        r'defunct',
        r'dissolved',
        r'delisted',
        r'chapter 11',
        
        # Generic names
        r'unknown',
        r'placeholder',
        r'temporary',
        r'test.*corp',
        r'^tbd\b',
        r'^tba\b',
    ]
    
    # Check against all patterns
    for pattern in problematic_patterns:
        if re.search(pattern, name, re.IGNORECASE):
            return True
    
    return False

def download_and_process_data(logger, args):
    try:
        dprint("Starting ticker data download and processing", level="INFO")
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        
        # Download data
        with LogPerformance("SEC ticker data download", logger=logger):
            session = requests.Session()
            retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
            adapter = HTTPAdapter(max_retries=retry)
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            
            headers = {
                'User-Agent': CONFIG["user_agent"],
                'Accept-Encoding': 'gzip, deflate',
                'Host': 'www.sec.gov'
            }
            
            dprint(f"Making request to {CONFIG['url']}", level="INFO")
            response = session.get(CONFIG["url"], headers=headers, timeout=30)
            response.raise_for_status()
            dprint("Request successful", level="SUCCESS")

            json_data = response.json()
            dprint(f"Received JSON data with {len(json_data['data'])} entries", level="INFO")
            
            # Create DataFrame
            df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
            dprint(f"Created DataFrame with shape {df.shape}", level="INFO")

            # Basic filtering - remove OTC and CBOE
            df = df[df['exchange'].notna()]
            df = df[~df['exchange'].isin(['OTC', 'CBOE'])]
            dprint(f"After exchange filtering: {df.shape}", level="INFO")

        # Filter problematic tickers
        with LogPerformance("Filtering problematic tickers", logger=logger):
            dprint("Identifying problematic tickers...", level="INFO")
            
            # Apply filtering function to the 'name' column
            problematic_mask = df['name'].apply(is_problematic_ticker)
            problematic_count = problematic_mask.sum()
            clean_count = len(df) - problematic_count
            
            dprint(f"Found {problematic_count} problematic tickers, {clean_count} clean tickers", level="INFO")
            
            # Show examples of what's being filtered if verbose
            if args.verbose and problematic_count > 0:
                dprint("Examples of problematic tickers being filtered:", level="INFO")
                problematic_examples = df[problematic_mask].head(10)
                for _, row in problematic_examples.iterrows():
                    dprint(f"  {row['ticker']:8} | {row['name'][:60]}", level="INFO")
                
                dprint("Examples of clean tickers being kept:", level="INFO")
                clean_examples = df[~problematic_mask].head(10)
                for _, row in clean_examples.iterrows():
                    dprint(f"  {row['ticker']:8} | {row['name'][:60]}", level="INFO")
            
            # Create filtered dataset
            clean_df = df[~problematic_mask].copy()

        # Save files
        os.makedirs(os.path.dirname(CONFIG["parquet_file_path"].format(date=current_date)), exist_ok=True)
        
        results = []
        
        # Save filtered (clean) dataset as main output
        main_path = CONFIG["parquet_file_path"].format(date=current_date)
        with LogPerformance("Saving filtered dataset", logger=logger):
            clean_df.to_parquet(main_path, index=False)
            results.append(("filtered (clean) dataset", main_path, len(clean_df)))
            dprint(f"Saved {len(clean_df)} clean tickers to {main_path}", level="SUCCESS")
        
        # Optionally save unfiltered dataset
        if args.save_unfiltered:
            raw_path = CONFIG["parquet_file_path"].format(date=current_date).replace('.parquet', '_raw.parquet')
            with LogPerformance("Saving unfiltered dataset", logger=logger):
                df.to_parquet(raw_path, index=False)
                results.append(("unfiltered (raw) dataset", raw_path, len(df)))
                dprint(f"Saved {len(df)} raw tickers to {raw_path}", level="SUCCESS")
        
        return results

    except requests.exceptions.RequestException as e:
        logger.error(f"Request error occurred: {e}")
        dprint(f"Request error: {e}", level="ERROR")
        raise
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        dprint(f"Error: {e}", level="ERROR")
        raise

if __name__ == "__main__":
    args = setup_args()
    
    logger = get_logger(debug=args.verbose)
    
    logger.info("Ticker CIK Downloader started")
    dprint("Ticker CIK Downloader started", level="INFO")
    
    if args.ImmediateDownload:
        dprint("Immediate download requested", level="INFO")
        try:
            results = download_and_process_data(logger, args)
            
            dprint("Download and processing summary:", level="SUCCESS")
            for description, path, count in results:
                dprint(f"  - {description}: {count} tickers saved to {path}", level="INFO")
            
        except Exception as e:
            dprint(f"Download failed: {e}", level="ERROR")
            sys.exit(1)
    else:
        dprint("No immediate download requested, use --ImmediateDownload to download now", level="INFO")
    
    dprint("Script completed", level="SUCCESS")
    logger.info("Script completed")