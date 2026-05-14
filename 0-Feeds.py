#!/usr/bin/env python3
"""
Stock-Market Financial News Aggregator
------------------------------------------
This module collects financial news from various RSS feeds and saves them
in the Data/News directory for further analysis. It focuses on collecting data
useful for swing trading on a multi-day timeframe.
"""

import os
import json
import time
import datetime
import logging
import requests
import xmltodict
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

# Import utilities if available in your project
try:
    from Util import get_logger
    logger = get_logger(script_name="ZZnews")
except ImportError:
    # Set up basic logging if Util module is not available
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("Data/logging/ZZnews.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("Z-Feeds")

# Create News directory in Data folder if it doesn't exist
DATA_DIR = Path("Data")
NEWS_DIR = DATA_DIR / "News"
NEWS_DIR.mkdir(exist_ok=True, parents=True)



# Your RSS feed URLs - combined with additional financial sources



RSS_FEEDS = [
    # NYT
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml", "name": "NYT Economy", "category": "economy"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "name": "NYT Business", "category": "business"},
    {"url": "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml", "name": "NYT Politics", "category": "politics"},
    
    # Yahoo
    {"url": "https://www.yahoo.com/news/rss", "name": "Yahoo News", "category": "news"},
    {"url": "https://finance.yahoo.com/news/rssindex", "name": "Yahoo Finance", "category": "finance"},
    
    # LA Times
    {"url": "https://www.latimes.com/local/rss2.0.xml", "name": "LA Times Local", "category": "local"},
    
    # BBC
    {"url": "http://feeds.bbci.co.uk/news/business/rss.xml", "name": "BBC Business", "category": "business"},
    {"url": "http://feeds.bbci.co.uk/news/technology/rss.xml", "name": "BBC Technology", "category": "technology"},
    {"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "name": "BBC World", "category": "world"},
    {"url": "http://feeds.bbci.co.uk/news/economy/rss.xml", "name": "BBC Economy", "category": "economy"},
    
    # AP
    {"url": "https://feedx.net/rss/ap.xml", "name": "AP News", "category": "news"},
    
    # Euro News
    {"url": "https://www.euronews.com/rss", "name": "Euro News", "category": "news"},
    
    # Le Monde
    {"url": "https://www.lemonde.fr/en/rss/une.xml", "name": "Le Monde", "category": "news"},
    
    # Time
    {"url": "https://time.com/feed/", "name": "Time", "category": "news"},
    
    # Fox News
    {"url": "https://moxie.foxnews.com/google-publisher/latest.xml", "name": "Fox Latest", "category": "news"},
    {"url": "https://moxie.foxnews.com/google-publisher/world.xml", "name": "Fox World", "category": "world"},
    {"url": "https://moxie.foxnews.com/google-publisher/politics.xml", "name": "Fox Politics", "category": "politics"},
    {"url": "https://moxie.foxnews.com/google-publisher/science.xml", "name": "Fox Science", "category": "science"},
    {"url": "https://moxie.foxnews.com/google-publisher/health.xml", "name": "Fox Health", "category": "health"},
    {"url": "https://moxie.foxnews.com/google-publisher/tech.xml", "name": "Fox Tech", "category": "technology"},
    
    # Financial Times
    {"url": "https://www.ft.com/rss/home", "name": "Financial Times Home", "category": "business"},
    {"url": "https://www.ft.com/rss/world", "name": "Financial Times World", "category": "world"},
    {"url": "https://www.ft.com/rss/companies", "name": "Financial Times Companies", "category": "business"},
    
    # CNBC
    {"url": "https://www.cnbc.com/id/10000664/device/rss/rss.html", "name": "CNBC Finance", "category": "finance"},
    {"url": "https://www.cnbc.com/id/20910258/device/rss/rss.html", "name": "CNBC Economy", "category": "economy"},
    {"url": "https://www.cnbc.com/id/10001147/device/rss/rss.html", "name": "CNBC Business", "category": "business"},
    
    # Wall Street Journal
    {"url": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml", "name": "WSJ Markets", "category": "markets"},
    {"url": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml", "name": "WSJ Business", "category": "business"},
    
    # MarketWatch
    {"url": "http://feeds.marketwatch.com/marketwatch/topstories/", "name": "MarketWatch Top Stories", "category": "finance"},
    {"url": "http://feeds.marketwatch.com/marketwatch/marketpulse/", "name": "MarketWatch Market Pulse", "category": "markets"},
    
    # Reuters
    {"url": "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best", "name": "Reuters Business", "category": "business"},
    
    # Bloomberg (via Google News)
    {"url": "https://news.google.com/rss/search?q=site:bloomberg.com+finance&hl=en-US&gl=US&ceid=US:en", "name": "Bloomberg Finance via Google News", "category": "finance"},
    
    # The Economist
    {"url": "https://www.economist.com/finance-and-economics/rss.xml", "name": "Economist Finance", "category": "finance"},
    
    # Seeking Alpha
    {"url": "https://seekingalpha.com/feed.xml", "name": "Seeking Alpha", "category": "investing"},
    
    # Investing.com
    {"url": "https://www.investing.com/rss/news.rss", "name": "Investing.com News", "category": "investing"},
    {"url": "https://www.investing.com/rss/market_overview.rss", "name": "Investing.com Market Overview", "category": "markets"},
    
    # Barron's
    {"url": "https://www.barrons.com/feed/rssheadlines", "name": "Barron's Headlines", "category": "investing"},
    
    # Fortune
    {"url": "https://fortune.com/feed", "name": "Fortune", "category": "business"},
    
    # Forbes
    {"url": "https://www.forbes.com/business/feed/", "name": "Forbes Business", "category": "business"},
    {"url": "https://www.forbes.com/money/feed/", "name": "Forbes Money", "category": "finance"},
    
    # International Sources
    {"url": "https://economictimes.indiatimes.com/rssfeedsdefault.cms", "name": "Economic Times India", "category": "business"},
    {"url": "https://www.afr.com/rss/latest-news", "name": "Australian Financial Review", "category": "business"},



    # =============================================================================
    # CENTRAL BANKS & GOVERNMENT ECONOMIC SOURCES
    # =============================================================================
    
    # United States Federal Reserve
    {"url": "https://www.federalreserve.gov/feeds/press_monetary.xml", "name": "Fed Monetary Policy", "category": "monetary_policy"},
    {"url": "https://www.federalreserve.gov/feeds/press_bcreg.xml", "name": "Fed Banking Regulation", "category": "banking"},
    {"url": "https://www.federalreserve.gov/feeds/press_enforcement.xml", "name": "Fed Enforcement Actions", "category": "banking"},
    {"url": "https://www.federalreserve.gov/feeds/speeches_and_testimony.xml", "name": "Fed Speeches & Testimony", "category": "monetary_policy"},
    {"url": "https://www.federalreserve.gov/feeds/feds_notes.xml", "name": "Fed Notes", "category": "research"},
    {"url": "https://www.stlouisfed.org/rss", "name": "St. Louis Fed", "category": "research"},
    
    # European Central Bank
    {"url": "https://www.ecb.europa.eu/rss/press.html", "name": "ECB Press Releases", "category": "monetary_policy"},
    {"url": "https://www.ecb.europa.eu/press/blog/html/index.en.html", "name": "ECB Blog", "category": "monetary_policy"},
    {"url": "https://www.bankingsupervision.europa.eu/home/html/rss.en.html", "name": "ECB Banking Supervision", "category": "banking"},
    
    # Other Major Central Banks
    {"url": "https://www.boj.or.jp/en/rss/whatsnew.xml", "name": "Bank of Japan", "category": "monetary_policy"},
    {"url": "https://www.snb.ch/en/ifor/media/id/media_news_all", "name": "Swiss National Bank", "category": "monetary_policy"},
    {"url": "https://www.bankofcanada.ca/feed/?utility-nav-link=rss", "name": "Bank of Canada", "category": "monetary_policy"},
    {"url": "https://www.bankofengland.co.uk/rss", "name": "Bank of England", "category": "monetary_policy"},
    
    # =============================================================================
    # INTERNATIONAL ECONOMIC ORGANIZATIONS
    # =============================================================================
    
    # International Monetary Fund
    {"url": "https://www.imf.org/en/Publications/WEO/RSS", "name": "IMF World Economic Outlook", "category": "global_economy"},
    {"url": "https://www.imf.org/en/Blogs/RSS", "name": "IMF Blog", "category": "global_economy"},
    {"url": "https://www.imf.org/external/pubs/ft/survey/so/home.xml", "name": "IMF Survey", "category": "global_economy"},
    
    # World Bank
    {"url": "https://www.worldbank.org/en/research/rss", "name": "World Bank Research", "category": "development"},
    {"url": "https://openknowledge.worldbank.org/feed", "name": "World Bank Publications", "category": "research"},
    
    # OECD & Other Organizations
    {"url": "https://www.oecd.org/rss/latest-news.xml", "name": "OECD Latest News", "category": "global_economy"},
    {"url": "https://www.wto.org/library/rss/latest_news_e.xml", "name": "WTO News", "category": "trade"},
    {"url": "https://www.bis.org/doclist/cbspeeches.rss", "name": "BIS Central Bank Speeches", "category": "monetary_policy"},
    {"url": "https://www.adb.org/rss", "name": "Asian Development Bank", "category": "development"},
    
    # =============================================================================
    # ECONOMIC RESEARCH INSTITUTIONS & THINK TANKS
    # =============================================================================
    
    # Peterson Institute for International Economics
    {"url": "https://www.piie.com/rss.xml", "name": "Peterson Institute", "category": "research"},
    {"url": "https://www.piie.com/blogs/realtime-economic-issues-watch/feed", "name": "Peterson RealTime Economics", "category": "analysis"},
    
    # Centre for Economic Policy Research
    {"url": "https://cepr.org/rss/vox-content", "name": "VoxEU", "category": "research"},
    {"url": "https://cepr.org/rss/discussion-paper", "name": "CEPR Discussion Papers", "category": "research"},
    
    # Other Research Institutions
    {"url": "https://www.brookings.edu/feed/", "name": "Brookings Institution", "category": "research"},
    {"url": "https://economic-research.bnpparibas.com/RSS/en-US", "name": "BNP Paribas Economic Research", "category": "analysis"},
    
    # =============================================================================
    # ASIAN FINANCIAL MARKETS
    # =============================================================================
    
    # Asian News Sources
    {"url": "https://asia.nikkei.com/rss/feed/nar", "name": "Nikkei Asia", "category": "asia"},
    {"url": "https://www.scmp.com/rss/91/feed", "name": "South China Morning Post", "category": "asia"},
    {"url": "https://www.financeasia.com/rss/latest", "name": "FinanceAsia", "category": "finance"},
    {"url": "https://www.scmp.com/economy/china-economy/rss", "name": "SCMP China Economy", "category": "economy"},
    
    # Regional Asian Sources
    {"url": "https://www.bangkokpost.com/rss.xml", "name": "Bangkok Post", "category": "asia"},
    {"url": "https://en.vietnamplus.vn/rss/home.rss", "name": "Vietnam Plus", "category": "asia"},
    {"url": "https://www.straitstimes.com/global", "name": "Straits Times Global", "category": "asia"},
    
    # =============================================================================
    # EUROPEAN ECONOMIC SOURCES
    # =============================================================================
    
    # European Union
    {"url": "https://ec.europa.eu/eurostat/news/rss", "name": "Eurostat", "category": "statistics"},
    {"url": "https://economy-finance.ec.europa.eu/rss", "name": "EU Economy & Finance", "category": "europe"},
    {"url": "https://www.consilium.europa.eu/en/press/press-releases/rss/", "name": "EU Council Press", "category": "policy"},
    
    # European Media
    {"url": "https://www.euronews.com/rss?format=mrss", "name": "Euronews Business", "category": "europe"},
    {"url": "https://www.europeanfinancialreview.com/feed/", "name": "European Financial Review", "category": "finance"},
    
    # =============================================================================
    # LATIN AMERICAN SOURCES
    # =============================================================================
    
    {"url": "https://en.mercopress.com/rss/latin-america", "name": "MercoPress Latin America", "category": "latam"},
    {"url": "https://en.mercopress.com/rss/brazil", "name": "MercoPress Brazil", "category": "latam"},
    {"url": "https://en.mercopress.com/rss/argentina", "name": "MercoPress Argentina", "category": "latam"},
    {"url": "https://latinvex.com/feed/", "name": "LatinVex", "category": "latam"},
    {"url": "https://latinfinance.com/feed/", "name": "LatinFinance", "category": "finance"},
    
    # =============================================================================
    # COMMODITIES, FOREX & TRADING
    # =============================================================================
    
    # Commodities
    {"url": "https://www.spglobal.com/commodityinsights/en/rss", "name": "S&P Global Commodity Insights", "category": "commodities"},
    {"url": "https://www.investing.com/rss/news_285.rss", "name": "Investing.com Commodities", "category": "commodities"},
    {"url": "https://www.commodity-tv.com/api/feeds/rss/", "name": "Commodity TV", "category": "commodities"},
    
    # Forex & Trading
    {"url": "https://www.dailyforex.com/feed", "name": "DailyForex", "category": "forex"},
    {"url": "https://www.forexlive.com/feed/", "name": "ForexLive", "category": "forex"},
    {"url": "https://www.fxstreet.com/rss/news", "name": "FXStreet", "category": "forex"},
    
    # Trading Economics
    {"url": "https://tradingeconomics.com/rss/", "name": "Trading Economics", "category": "economic_data"},
    {"url": "https://tradingeconomics.com/rss/commodities", "name": "Trading Economics Commodities", "category": "commodities"},
    
    # =============================================================================
    # EMERGING MARKETS
    # =============================================================================
    
    {"url": "https://www.ft.com/emerging-markets?format=rss", "name": "FT Emerging Markets", "category": "emerging_markets"},
    {"url": "https://www.emergingmarketskeptic.substack.com/feed", "name": "Emerging Markets Skeptic", "category": "emerging_markets"},
    {"url": "https://www.em-views.com/feed", "name": "Emerging Market Views", "category": "emerging_markets"},
    {"url": "https://www.emis.com/rss", "name": "EMIS Emerging Markets", "category": "emerging_markets"},
    
    # =============================================================================
    # ADDITIONAL FINANCIAL NEWS SOURCES
    # =============================================================================
    
    # Specialized Financial Publications
    {"url": "https://www.cfi.co/feed/", "name": "Capital Finance International", "category": "finance"},
    {"url": "https://www.finance-monthly.com/feed", "name": "Finance Monthly", "category": "finance"},
    {"url": "https://www.worldfinance.com/feed", "name": "World Finance", "category": "finance"},
    {"url": "https://www.finews.com/news/english-news/rss", "name": "Finews Switzerland", "category": "finance"},
    
    # Investment & Asset Management
    {"url": "https://www.pensions-investments.com/rss/news", "name": "Pensions & Investments", "category": "investments"},
    {"url": "https://www.institutionalinvestor.com/rss/news", "name": "Institutional Investor", "category": "investments"},
    
    # Risk & Regulation
    {"url": "https://www.risk.net/rss", "name": "Risk Magazine", "category": "risk"},
    {"url": "https://www.centralbanking.com/rss", "name": "Central Banking", "category": "monetary_policy"},
    
    # =============================================================================
    # ALTERNATIVE DATA & SPECIALTY SOURCES
    # =============================================================================
    
    # Crypto (for macro correlation analysis)
    {"url": "https://invezz.com/news/cryptocurrency/feed/", "name": "Invezz Crypto", "category": "crypto"},
    {"url": "https://www.coindesk.com/arc/outboundfeeds/rss/", "name": "CoinDesk", "category": "crypto"},
    
    # ESG & Climate Finance
    {"url": "https://www.ft.com/climate-capital?format=rss", "name": "FT Climate Capital", "category": "esg"},
    {"url": "https://www.responsible-investor.com/feed/", "name": "Responsible Investor", "category": "esg"},
    
    # Business Intelligence
    {"url": "https://www.investors.com/feed/", "name": "Investor's Business Daily", "category": "business"},
    {"url": "https://www.businesswire.com/rss/home/20050606005787/en/", "name": "BusinessWire", "category": "business"},
    
    # =============================================================================
    # REGIONAL ADDITIONS
    # =============================================================================
    
    # Middle East & Africa
    {"url": "https://gulfnews.com/rss", "name": "Gulf News", "category": "mena"},
    {"url": "https://www.arabianbusiness.com/rss.xml", "name": "Arabian Business", "category": "mena"},
    
    # India & South Asia
    {"url": "https://www.livemint.com/rss/markets", "name": "Mint Markets", "category": "markets"},
    {"url": "https://www.moneycontrol.com/rss/business.xml", "name": "Moneycontrol Business", "category": "business"},
    
    # Australia & Pacific
    {"url": "https://www.afr.com/rss/latest-news", "name": "Australian Financial Review", "category": "asia_pacific"},
    {"url": "https://www.interest.co.nz/rss.xml", "name": "Interest.co.nz", "category": "asia_pacific"},
    
    # =============================================================================
    # DATA & STATISTICS SOURCES
    # =============================================================================
    
    # Economic Statistics
    {"url": "https://www.census.gov/rss/", "name": "US Census Bureau", "category": "statistics"},
    {"url": "https://www.bea.gov/rss", "name": "Bureau of Economic Analysis", "category": "statistics"},
    {"url": "https://www.bls.gov/feed/news_releases/rss.xml", "name": "Bureau of Labor Statistics", "category": "statistics"},
    
    # Industry Data
    {"url": "https://www.api.org/news-policy-and-issues/news/rss", "name": "American Petroleum Institute", "category": "energy"},
    {"url": "https://www.eia.gov/rss/press_releases.xml", "name": "EIA Press Releases", "category": "energy"},
    
    # =============================================================================
    # ADDITIONAL HIGH-VALUE SOURCES
    # =============================================================================
    
    # More Central Banking
    {"url": "https://www.centralbanking.com/rss-feeds", "name": "Central Banking Magazine", "category": "monetary_policy"},
    {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "name": "CNBC World Markets", "category": "markets"},
    {"url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=yhoo", "name": "Yahoo Finance Headlines", "category": "finance"},
    
    # International Business
    {"url": "https://www.spglobal.com/spdji/en/rss/", "name": "S&P Dow Jones Indices", "category": "indices"},
    {"url": "https://www.moodys.com/rss/research", "name": "Moody's Research", "category": "credit"},
    {"url": "https://www.fitchratings.com/rss", "name": "Fitch Ratings", "category": "credit"},
    
    # Technology & Innovation (affects macro trends)
    {"url": "https://techcrunch.com/tag/fintech/feed/", "name": "TechCrunch FinTech", "category": "fintech"},
    {"url": "https://www.americanbanker.com/feed", "name": "American Banker", "category": "banking"},
    
    # Regional Development Banks
    {"url": "https://www.ebrd.com/rss/news.rss", "name": "European Bank for Reconstruction", "category": "development"},
    {"url": "https://www.iadb.org/rss/news", "name": "Inter-American Development Bank", "category": "development"},
    {"url": "https://www.afdb.org/rss/news", "name": "African Development Bank", "category": "development"},

]





def fetch_rss(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch and process an RSS feed
    
    Args:
        source: Source information including URL
        
    Returns:
        List of extracted articles
    """
    url = source["url"]
    name = source["name"]
    articles = []
    
    try:
        logger.info(f"Fetching: {name} ({url})")
        response = requests.get(url, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"Error: Received status code {response.status_code} for {name}")
            return articles
            
        data = xmltodict.parse(response.content)
        
        # Most RSS feeds follow this structure
        if 'rss' in data and 'channel' in data['rss']:
            channel = data['rss']['channel']
            items = channel.get('item', [])
            
            # Handle case when there's only one item
            if not isinstance(items, list):
                items = [items]
                
            for item in items:
                # Skip if item is None or not a dict
                if not item or not isinstance(item, dict):
                    continue
                    
                # Extract article information
                article = {
                    "title": item.get('title', ''),
                    "description": item.get('description', ''),
                    "link": item.get('link', ''),
                    "pub_date": item.get('pubDate', ''),
                    "source_name": source['name'],
                    "source_url": source['url'],
                    "category": source['category'],
                    "fetch_time": datetime.datetime.now().isoformat()
                }
                
                # Add guid if available
                if 'guid' in item:
                    if isinstance(item['guid'], dict):
                        article["guid"] = item['guid'].get('#text', '')
                    else:
                        article["guid"] = item['guid']
                
                # Add content if available
                if 'content:encoded' in item:
                    article["content"] = item['content:encoded']
                    
                # Add the article to our list
                articles.append(article)
        
        elif 'feed' in data:  # Atom feed format
            feed = data['feed']
            entries = feed.get('entry', [])
            
            # Handle case when there's only one entry
            if not isinstance(entries, list):
                entries = [entries]
                
            for entry in entries:
                # Skip if entry is None or not a dict
                if not entry or not isinstance(entry, dict):
                    continue
                
                # Extract link
                link = entry.get('link', '')
                if isinstance(link, list):
                    for l in link:
                        if l.get('@rel') == 'alternate':
                            link = l.get('@href', '')
                            break
                elif isinstance(link, dict):
                    link = link.get('@href', '')
                
                # Extract article information
                article = {
                    "title": entry.get('title', ''),
                    "description": entry.get('summary', ''),
                    "link": link,
                    "pub_date": entry.get('updated', ''),
                    "source_name": source['name'],
                    "source_url": source['url'],
                    "category": source['category'],
                    "fetch_time": datetime.datetime.now().isoformat()
                }
                
                # Add id if available
                if 'id' in entry:
                    article["guid"] = entry['id']
                
                # Add content if available
                if 'content' in entry:
                    if isinstance(entry['content'], dict):
                        article["content"] = entry['content'].get('#text', '')
                    else:
                        article["content"] = entry['content']
                
                # Add the article to our list
                articles.append(article)
        
        logger.info(f"Found {len(articles)} articles from {name}")
        return articles
        
    except Exception as e:
        logger.error(f"Error processing {name}: {str(e)}")
        return articles

def save_articles(articles: List[Dict[str, Any]], filename: str) -> None:
    """Save articles to a JSON file
    
    Args:
        articles: List of articles to save
        filename: Filename to save the articles
    """
    filepath = NEWS_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Saved {len(articles)} articles to {filepath}")

def save_articles_parquet(articles: List[Dict[str, Any]], filepath: str) -> None:
    """Save articles to a Parquet file
    
    Args:
        articles: List of articles to save
        filepath: Path to save the Parquet file
    """
    # Convert to pandas DataFrame
    df = pd.DataFrame(articles)
    
    # Save to Parquet
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved {len(articles)} articles to {filepath}")

def clean_article_data(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and normalize article data
    
    Args:
        articles: List of articles to clean
        
    Returns:
        List of cleaned articles
    """
    cleaned_articles = []
    
    for article in articles:
        # Remove HTML tags from description using pandas
        if 'description' in article and article['description']:
            try:
                article['description'] = pd.Series(article['description']).str.replace(r'<.*?>', '', regex=True)[0]
            except:
                pass
        
        # Normalize pub_date formats
        if 'pub_date' in article and article['pub_date']:
            try:
                # Handle common date formats
                for fmt in [
                    '%a, %d %b %Y %H:%M:%S %z',  # RFC 822
                    '%a, %d %b %Y %H:%M:%S %Z',  # RFC 822 with timezone name
                    '%Y-%m-%dT%H:%M:%S%z',       # ISO 8601
                    '%Y-%m-%dT%H:%M:%SZ',        # ISO 8601 UTC
                    '%Y-%m-%d %H:%M:%S',         # Simple format
                ]:
                    try:
                        dt = datetime.datetime.strptime(article['pub_date'], fmt)
                        article['pub_date_iso'] = dt.isoformat()
                        break
                    except ValueError:
                        continue
            except:
                # If parsing fails, keep the original
                pass
        
        cleaned_articles.append(article)
    
    return cleaned_articles

def fetch_all_feeds(max_workers: int = 10) -> List[Dict[str, Any]]:
    """Fetch all RSS feeds in parallel
    
    Args:
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of all extracted articles
    """
    all_articles = []
    
    logger.info(f"Fetching {len(RSS_FEEDS)} feeds with {max_workers} parallel workers...")
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_rss, RSS_FEEDS))
    
    # Combine all results
    for articles in results:
        all_articles.extend(articles)
    
    logger.info(f"Fetched a total of {len(all_articles)} articles")
    return all_articles

def deduplicate_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate articles based on title or link
    
    Args:
        articles: List of articles to deduplicate
        
    Returns:
        List of deduplicated articles
    """
    unique_articles = []
    seen_titles = set()
    seen_links = set()
    
    for article in articles:
        title = article.get('title', '').strip()
        link = article.get('link', '').strip()
        
        # Skip if we've seen this title or link before
        if title and title in seen_titles:
            continue
        if link and link in seen_links:
            continue
        
        # Add to our unique list
        unique_articles.append(article)
        
        # Remember we've seen this title and link
        if title:
            seen_titles.add(title)
        if link:
            seen_links.add(link)
    
    logger.info(f"Removed {len(articles) - len(unique_articles)} duplicate articles")
    return unique_articles

def main():
    """Main function to run the RSS aggregator"""
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    
    logger.info("Starting RSS Feed Aggregator...")
    logger.info(f"Found {len(RSS_FEEDS)} RSS feeds to process")
    
    # Fetch all feeds in parallel
    articles = fetch_all_feeds(max_workers=15)
    
    # Clean the article data
    articles = clean_article_data(articles)
    
    # Deduplicate articles
    articles = deduplicate_articles(articles)
    
    # Save the articles to JSON
    json_filename = f"financial_news_{timestamp}.json"
    save_articles(articles, json_filename)
    
    # Save to Parquet (integrates better with your system)
    parquet_path = NEWS_DIR / f"financial_news_{timestamp}.parquet"
    save_articles_parquet(articles, parquet_path)
    
    # Also maintain a latest copy for easy access
    latest_parquet_path = NEWS_DIR / "latest_financial_news.parquet"
    save_articles_parquet(articles, latest_parquet_path)
    
    # Print summary
    elapsed_time = time.time() - start_time
    logger.info(f"Done! Completed in {elapsed_time:.2f} seconds")
    logger.info(f"- Processed {len(RSS_FEEDS)} RSS feeds")
    logger.info(f"- Fetched {len(articles)} unique articles")
    logger.info(f"- Saved to {NEWS_DIR / json_filename}")
    logger.info(f"- Saved to {parquet_path}")
    logger.info(f"- Updated {latest_parquet_path}")



if __name__ == "__main__":
    main()

