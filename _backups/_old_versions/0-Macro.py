#!/usr/bin/env python3
"""
Enhanced SEC EDGAR Financial Data Processor
-------------------------------------------
A complete solution for processing large volumes of SEC filing data,
extracting financial metrics, and deriving missing values.

This script is designed to handle 20GB+ of JSON data efficiently with
parallel processing, smart tag mapping, and metric derivation.
"""

import os
import json
import re
import gc
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import difflib
##import all util functions to set up the logging 
from Util import get_logger



logger = get_logger(script_name="Z-Macro")

# Define paths - adjust these to match your environment
BASE_DIR = Path("Data")
SEC_DIR = BASE_DIR / "SEC_Data"
FILINGS_DIR = BASE_DIR / "Filings"
OUTPUT_DIR = SEC_DIR / "Enhanced"
METRICS_DIR = SEC_DIR / "Metrics"

# Create output directories if they don't exist
for directory in [OUTPUT_DIR, METRICS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)





# Primary and fallback tag mappings for financial metrics
METRIC_TAG_MAPPING = {
    # Revenue - Add more fallbacks and ensure revenue is properly captured
    "Revenue": {
        "primary": [
            "us-gaap:Revenues", 
            "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap:SalesRevenueNet",
            "ifrs-full:Revenue"
        ],
        "fallback": [
            "us-gaap:SalesRevenueGoodsNet",
            "us-gaap:RegulatedAndUnregulatedOperatingRevenue",
            "us-gaap:OilAndGasRevenue",
            "us-gaap:RevenueFromSaleOfCrudeOil",
            "us-gaap:RevenueFromSaleOfNaturalGas"
        ]
    },
    
    # Net Income - Consider additional tags and fallbacks
    "Net Income": {
        "primary": [
            "us-gaap:NetIncomeLoss",
            "us-gaap:ProfitLoss"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperations",
            "us-gaap:IncomeLossAttributableToParent",
            "us-gaap:ComprehensiveIncomeNetOfTax",
            "ifrs-full:ProfitLoss"
        ]
    },
    
    # Operating Income - More comprehensive mapping
    "Operating Income": {
        "primary": [
            "us-gaap:OperatingIncomeLoss"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
            "ifrs-full:ProfitLossFromOperatingActivities"
        ]
    },
    
    # Income Before Tax - Additional tags
    "Income Before Tax": {
        "primary": [
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
        ],
        "fallback": [
            "us-gaap:IncomeLossBeforeIncomeTaxes",
            "ifrs-full:ProfitLossBeforeTax"
        ]
    },
    
    # COGS - Include more variants
    "COGS": {
        "primary": [
            "us-gaap:CostOfGoodsAndServicesSold",
            "us-gaap:CostOfRevenue"
        ],
        "fallback": [
            "us-gaap:CostOfServices",
            "us-gaap:CostOfGoodsSold",
            "ifrs-full:CostOfSales"
        ]
    },
    
    # Gross Profit - Include IFRS versions
    "Gross Profit": {
        "primary": [
            "us-gaap:GrossProfit"
        ],
        "fallback": [
            "ifrs-full:GrossProfit"
        ]
    },
    
    # EPS metrics - More variants
    "EPS (Basic)": {
        "primary": [
            "us-gaap:EarningsPerShareBasic",
            "us-gaap:EarningsPerShareBasicAndDiluted"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperationsPerBasicShare",
            "ifrs-full:BasicEarningsLossPerShare"
        ]
    },
    
    "EPS (Diluted)": {
        "primary": [
            "us-gaap:EarningsPerShareDiluted",
            "us-gaap:EarningsPerShareBasicAndDiluted"
        ],
        "fallback": [
            "us-gaap:IncomeLossFromContinuingOperationsPerDilutedShare",
            "ifrs-full:DilutedEarningsLossPerShare"
        ]
    },
    
    # Share count metrics - More variants
    "Shares Outstanding (Basic)": {
        "primary": [
            "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic"
        ],
        "fallback": [
            "us-gaap:CommonStockSharesOutstanding",
            "dei:EntityCommonStockSharesOutstanding"
        ]
    },
    
    "Shares Outstanding (Diluted)": {
        "primary": [
            "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
            "us-gaap:WeightedAverageNumberOfSharesOutstandingDiluted"
        ],
        "fallback": [
            "us-gaap:CommonStockSharesOutstanding"
        ]
    },
    
    # R&D Expense - Include IFRS version
    "R&D Expense": {
        "primary": [
            "us-gaap:ResearchAndDevelopmentExpense"
        ],
        "fallback": [
            "ifrs-full:ResearchAndDevelopmentExpense"
        ]
    },
    
    # Interest Expense - Include IFRS version
    "Interest Expense": {
        "primary": [
            "us-gaap:InterestExpense"
        ],
        "fallback": [
            "us-gaap:InterestAndDebtExpense",
            "ifrs-full:FinanceCosts"
        ]
    },
    
    # Cash Flow metrics - More comprehensive mappings
    "Operating Cash Flow": {
        "primary": [
            "us-gaap:NetCashProvidedByUsedInOperatingActivities"
        ],
        "fallback": [
            "ifrs-full:CashFlowsFromUsedInOperatingActivities",
            "ifrs-full:CashFlowsFromUsedInOperations"
        ]
    },
    
    "Investing Cash Flow": {
        "primary": [
            "us-gaap:NetCashProvidedByUsedInInvestingActivities"
        ],
        "fallback": [
            "ifrs-full:CashFlowsFromUsedInInvestingActivities"
        ]
    },
    
    "Financing Cash Flow": {
        "primary": [
            "us-gaap:NetCashProvidedByUsedInFinancingActivities"
        ],
        "fallback": [
            "ifrs-full:CashFlowsFromUsedInFinancingActivities"
        ]
    },
    
    "Net Change in Cash": {
        "primary": [
            "us-gaap:CashAndCashEquivalentsPeriodIncreaseDecrease"
        ],
        "fallback": [
            "ifrs-full:IncreaseDecreaseInCashAndCashEquivalents"
        ]
    },
    
    # Share-based Compensation - More variants
    "Share-based Compensation": {
        "primary": [
            "us-gaap:ShareBasedCompensation",
            "us-gaap:AllocatedShareBasedCompensationExpense"
        ],
        "fallback": [
            "us-gaap:StockBasedCompensation",
            "ifrs-full:SharebasedPaymentArrangementsBySharebasedPaymentArrangementGroupExtensions"
        ]
    },
    
    # Depreciation & Amortization - More comprehensive
    "Depreciation & Amortization": {
        "primary": [
            "us-gaap:DepreciationDepletionAndAmortization",
            "us-gaap:DepreciationAndAmortization"
        ],
        "fallback": [
            "us-gaap:Depreciation",
            "us-gaap:AmortizationOfIntangibleAssets",
            "ifrs-full:DepreciationAndAmortisationExpense"
        ]
    }
}




# Ratio metrics to calculate
RATIO_METRICS = {
    "GrossMargin": {
        "formula": lambda metrics: metrics.get("Gross Profit") / metrics.get("Revenue") if metrics.get("Revenue", 0) != 0 else None,
        "required_metrics": ["Gross Profit", "Revenue"]
    },
    "OperatingMargin": {
        "formula": lambda metrics: metrics.get("Operating Income") / metrics.get("Revenue") if metrics.get("Revenue", 0) != 0 else None,
        "required_metrics": ["Operating Income", "Revenue"]
    },
    "NetMargin": {
        "formula": lambda metrics: metrics.get("Net Income") / metrics.get("Revenue") if metrics.get("Revenue", 0) != 0 else None,
        "required_metrics": ["Net Income", "Revenue"]
    },
    "ROE": {
        "formula": lambda metrics: metrics.get("Net Income") / metrics.get("StockholdersEquity") if metrics.get("StockholdersEquity", 0) != 0 else None,
        "required_metrics": ["Net Income", "StockholdersEquity"]
    },
    "InterestCoverage": {
        "formula": lambda metrics: metrics.get("Operating Income") / metrics.get("Interest Expense") if metrics.get("Interest Expense", 0) != 0 else None,
        "required_metrics": ["Operating Income", "Interest Expense"]
    }
}

# Highly correlated metrics for derivation
METRIC_CORRELATIONS = {
    "Revenue": {
        "COGS": 0.98,
        "Gross Profit": 0.99,
        "Operating Income": 0.95,
        "Net Income": 0.90
    },
    "Gross Profit": {
        "Revenue": 0.99,
        "COGS": 0.95,
        "Operating Income": 0.97
    },
    "COGS": {
        "Revenue": 0.98,
        "Gross Profit": 0.95
    },
    "Operating Income": {
        "Income Before Tax": 1.00,
        "Net Income": 0.97,
        "Gross Profit": 0.97,
        "Revenue": 0.95
    },
    "Income Before Tax": {
        "Operating Income": 1.00,
        "Net Income": 0.99
    },
    "Net Income": {
        "Income Before Tax": 0.99,
        "Operating Income": 0.97,
        "EPS (Basic)": 0.99,
        "EPS (Diluted)": 0.99
    },
    "EPS (Basic)": {
        "EPS (Diluted)": 1.00,
        "Net Income": 0.99
    },
    "EPS (Diluted)": {
        "EPS (Basic)": 1.00,
        "Net Income": 0.99
    },
    "Shares Outstanding (Diluted)": {
        "Shares Outstanding (Basic)": 0.99
    },
    "Shares Outstanding (Basic)": {
        "Shares Outstanding (Diluted)": 0.99
    },
    "Net Change in Cash": {
        "Operating Cash Flow": 0.80,
        "Investing Cash Flow": 0.70,
        "Financing Cash Flow": 0.70
    },
    "Interest Expense": {
        "Operating Income": 0.70,
        "Income Before Tax": 0.75
    },
    "Depreciation & Amortization": {
        "Operating Income": 0.80,
        "COGS": 0.70
    }
}



INDUSTRY_TAG_MAPPINGS = {
    # Finance/Banking industry (SIC codes 6000-6799)
    "FINANCIAL": {
        "Revenue": {
            "primary": [
                "us-gaap:InterestAndDividendIncomeOperating",
                "us-gaap:NoninterestIncome",
                "us-gaap:InterestIncome",
                "us-gaap:RevenuesNetOfInterestExpense",
                "us-gaap:InterestAndFeeIncomeLoans"
            ],
            "fallback": [
                "us-gaap:RevenuesNetOfInterestExpense",
                "us-gaap:TotalInterestAndDividendIncome"
            ]
        },
        "Net Income": {
            "primary": [
                "us-gaap:NetIncomeLoss",
                "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic"
            ],
            "fallback": [
                "us-gaap:ProfitLoss",
                "us-gaap:ComprehensiveIncomeNetOfTax"
            ]
        },
        "Operating Income": {
            "primary": [
                "us-gaap:IncomeFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxes"
            ],
            "fallback": [
                "us-gaap:OperatingIncomeLoss"
            ]
        }
    },
    
    # Oil & Gas industry (SIC codes 1300-1399, 2900-2999)
    "OIL_GAS": {
        "Revenue": {
            "primary": [
                "us-gaap:OilAndGasRevenue",
                "us-gaap:RevenueFromSaleOfCrudeOil",
                "us-gaap:RevenueFromSaleOfNaturalGas"
            ],
            "fallback": [
                "us-gaap:Revenues",
                "us-gaap:SalesRevenueNet"
            ]
        },
        "Operating Expenses": {
            "primary": [
                "us-gaap:LeaseOperatingExpenses",
                "us-gaap:OilAndGasOperatingExpenses"
            ],
            "fallback": [
                "us-gaap:OperatingExpenses"
            ]
        }
    },
    
    # Technology industry (SIC codes 7370-7379, 3570-3579)
    "TECHNOLOGY": {
        "Revenue": {
            "primary": [
                "us-gaap:Revenues",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "us-gaap:SalesRevenueServicesNet"
            ],
            "fallback": [
                "us-gaap:SalesRevenueNet"
            ]
        },
        "R&D Expense": {
            "primary": [
                "us-gaap:ResearchAndDevelopmentExpense",
                "us-gaap:TechnologyExpense"
            ],
            "fallback": [
                "us-gaap:EngineeeringExpense"
            ]
        }
    },
    
    # Retail industry (SIC codes 5200-5999)
    "RETAIL": {
        "Revenue": {
            "primary": [
                "us-gaap:SalesRevenueNet",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
            ],
            "fallback": [
                "us-gaap:Revenues"
            ]
        },
        "COGS": {
            "primary": [
                "us-gaap:CostOfGoodsSold",
                "us-gaap:CostOfGoodsAndServicesSold"
            ],
            "fallback": [
                "us-gaap:CostOfRevenue"
            ]
        },
        "Same Store Sales": {
            "primary": [
                "us-gaap:ComparableStoresSalesGrowthDecline"
            ],
            "fallback": []
        }
    },
    
    # Healthcare industry (SIC codes 8000-8099)
    "HEALTHCARE": {
        "Revenue": {
            "primary": [
                "us-gaap:PatientServiceRevenue",
                "us-gaap:HealthCareOrganizationRevenue"
            ],
            "fallback": [
                "us-gaap:Revenues",
                "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
            ]
        },
        "R&D Expense": {
            "primary": [
                "us-gaap:ResearchAndDevelopmentExpensePharma",
                "us-gaap:ResearchAndDevelopmentExpense"
            ],
            "fallback": []
        }
    }
}

# SIC code to industry type mapping
SIC_TO_INDUSTRY = {
    # Financial industry
    **{str(sic): "FINANCIAL" for sic in range(6000, 6800)},
    
    # Oil & Gas industry
    **{str(sic): "OIL_GAS" for sic in list(range(1300, 1400)) + list(range(2900, 3000))},
    
    # Technology industry
    **{str(sic): "TECHNOLOGY" for sic in list(range(7370, 7380)) + list(range(3570, 3580)) + list(range(3600, 3700))},
    
    # Retail industry
    **{str(sic): "RETAIL" for sic in range(5200, 6000)},
    
    # Healthcare industry - FIX HERE
    **{str(sic): "HEALTHCARE" for sic in list(range(8000, 8100))},
    **{str(sic): "HEALTHCARE" for sic in list(range(2830, 2837))}
}



#ALT SIC code to industry type mapping
SIC_TO_INDUSTRY_ALT = {
    # Financial industry
    **{str(sic): "FINANCIAL" for sic in range(6000, 6800)},
    
    # Oil & Gas industry
    **{str(sic): "OIL_GAS" for sic in list(range(1300, 1400)) + list(range(2900, 3000))},
    
    # Technology industry
    **{str(sic): "TECHNOLOGY" for sic in list(range(7370, 7380)) + list(range(3570, 3580)) + list(range(3600, 3700))},
    
    # Retail industry
    **{str(sic): "RETAIL" for sic in range(5200, 6000)},
    
    # Healthcare industry - Alternative approach
    **{str(sic): "HEALTHCARE" for sic in [*range(8000, 8100), *range(2830, 2837)]}
}





def derive_missing_metrics(metrics):
    """
    Enhanced function to derive missing metrics using correlations and financial relationships.
    """
    derived_metrics = {}
    derivation_log = []
    
    # Track which metrics were derived in this iteration to avoid circular logic
    derived_this_round = set()
    
    # Multiple passes to handle dependent derivations
    for _ in range(3):  # Limited to 3 passes to prevent infinite loops
        derived_this_round.clear()
        
        # Try standard correlation-based derivation first
        for target_metric, correlations in METRIC_CORRELATIONS.items():
            # Skip if target already exists
            if target_metric in metrics:
                continue
                
            # Try each correlated metric
            for source_metric, correlation in correlations.items():
                if source_metric in metrics and correlation > 0.90:  # Higher threshold for reliability
                    # Use a direct value transfer for highly correlated metrics
                    derived_metrics[target_metric] = {
                        "value": metrics[source_metric],
                        "source": f"correlation with {source_metric}",
                        "correlation": correlation,
                        "confidence": correlation
                    }
                    derived_this_round.add(target_metric)
                    derivation_log.append(f"Derived {target_metric} from {source_metric} (correlation: {correlation})")
                    break
        
        # Apply financial relationship derivations
        
        # Revenue = Gross Profit + COGS
        if "Revenue" not in metrics and "Gross Profit" in metrics and "COGS" in metrics:
            value = metrics["Gross Profit"] + metrics["COGS"]
            derived_metrics["Revenue"] = {
                "value": value,
                "source": "calculated from Gross Profit + COGS",
                "correlation": 0.98,
                "confidence": 0.98
            }
            derived_this_round.add("Revenue")
            derivation_log.append("Derived Revenue from Gross Profit + COGS")
            
        # Gross Profit = Revenue - COGS
        if "Gross Profit" not in metrics and "Revenue" in metrics and "COGS" in metrics:
            value = metrics["Revenue"] - metrics["COGS"]
            derived_metrics["Gross Profit"] = {
                "value": value,
                "source": "calculated from Revenue - COGS",
                "correlation": 0.98,
                "confidence": 0.98
            }
            derived_this_round.add("Gross Profit")
            derivation_log.append("Derived Gross Profit from Revenue - COGS")
            
        # COGS = Revenue - Gross Profit
        if "COGS" not in metrics and "Revenue" in metrics and "Gross Profit" in metrics:
            value = metrics["Revenue"] - metrics["Gross Profit"]
            derived_metrics["COGS"] = {
                "value": value,
                "source": "calculated from Revenue - Gross Profit",
                "correlation": 0.98,
                "confidence": 0.98
            }
            derived_this_round.add("COGS")
            derivation_log.append("Derived COGS from Revenue - Gross Profit")
        
        # Net Change in Cash = Operating Cash Flow + Investing Cash Flow + Financing Cash Flow
        if "Net Change in Cash" not in metrics and all(m in metrics for m in ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]):
            value = metrics["Operating Cash Flow"] + metrics["Investing Cash Flow"] + metrics["Financing Cash Flow"]
            derived_metrics["Net Change in Cash"] = {
                "value": value,
                "source": "calculated from OCF + ICF + FCF",
                "correlation": 0.95,
                "confidence": 0.95
            }
            derived_this_round.add("Net Change in Cash")
            derivation_log.append("Derived Net Change in Cash from cash flow components")
            
        # EPS derivations using shares
        if "EPS (Basic)" not in metrics and "Net Income" in metrics and "Shares Outstanding (Basic)" in metrics and metrics["Shares Outstanding (Basic)"] > 0:
            value = metrics["Net Income"] / metrics["Shares Outstanding (Basic)"]
            derived_metrics["EPS (Basic)"] = {
                "value": value,
                "source": "calculated from Net Income / Shares Outstanding (Basic)",
                "correlation": 0.99,
                "confidence": 0.99
            }
            derived_this_round.add("EPS (Basic)")
            derivation_log.append("Derived EPS (Basic) from Net Income / Shares Outstanding (Basic)")
            
        if "EPS (Diluted)" not in metrics and "Net Income" in metrics and "Shares Outstanding (Diluted)" in metrics and metrics["Shares Outstanding (Diluted)"] > 0:
            value = metrics["Net Income"] / metrics["Shares Outstanding (Diluted)"]
            derived_metrics["EPS (Diluted)"] = {
                "value": value,
                "source": "calculated from Net Income / Shares Outstanding (Diluted)",
                "correlation": 0.99,
                "confidence": 0.99
            }
            derived_this_round.add("EPS (Diluted)")
            derivation_log.append("Derived EPS (Diluted) from Net Income / Shares Outstanding (Diluted)")
            
        # EPS can be derived from each other if one exists
        if "EPS (Basic)" not in metrics and "EPS (Diluted)" in metrics:
            derived_metrics["EPS (Basic)"] = {
                "value": metrics["EPS (Diluted)"],
                "source": "estimated from EPS (Diluted)",
                "correlation": 0.99,
                "confidence": 0.95
            }
            derived_this_round.add("EPS (Basic)")
            derivation_log.append("Derived EPS (Basic) from EPS (Diluted)")
            
        if "EPS (Diluted)" not in metrics and "EPS (Basic)" in metrics:
            # Diluted is typically slightly lower than Basic
            value = metrics["EPS (Basic)"] * 0.98
            derived_metrics["EPS (Diluted)"] = {
                "value": value,
                "source": "estimated from EPS (Basic)",
                "correlation": 0.99,
                "confidence": 0.95
            }
            derived_this_round.add("EPS (Diluted)")
            derivation_log.append("Derived EPS (Diluted) from EPS (Basic)")
        
        # Update metrics with derived values for next iteration
        for metric, details in derived_metrics.items():
            if metric in derived_this_round:
                metrics[metric] = details["value"]
        
        # If nothing was derived this round, break early
        if not derived_this_round:
            break
    
    # Add additional metadata
    for metric, details in derived_metrics.items():
        metrics[metric] = details["value"]
    
    return metrics, derived_metrics, derivation_log






class SECDataProcessor:
    """
    Enhanced SEC EDGAR data processor that implements sophisticated strategies
    for extracting financial metrics from SEC filings.
    """
    
    def __init__(self, base_dir=BASE_DIR, filings_dir=FILINGS_DIR, output_dir=OUTPUT_DIR, max_workers=32, logger=logger):
        self.base_dir = Path(base_dir)
        self.filings_dir = Path(filings_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_workers = max_workers
        self.logger = logger  # Use the passed logger
        self.cik_to_ticker = {}
        self.ticker_to_cik = {}
        self.cik_to_name = {}
        self.cik_to_industry = {}
        self.company_data = {}  # Store processed company data
        
        # Track statistics
        self.stats = {
            "total_files": 0,
            "processed_files": 0,
            "error_files": 0,
            "metrics_found": defaultdict(int),
            "companies_processed": set(),
            "tag_usage": defaultdict(Counter)
        }
    
    def load_ticker_mapping(self, ticker_file):
        """Load CIK to ticker mappings from a parquet file"""
        try:
            logger.info(f"Loading ticker mapping from {ticker_file}")
            ticker_df = pd.read_parquet(ticker_file)
            
            for _, row in ticker_df.iterrows():
                cik = str(row['cik']).zfill(10)
                ticker = row['ticker']
                name = row.get('name', '')
                
                self.cik_to_ticker[cik] = ticker
                self.ticker_to_cik[ticker] = cik
                
                if name:
                    self.cik_to_name[cik] = name
            
            logger.info(f"Loaded {len(self.cik_to_ticker)} ticker-CIK mappings")
        except Exception as e:
            logger.error(f"Error loading ticker mapping: {e}")
            raise
    

    def filter_valid_filing_files(self, max_files=None):
        """
        Filter out empty or invalid filing files before processing.
        Returns a list of valid filing files for processing.

        Args:
            max_files: Optional limit on the number of files to process

        Returns:
            List of valid filing file paths
        """
        logger.info("Filtering valid filing files")

        valid_files = []
        total_files = 0
        empty_files = 0

        # Process all files in the filings directory
        for file_path in self.filings_dir.glob("CIK*.json"):
            total_files += 1

            # Quick size check first - if it's less than 100 bytes, it's likely empty
            if os.path.getsize(file_path) < 100:
                empty_files += 1
                continue

            try:
                # Open and check for minimal content
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample = f.read(1000)  # Just read the first 1KB

                    # Skip if it appears to be an empty file
                    if '"facts":{}' in sample or '"facts": {}' in sample:
                        empty_files += 1
                        continue

                    # Extract CIK from filename to check if we should process this file
                    cik_match = re.search(r'CIK(\d+)\.json', str(file_path))
                    if not cik_match:
                        continue

                    cik = cik_match.group(1).zfill(10)
                    if cik not in self.cik_to_ticker:
                        continue

                    # If it passed all checks, add to valid files
                    valid_files.append(file_path)

                    # Stop if we've reached the maximum number of files
                    if max_files and len(valid_files) >= max_files:
                        break

            except Exception as e:
                logger.debug(f"Error checking file {file_path}: {e}")

        logger.info(f"Found {total_files} total files, {empty_files} empty/invalid files")
        logger.info(f"Filtered to {len(valid_files)} valid files for processing")

        return valid_files










    def extract_metadata(self, file_content, file_path):
        """
        Extract key metadata from the filing content to determine processing strategy.
        Enhanced with more flexible pattern matching.
        """
        metadata = {}

        # Extract CIK - more flexible pattern matching
        cik_patterns = [
            r'"cik"\s*:\s*"?(\d+)"?',
            r'"CIK"\s*:\s*"?(\d+)"?',
            r'CIK=(\d+)',
            r'CIK-(\d+)'
        ]

        for pattern in cik_patterns:
            cik_match = re.search(pattern, file_content)
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                metadata['cik'] = cik
                metadata['ticker'] = self.cik_to_ticker.get(cik, "")
                metadata['name'] = self.cik_to_name.get(cik, "")
                break
            
        if 'cik' not in metadata:
            # Try to extract from filename
            cik_match = re.search(r'CIK(\d+)\.json', str(file_path))
            if cik_match:
                cik = cik_match.group(1).zfill(10)
                metadata['cik'] = cik
                metadata['ticker'] = self.cik_to_ticker.get(cik, "")
                metadata['name'] = self.cik_to_name.get(cik, "")

        # Extract SIC code - more flexible pattern matching
        sic_patterns = [
            r'"sic"\s*:\s*"?(\d+)"?', 
            r'"SIC"\s*:\s*"?(\d+)"?',
            r'SIC=(\d+)',
            r'SIC-(\d+)'
        ]

        for pattern in sic_patterns:
            sic_match = re.search(pattern, file_content)
            if sic_match:
                sic = sic_match.group(1)
                metadata['sic'] = sic
                metadata['industry'] = SIC_TO_INDUSTRY.get(sic, "OTHER")
                break
            
        # Extract form type - more flexible pattern matching
        form_patterns = [
            r'"form"\s*:\s*"([^"]+)"',
            r'"FORM"\s*:\s*"([^"]+)"',
            r'FORM=([^\s&]+)',
            r'FORM-([^\s&]+)'
        ]

        for pattern in form_patterns:
            form_match = re.search(pattern, file_content)
            if form_match:
                metadata['form'] = form_match.group(1)

                # Determine if 10-K or 10-Q
                form = form_match.group(1).upper()
                if "10-K" in form:
                    metadata['filing_type'] = "10-K"
                elif "10-Q" in form:
                    metadata['filing_type'] = "10-Q"
                else:
                    metadata['filing_type'] = form
                break
            
        # Extract period - more flexible pattern matching
        period_patterns = [
            r'"period"\s*:\s*"?(\d{8})"?',
            r'"PERIOD"\s*:\s*"?(\d{8})"?',
            r'PERIOD=(\d{8})',
            r'PERIOD-(\d{8})'
        ]

        for pattern in period_patterns:
            period_match = re.search(pattern, file_content)
            if period_match:
                period = period_match.group(1)
                metadata['period'] = period

                # Convert YYYYMMDD to YYYY-MM-DD format
                try:
                    year = period[:4]
                    month = period[4:6]
                    day = period[6:8]
                    metadata['period_formatted'] = f"{year}-{month}-{day}"
                except:
                    metadata['period_formatted'] = period
                break
            
        # If we still don't have a period, look for other date formats
        if 'period' not in metadata:
            # Try to find a date in YYYY-MM-DD format
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', file_content)
            if date_match:
                date_str = date_match.group(1)
                metadata['period_formatted'] = date_str
                # Convert to YYYYMMDD format
                try:
                    metadata['period'] = date_str.replace('-', '')
                except:
                    pass
                
        # Extract fiscal year
        fy_patterns = [
            r'"fy"\s*:\s*"?(\d{4})"?',
            r'"FY"\s*:\s*"?(\d{4})"?',
            r'FY=(\d{4})',
            r'FY-(\d{4})'
        ]

        for pattern in fy_patterns:
            fy_match = re.search(pattern, file_content)
            if fy_match:
                metadata['fiscal_year'] = fy_match.group(1)
                break
            
        # If no fiscal year found, try to extract from period
        if 'fiscal_year' not in metadata and 'period' in metadata:
            try:
                metadata['fiscal_year'] = metadata['period'][:4]
            except:
                pass
            
        # Extract fiscal period focus (Q1, Q2, Q3, Q4, FY)
        fp_patterns = [
            r'"fp"\s*:\s*"([^"]+)"',
            r'"FP"\s*:\s*"([^"]+)"',
            r'FP=([^\s&]+)',
            r'FP-([^\s&]+)'
        ]

        for pattern in fp_patterns:
            fp_match = re.search(pattern, file_content)
            if fp_match:
                metadata['fiscal_period'] = fp_match.group(1)
                break
            
        # If fiscal period not found, try to determine from form type
        if 'fiscal_period' not in metadata and 'form' in metadata:
            form = metadata['form'].upper()
            if '10-K' in form:
                metadata['fiscal_period'] = 'FY'
            elif '10-Q' in form:
                # Can't determine which quarter without more info
                metadata['fiscal_period'] = 'QX'

        # Generate period_id if possible
        if 'fiscal_year' in metadata:
            if metadata.get('fiscal_period') in ('Q1', 'Q2', 'Q3', 'Q4', 'QX'):
                metadata['period_id'] = f"{metadata['fiscal_year']}-{metadata['fiscal_period']}"
            elif metadata.get('fiscal_period') == 'FY':
                metadata['period_id'] = metadata['fiscal_year']
        elif 'period' in metadata:
            # Use the period as a fallback for period_id
            metadata['period_id'] = metadata['period']

        # Special case: If we have both period and filing_type but no period_id
        if 'period_id' not in metadata and 'period' in metadata and 'filing_type' in metadata:
            metadata['period_id'] = metadata['period']

        # If we've gotten this far and still don't have filing_type and period, use defaults
        # This is a fallback to ensure processing continues
        if not metadata.get('filing_type') and metadata.get('form'):
            # Default to the form value if a specific type couldn't be determined
            metadata['filing_type'] = metadata['form']

        if not metadata.get('period') and metadata.get('period_formatted'):
            # Use the formatted period if the raw period isn't available
            metadata['period'] = metadata['period_formatted'].replace('-', '')

        return metadata
    
    def determine_tag_mapping(self, metadata):
        """
        Determine the appropriate tag mapping based on metadata.
        """
        tag_mapping = {}
        
        # Start with standard tag mappings
        for metric, mapping in METRIC_TAG_MAPPING.items():
            tag_mapping[metric] = {
                "primary": mapping["primary"].copy(),
                "fallback": mapping["fallback"].copy()
            }
        
        # Apply industry-specific mappings if available
        industry = metadata.get('industry')
        if industry and industry in INDUSTRY_TAG_MAPPINGS:
            industry_mappings = INDUSTRY_TAG_MAPPINGS[industry]
            
            for metric, mapping in industry_mappings.items():
                if metric in tag_mapping:
                    # Add industry-specific primary tags to the front of the list
                    tag_mapping[metric]["primary"] = mapping["primary"] + tag_mapping[metric]["primary"]
                    tag_mapping[metric]["fallback"] = mapping["fallback"] + tag_mapping[metric]["fallback"]
                else:
                    tag_mapping[metric] = mapping
        
        return tag_mapping
    
    def find_tag_recursive(self, data, tag_pattern, results=None):
        """
        Recursively search for tags matching a pattern in nested data structures.
        """
        if results is None:
            results = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                if re.match(tag_pattern, key):
                    results.append((key, value))
                if isinstance(value, (dict, list)):
                    self.find_tag_recursive(value, tag_pattern, results)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self.find_tag_recursive(item, tag_pattern, results)
        
        return results
    
    def extract_value_for_tag(self, json_data, tag, fallback_tags=None):
        """
        Extract value for a specific tag with fallback options.
        """
        # Try exact match first
        if tag in json_data:
            return json_data[tag], tag
        
        # Try to find it recursively
        tag_results = self.find_tag_recursive(json_data, f"^{re.escape(tag)}$")
        if tag_results:
            return tag_results[0][1], tag_results[0][0]
        
        # Try fallback tags if provided
        if fallback_tags:
            for fallback_tag in fallback_tags:
                if fallback_tag in json_data:
                    return json_data[fallback_tag], fallback_tag
                
                # Try to find fallback recursively
                fallback_results = self.find_tag_recursive(json_data, f"^{re.escape(fallback_tag)}$")
                if fallback_results:
                    return fallback_results[0][1], fallback_results[0][0]
        
        # Try semantic matching as last resort
        potential_matches = self.find_semantically_similar_tags(json_data, tag, 0.85)
        if potential_matches:
            best_match_tag = potential_matches[0][0]
            return json_data.get(best_match_tag), best_match_tag
        
        return None, None
    
    def find_semantically_similar_tags(self, json_data, target_tag, similarity_threshold=0.7):
        """
        Find tags in the data that are semantically similar to the target tag.
        """
        all_tags = []
        for key in json_data.keys():
            if isinstance(key, str):
                all_tags.append(key)
        
        # Also search for nested tags
        for tag_tuple in self.find_tag_recursive(json_data, r".*"):
            if isinstance(tag_tuple[0], str):
                all_tags.append(tag_tuple[0])
        
        potential_matches = []
        target_tag_simple = target_tag.split(':')[-1].lower()
        
        for tag in all_tags:
            tag_simple = tag.split(':')[-1].lower()
            similarity = difflib.SequenceMatcher(None, target_tag_simple, tag_simple).ratio()
            if similarity >= similarity_threshold:
                potential_matches.append((tag, similarity))
        
        return sorted(potential_matches, key=lambda x: x[1], reverse=True)
    
    def extract_value_from_fact(self, fact, filing_type):
        """
        Extract the appropriate value from a fact structure.
        """
        if not fact:
            return None
        
        # Handle various fact structures
        if isinstance(fact, dict):
            # Direct value access
            if 'val' in fact:
                return fact['val']
            
            # Handle units section
            if 'units' in fact:
                units = fact['units']
                
                # Try USD for monetary values
                if 'USD' in units and units['USD']:
                    values = units['USD']
                    for value_obj in values:
                        # Filter for the right form type
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    # If no match by form, take the first value
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try shares for share counts
                if 'shares' in units and units['shares']:
                    values = units['shares']
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try USD/shares for per-share values
                if 'USD/shares' in units and units['USD/shares']:
                    values = units['USD/shares']
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try pure/xbrl for ratios
                if 'pure' in units and units['pure']:
                    values = units['pure']
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    if values and 'val' in values[0]:
                        return values[0]['val']
        
        # Handle simple value
        return fact
    
    def extract_metrics(self, json_data, metadata, tag_mapping):
        """
        Extract financial metrics from JSON data using the provided tag mapping.
        """
        extracted_metrics = {}
        tag_sources = {}
        
        # Process US GAAP facts first
        facts = json_data.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Extract each metric using appropriate tags
        for metric, tags in tag_mapping.items():
            primary_tags = tags['primary']
            fallback_tags = tags['fallback']
            
            for tag in primary_tags:
                tag_base = tag.split(':')[-1]  # Remove namespace prefix
                
                if tag_base in us_gaap:
                    value = self.extract_value_from_fact(us_gaap[tag_base], metadata.get('filing_type', ''))
                    
                    # Convert to float if possible
                    try:
                        if value is not None:
                            extracted_metrics[metric] = float(value)
                            tag_sources[metric] = tag
                            
                            # Update tag usage statistics
                            self.stats["tag_usage"][metric][tag] += 1
                            self.stats["metrics_found"][metric] += 1
                            break
                    except (ValueError, TypeError):
                        continue
            
            # If not found with primary tags, try fallback tags
            if metric not in extracted_metrics and fallback_tags:
                for tag in fallback_tags:
                    tag_base = tag.split(':')[-1]
                    
                    if tag_base in us_gaap:
                        value = self.extract_value_from_fact(us_gaap[tag_base], metadata.get('filing_type', ''))
                        
                        try:
                            if value is not None:
                                extracted_metrics[metric] = float(value)
                                tag_sources[metric] = tag
                                
                                # Update tag usage statistics
                                self.stats["tag_usage"][metric][tag] += 1
                                self.stats["metrics_found"][metric] += 1
                                break
                        except (ValueError, TypeError):
                            continue
        
        # Calculate derived ratio metrics
        self.derive_ratio_metrics(extracted_metrics)
        
        return extracted_metrics, tag_sources
    
    def derive_ratio_metrics(self, metrics):
        """
        Calculate derived ratio metrics from available metrics.
        """
        for ratio, details in RATIO_METRICS.items():
            # Check if all required metrics are available
            if all(m in metrics for m in details["required_metrics"]):
                try:
                    ratio_value = details["formula"](metrics)
                    if ratio_value is not None:
                        metrics[ratio] = ratio_value
                except Exception as e:
                    logger.debug(f"Error calculating {ratio}: {e}")
    
    def derive_missing_metrics(self, metrics):
        """
        Attempt to derive missing metrics using correlations.
        """
        derived_metrics = {}
        
        # Try to derive each missing metric using correlations
        for target_metric, correlations in METRIC_CORRELATIONS.items():
            # Skip if target already exists
            if target_metric in metrics:
                continue
                
            # Try each correlated metric
            for source_metric, correlation in correlations.items():
                if source_metric in metrics:
                    derived_metrics[target_metric] = {
                        "value": metrics[source_metric],
                        "source": source_metric,
                        "correlation": correlation,
                        "confidence": correlation
                    }
                    break
        
        # Special case calculations
        if "Gross Profit" not in metrics and "COGS" in metrics and "Revenue" in metrics:
            derived_metrics["Gross Profit"] = {
                "value": metrics["Revenue"] - metrics["COGS"],
                "source": "calculated",
                "correlation": 0.95,
                "confidence": 0.95
            }
        
        if "Net Change in Cash" not in metrics and all(m in metrics for m in ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]):
            derived_metrics["Net Change in Cash"] = {
                "value": metrics["Operating Cash Flow"] + metrics["Investing Cash Flow"] + metrics["Financing Cash Flow"],
                "source": "calculated",
                "correlation": 0.95,
                "confidence": 0.95
            }
        
        # Add derived metrics to the original set
        for metric, details in derived_metrics.items():
            metrics[metric] = details["value"]
        
        return metrics, derived_metrics
    





    def process_filing(self, file_path):
        """
        Process a single SEC filing file.
        """
        try:
            # Extract CIK from filename to check if we should process this file
            cik_match = re.search(r'CIK(\d+)\.json', str(file_path))
            if not cik_match:
                logger.debug(f"Skipping {file_path}: No CIK in filename")
                return None

            cik = cik_match.group(1).zfill(10)
            if cik not in self.cik_to_ticker:
                logger.debug(f"Skipping {file_path}: CIK {cik} not in ticker mapping")
                return None

            # First pass - read a sample to extract metadata
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sample_content = f.read(50000)  # Read enough to extract metadata
            except Exception as e:
                logger.debug(f"Error reading file {file_path}: {e}")
                return None

            # Extract metadata
            metadata = self.extract_metadata(sample_content, file_path)
            logger.debug(f"Extracted metadata from {file_path}: {metadata}")

            # Skip if we couldn't determine key metadata
            if not metadata.get('filing_type'):
                logger.debug(f"Skipping {file_path}: No filing_type in metadata")
                return None

            if not metadata.get('period'):
                logger.debug(f"Skipping {file_path}: No period in metadata")
                return None
            
            # Determine appropriate tag mapping
            tag_mapping = self.determine_tag_mapping(metadata)
            
            # Second pass - load full data for processing
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        full_data = json.load(f)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode error: {e}. Trying alternate approach...")
                        # If file too large or has issues, use a different approach
                        f.seek(0)  # Go back to beginning of file
                        content = f.read()
                        
                        # Look for facts section using more flexible pattern
                        facts_patterns = [
                            r'"facts"\s*:\s*({.*})',
                            r'"FACTS"\s*:\s*({.*})',
                            r'"Facts"\s*:\s*({.*})'
                        ]
                        
                        facts_found = False
                        for pattern in facts_patterns:
                            facts_match = re.search(pattern, content, re.DOTALL)
                            if facts_match:
                                facts_str = facts_match.group(1)
                                try:
                                    facts = json.loads(facts_str)
                                    full_data = {"facts": facts}
                                    facts_found = True
                                    break
                                except json.JSONDecodeError:
                                    continue
                                
                        if not facts_found:
                            # If still can't extract facts, try to find US-GAAP data directly
                            us_gaap_match = re.search(r'"us-gaap"\s*:\s*({.*?})', content, re.DOTALL)
                            if us_gaap_match:
                                try:
                                    us_gaap_str = us_gaap_match.group(1)
                                    us_gaap_data = json.loads(us_gaap_str)
                                    full_data = {"facts": {"us-gaap": us_gaap_data}}
                                except:
                                    logger.debug(f"Failed to parse us-gaap section from {file_path}")
                                    return {
                                        "cik": cik,
                                        "ticker": metadata.get('ticker', ''),
                                        "name": metadata.get('name', ''),
                                        "metadata": metadata,
                                        "metrics": {},
                                        "error": "Failed to parse us-gaap section"
                                    }
                            else:
                                logger.debug(f"No facts or us-gaap section found in {file_path}")
                                return {
                                    "cik": cik,
                                    "ticker": metadata.get('ticker', ''),
                                    "name": metadata.get('name', ''),
                                    "metadata": metadata,
                                    "metrics": {},
                                    "error": "No facts section found"
                                }
            except Exception as e:
                logger.error(f"Error reading or parsing {file_path}: {e}")
                return {
                    "cik": cik,
                    "ticker": metadata.get('ticker', ''),
                    "name": metadata.get('name', ''),
                    "metadata": metadata,
                    "metrics": {},
                    "error": f"File reading error: {str(e)}"
                }
            

            metrics, tag_sources = self.extract_metrics(full_data, metadata, tag_mapping)
            
            # Attempt to derive missing metrics
            metrics, derived_metrics = self.derive_missing_metrics(metrics)
            
            # Add metadata to metrics
            if 'period_id' in metadata:
                period_id = metadata['period_id']
            elif 'period_formatted' in metadata:
                period_id = metadata['period_formatted']
            else:
                period_id = metadata['period']
            
            # Format output
            result = {
                "cik": cik,
                "ticker": metadata.get('ticker', ''),
                "name": metadata.get('name', ''),
                "metadata": metadata,
                "period_id": period_id,
                "metrics": metrics,
                "tag_sources": tag_sources,
                "derived_metrics": derived_metrics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return {
                "cik": cik if 'cik' in locals() else None,
                "error": str(e),
                "file_path": str(file_path)
            }
    
    def update_company_data(self, filing_result):
        """
        Update company data with results from a processed filing.
        """
        if not filing_result or "error" in filing_result and not filing_result.get("metrics"):
            return False
            
        cik = filing_result["cik"]
        ticker = filing_result["ticker"]
        period_id = filing_result["period_id"]
        
        # Initialize company data if not exists
        if cik not in self.company_data:
            self.company_data[cik] = {
                "cik": cik,
                "ticker": ticker,
                "name": filing_result["name"],
                "periods": set(),
                "metrics": defaultdict(dict),
                "filing_dates": {},
                "data_sources": defaultdict(dict)
            }
        
        # Add period
        self.company_data[cik]["periods"].add(period_id)
        
        # Add filing date
        if "filing_date" in filing_result["metadata"]:
            self.company_data[cik]["filing_dates"][period_id] = filing_result["metadata"]["filing_date"]
        
        # Add metrics
        for metric, value in filing_result["metrics"].items():
            self.company_data[cik]["metrics"][metric][period_id] = value
            
            # Add data source information
            if "tag_sources" in filing_result and metric in filing_result["tag_sources"]:
                self.company_data[cik]["data_sources"][metric][period_id] = {
                    "source": "direct",
                    "tag": filing_result["tag_sources"][metric]
                }
            elif "derived_metrics" in filing_result and metric in filing_result["derived_metrics"]:
                derived_info = filing_result["derived_metrics"][metric]
                self.company_data[cik]["data_sources"][metric][period_id] = {
                    "source": "derived",
                    "method": derived_info.get("source", "unknown"),
                    "confidence": derived_info.get("confidence", 0.5)
                }
        
        # Track statistics
        self.stats["companies_processed"].add(cik)
        
        return True
    






    # Memory-optimized version of process_filings_in_batches
    def process_filings_in_batches(self, batch_size=500, max_workers=None):
        """
        Process valid filing files in smaller batches with better memory management.
        """
        logger.info("Starting optimized batch processing of filings")

        # Get list of validated filing files
        valid_files = self.filter_valid_filing_files()

        self.stats["total_files"] = len(valid_files)
        logger.info(f"Found {len(valid_files)} valid filing files to process")

        # Use fewer workers if specified
        if max_workers is None:
            max_workers = min(32, (os.cpu_count() or 1) * 2)  # Default to 2x CPU cores, max 32
        else:
            max_workers = min(max_workers, (os.cpu_count() or 1) * 2)  # Limit to 2x CPU cores

        logger.info(f"Using {max_workers} worker processes")

        # Process in smaller batches
        batch_size = min(batch_size, 500)  # Cap batch size at 500
        total_batches = (len(valid_files) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(valid_files))
            batch_files = valid_files[start_idx:end_idx]

            logger.info(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch_files)} files)")

            # Process batch in parallel with fewer workers
            results = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {executor.submit(self.process_filing, str(file_path)): file_path 
                    for file_path in batch_files}

                for future in tqdm(as_completed(future_to_file), total=len(batch_files), 
                                 desc=f"Batch {batch_idx+1}/{total_batches}", unit="file"):
                    file_path = future_to_file[future]

                    try:
                        result = future.result()
                        if result:
                            if "error" in result and not result.get("metrics"):
                                self.stats["error_files"] += 1
                            else:
                                self.stats["processed_files"] += 1
                                results.append(result)
                    except Exception as e:
                        self.stats["error_files"] += 1
                        logger.error(f"Error processing {file_path}: {e}")

            # Update company data with batch results
            for result in results:
                self.update_company_data(result)

            # Free up memory explicitly
            results = None
            gc.collect()

            logger.info(f"Completed batch {batch_idx+1}/{total_batches}. "
                      f"Processed: {self.stats['processed_files']}, "
                      f"Errors: {self.stats['error_files']}")

            # Break if we encounter memory issues
            if self.stats['error_files'] > self.stats['processed_files'] * 0.5:
                logger.warning("High error rate detected, possibly due to memory issues. Stopping processing.")
                break

        logger.info(f"Completed all batches. Total files processed: {self.stats['processed_files']}")
        return self.stats["processed_files"]
    
    def save_company_data(self):
        """
        Save processed company data to parquet files.
        """
        logger.info(f"Saving data for {len(self.company_data)} companies")
        
        companies_saved = 0
        
        for cik, company in tqdm(self.company_data.items(), desc="Saving company data"):
            try:
                ticker = company["ticker"]
                if not ticker:
                    continue
                
                # Convert company data to DataFrame
                rows = []
                periods = sorted(list(company["periods"]))
                
                for period in periods:
                    row = {
                        "Period": period,
                        "FilingDate": company["filing_dates"].get(period, "")
                    }
                    
                    # Add metrics
                    for metric, values in company["metrics"].items():
                        if period in values:
                            row[metric] = values[period]
                    
                    rows.append(row)
                
                # Create DataFrame and save
                if rows:
                    df = pd.DataFrame(rows)
                    output_file = self.output_dir / f"{ticker}.parquet"
                    df.to_parquet(output_file)
                    companies_saved += 1
            except Exception as e:
                logger.error(f"Error saving data for {company['ticker']}: {e}")
        
        logger.info(f"Saved data for {companies_saved} companies")
        return companies_saved
    


    def create_consolidated_metrics(self):
        """
        Create consolidated metrics files from all company data.
        """
        logger.info("Creating consolidated metrics files")
        
        # Annual metrics (most recent year for each company)
        annual_metrics = []
        quarterly_metrics = []
        
        for cik, company in self.company_data.items():
            ticker = company["ticker"]
            name = company["name"]
            periods = company["periods"]
            
            # Separate annual and quarterly periods
            annual_periods = sorted([p for p in periods if '-Q' not in p and len(p) == 4])
            quarterly_periods = sorted([p for p in periods if '-Q' in p])
            
            # Get most recent annual data
            if annual_periods:
                latest_year = annual_periods[-1]
                annual_row = {
                    "Ticker": ticker,
                    "Name": name,
                    "CIK": cik,
                    "Period": latest_year,
                    "FilingDate": company["filing_dates"].get(latest_year, "")
                }
                
                # Add all metrics for this year
                for metric, values in company["metrics"].items():
                    if latest_year in values:
                        annual_row[metric] = values[latest_year]
                
                annual_metrics.append(annual_row)
            
            # Get most recent quarterly data
            if quarterly_periods:
                latest_quarter = quarterly_periods[-1]
                quarterly_row = {
                    "Ticker": ticker,
                    "Name": name,
                    "CIK": cik,
                    "Period": latest_quarter,
                    "FilingDate": company["filing_dates"].get(latest_quarter, "")
                }
                
                # Add all metrics for this quarter
                for metric, values in company["metrics"].items():
                    if latest_quarter in values:
                        quarterly_row[metric] = values[latest_quarter]
                
                quarterly_metrics.append(quarterly_row)
        
        # Save consolidated metrics
        if annual_metrics:
            annual_df = pd.DataFrame(annual_metrics)
            annual_df.to_parquet(METRICS_DIR / "annual_metrics.parquet")
            logger.info(f"Saved annual metrics for {len(annual_metrics)} companies")
        
        if quarterly_metrics:
            quarterly_df = pd.DataFrame(quarterly_metrics)
            quarterly_df.to_parquet(METRICS_DIR / "quarterly_metrics.parquet")
            logger.info(f"Saved quarterly metrics for {len(quarterly_metrics)} companies")
        
        # Save metrics availability statistics
        self.save_metrics_statistics()
        
        return len(annual_metrics), len(quarterly_metrics)
    
    def save_metrics_statistics(self):
        """
        Save statistics about metrics availability to a JSON file.
        """
        # Calculate metrics availability
        metrics_stats = {}
        total_companies = len(self.stats["companies_processed"])
        
        for metric, count in self.stats["metrics_found"].items():
            metrics_stats[metric] = {
                "count": count,
                "availability": count / total_companies if total_companies > 0 else 0,
                "top_tags": self.stats["tag_usage"][metric].most_common(5)
            }
        
        # Save to JSON
        with open(METRICS_DIR / "metrics_statistics.json", "w") as f:
            json.dump({
                "total_files": self.stats["total_files"],
                "processed_files": self.stats["processed_files"],
                "error_files": self.stats["error_files"],
                "companies_processed": len(self.stats["companies_processed"]),
                "metrics_stats": metrics_stats
            }, f, indent=2)
        
        logger.info("Saved metrics statistics")
    



    def process_all_data(self, ticker_file, batch_size=500, max_workers=None):
        """
        Complete pipeline to process all SEC filing data with memory optimization.
        """
        start_time = time.time()
        logger.info("Starting SEC data processing pipeline with memory optimization")

        # Load ticker mapping
        self.load_ticker_mapping(ticker_file)

        # Process filings in batches with memory optimization
        files_processed = self.process_filings_in_batches(batch_size, max_workers)

        # Save company data
        companies_saved = self.save_company_data()

        # Create consolidated metrics
        annual_count, quarterly_count = self.create_consolidated_metrics()

        elapsed_time = time.time() - start_time
        logger.info(f"Completed SEC data processing in {elapsed_time:.2f} seconds")
        logger.info(f"Files processed: {files_processed}")
        logger.info(f"Companies saved: {companies_saved}")
        logger.info(f"Annual metrics: {annual_count}, Quarterly metrics: {quarterly_count}")

        # Output processing summary
        print("\n===== SEC Data Processing Summary =====")
        print(f"Total files found: {self.stats['total_files']}")
        print(f"Files processed successfully: {self.stats['processed_files']} ({self.stats['processed_files']/self.stats['total_files']*100:.1f}%)")
        print(f"Files with errors: {self.stats['error_files']} ({self.stats['error_files']/self.stats['total_files']*100:.1f}%)")
        print(f"Companies processed: {len(self.stats['companies_processed'])}")
        print(f"Companies saved: {companies_saved}")
        print("")
        print("Top 10 most available metrics:")

        # Sort metrics by availability
        sorted_metrics = sorted(
            [(m, c) for m, c in self.stats["metrics_found"].items()],
            key=lambda x: x[1],
            reverse=True
        )

        for metric, count in sorted_metrics[:10]:
            availability = count / len(self.stats["companies_processed"]) if self.stats["companies_processed"] else 0
            print(f"  {metric}: {count} companies ({availability*100:.1f}%)")

        print(f"\nProcessing time: {elapsed_time:.2f} seconds")
        print("=======================================\n")

        return {
            "files_processed": files_processed,
            "companies_saved": companies_saved,
            "annual_metrics": annual_count,
            "quarterly_metrics": quarterly_count,
            "processing_time": elapsed_time
        }



class EnhancedSECDataProcessor(SECDataProcessor):
    """
    Enhanced SEC EDGAR data processor with improved metric derivation strategies
    to reduce NaN values in financial datasets.
    """
    
    def extract_metrics(self, json_data, metadata, tag_mapping):
        """
        Enhanced extraction of financial metrics from JSON data using the provided tag mapping.
        Implements more flexible tag matching and better handling of units.
        """
        extracted_metrics = {}
        tag_sources = {}
        
        # Process US GAAP facts first
        facts = json_data.get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        
        # Also check for IFRS facts if present
        ifrs_full = facts.get('ifrs-full', {})
        
        # Extract each metric using appropriate tags
        for metric, tags in tag_mapping.items():
            primary_tags = tags['primary']
            fallback_tags = tags['fallback']
            
            # Helper function to check a tag in a namespace
            def check_tag_in_namespace(tag, namespace_dict):
                tag_base = tag.split(':')[-1]  # Remove namespace prefix
                if tag_base in namespace_dict:
                    value = self.extract_value_from_fact(namespace_dict[tag_base], metadata.get('filing_type', ''))
                    try:
                        if value is not None:
                            return float(value), tag
                    except (ValueError, TypeError):
                        pass
                return None, None
            
            # Try primary tags first in US GAAP
            found = False
            for tag in primary_tags:
                namespace_prefix = tag.split(':')[0] if ':' in tag else 'us-gaap'
                
                # Check in appropriate namespace
                if namespace_prefix == 'us-gaap':
                    value, used_tag = check_tag_in_namespace(tag, us_gaap)
                elif namespace_prefix == 'ifrs-full' and ifrs_full:
                    value, used_tag = check_tag_in_namespace(tag, ifrs_full)
                else:
                    # Try to find in other namespaces if available
                    for namespace_name, namespace_data in facts.items():
                        if namespace_name != 'us-gaap' and namespace_name != 'ifrs-full':
                            value, used_tag = check_tag_in_namespace(tag, namespace_data)
                            if value is not None:
                                break
                    else:
                        value, used_tag = None, None
                
                if value is not None:
                    extracted_metrics[metric] = value
                    tag_sources[metric] = used_tag
                    
                    # Update tag usage statistics
                    self.stats["tag_usage"][metric][used_tag] += 1
                    self.stats["metrics_found"][metric] += 1
                    found = True
                    break
            
            # If not found with primary tags, try fallback tags
            if not found and fallback_tags:
                for tag in fallback_tags:
                    namespace_prefix = tag.split(':')[0] if ':' in tag else 'us-gaap'
                    
                    # Check in appropriate namespace
                    if namespace_prefix == 'us-gaap':
                        value, used_tag = check_tag_in_namespace(tag, us_gaap)
                    elif namespace_prefix == 'ifrs-full' and ifrs_full:
                        value, used_tag = check_tag_in_namespace(tag, ifrs_full)
                    else:
                        # Try to find in other namespaces
                        for namespace_name, namespace_data in facts.items():
                            if namespace_name != 'us-gaap' and namespace_name != 'ifrs-full':
                                value, used_tag = check_tag_in_namespace(tag, namespace_data)
                                if value is not None:
                                    break
                        else:
                            value, used_tag = None, None
                    
                    if value is not None:
                        extracted_metrics[metric] = value
                        tag_sources[metric] = used_tag
                        
                        # Update tag usage statistics
                        self.stats["tag_usage"][metric][used_tag] += 1
                        self.stats["metrics_found"][metric] += 1
                        break
        
        # Calculate derived ratio metrics
        self.derive_ratio_metrics(extracted_metrics)
        
        return extracted_metrics, tag_sources
    
    def derive_missing_metrics(self, metrics):
        """
        Enhanced implementation to derive missing metrics using correlations and financial relationships.
        Uses multiple passes to handle interdependent metrics.
        """
        derived_metrics = {}
        derivation_log = []
        
        # Track which metrics were derived in this iteration to avoid circular logic
        derived_this_round = set()
        
        # Multiple passes to handle dependent derivations
        for iteration in range(3):  # Limited to 3 passes to prevent infinite loops
            derived_this_round.clear()
            
            # Try standard correlation-based derivation first
            for target_metric, correlations in METRIC_CORRELATIONS.items():
                # Skip if target already exists
                if target_metric in metrics:
                    continue
                    
                # Try each correlated metric 
                for source_metric, correlation in correlations.items():
                    if source_metric in metrics and correlation > 0.90:  # Higher threshold for reliability
                        # Use a direct value transfer for highly correlated metrics
                        derived_metrics[target_metric] = {
                            "value": metrics[source_metric],
                            "source": f"correlation with {source_metric}",
                            "correlation": correlation,
                            "confidence": correlation
                        }
                        derived_this_round.add(target_metric)
                        derivation_log.append(f"Derived {target_metric} from {source_metric} (correlation: {correlation})")
                        break
            
            # Apply financial relationship derivations
            
            # Revenue = Gross Profit + COGS
            if "Revenue" not in metrics and "Gross Profit" in metrics and "COGS" in metrics:
                value = metrics["Gross Profit"] + metrics["COGS"]
                derived_metrics["Revenue"] = {
                    "value": value,
                    "source": "calculated from Gross Profit + COGS",
                    "correlation": 0.98,
                    "confidence": 0.98
                }
                derived_this_round.add("Revenue")
                derivation_log.append("Derived Revenue from Gross Profit + COGS")
                
            # Gross Profit = Revenue - COGS
            if "Gross Profit" not in metrics and "Revenue" in metrics and "COGS" in metrics:
                value = metrics["Revenue"] - metrics["COGS"]
                derived_metrics["Gross Profit"] = {
                    "value": value,
                    "source": "calculated from Revenue - COGS",
                    "correlation": 0.98,
                    "confidence": 0.98
                }
                derived_this_round.add("Gross Profit")
                derivation_log.append("Derived Gross Profit from Revenue - COGS")
                
            # COGS = Revenue - Gross Profit
            if "COGS" not in metrics and "Revenue" in metrics and "Gross Profit" in metrics:
                value = metrics["Revenue"] - metrics["Gross Profit"]
                derived_metrics["COGS"] = {
                    "value": value,
                    "source": "calculated from Revenue - Gross Profit",
                    "correlation": 0.98,
                    "confidence": 0.98
                }
                derived_this_round.add("COGS")
                derivation_log.append("Derived COGS from Revenue - Gross Profit")
            
            # Net Change in Cash = Operating Cash Flow + Investing Cash Flow + Financing Cash Flow
            if "Net Change in Cash" not in metrics and all(m in metrics for m in ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]):
                value = metrics["Operating Cash Flow"] + metrics["Investing Cash Flow"] + metrics["Financing Cash Flow"]
                derived_metrics["Net Change in Cash"] = {
                    "value": value,
                    "source": "calculated from OCF + ICF + FCF",
                    "correlation": 0.95,
                    "confidence": 0.95
                }
                derived_this_round.add("Net Change in Cash")
                derivation_log.append("Derived Net Change in Cash from cash flow components")
                
            # EPS derivations using shares
            if "EPS (Basic)" not in metrics and "Net Income" in metrics and "Shares Outstanding (Basic)" in metrics and metrics["Shares Outstanding (Basic)"] > 0:
                value = metrics["Net Income"] / metrics["Shares Outstanding (Basic)"]
                derived_metrics["EPS (Basic)"] = {
                    "value": value,
                    "source": "calculated from Net Income / Shares Outstanding (Basic)",
                    "correlation": 0.99,
                    "confidence": 0.99
                }
                derived_this_round.add("EPS (Basic)")
                derivation_log.append("Derived EPS (Basic) from Net Income / Shares Outstanding (Basic)")
                
            if "EPS (Diluted)" not in metrics and "Net Income" in metrics and "Shares Outstanding (Diluted)" in metrics and metrics["Shares Outstanding (Diluted)"] > 0:
                value = metrics["Net Income"] / metrics["Shares Outstanding (Diluted)"]
                derived_metrics["EPS (Diluted)"] = {
                    "value": value,
                    "source": "calculated from Net Income / Shares Outstanding (Diluted)",
                    "correlation": 0.99,
                    "confidence": 0.99
                }
                derived_this_round.add("EPS (Diluted)")
                derivation_log.append("Derived EPS (Diluted) from Net Income / Shares Outstanding (Diluted)")
                
            # EPS can be derived from each other if one exists
            if "EPS (Basic)" not in metrics and "EPS (Diluted)" in metrics:
                derived_metrics["EPS (Basic)"] = {
                    "value": metrics["EPS (Diluted)"] * 1.02,  # Basic typically slightly higher than diluted
                    "source": "estimated from EPS (Diluted)",
                    "correlation": 0.99,
                    "confidence": 0.95
                }
                derived_this_round.add("EPS (Basic)")
                derivation_log.append("Derived EPS (Basic) from EPS (Diluted)")
                
            if "EPS (Diluted)" not in metrics and "EPS (Basic)" in metrics:
                value = metrics["EPS (Basic)"] * 0.98  # Diluted typically slightly lower than Basic
                derived_metrics["EPS (Diluted)"] = {
                    "value": value,
                    "source": "estimated from EPS (Basic)",
                    "correlation": 0.99,
                    "confidence": 0.95
                }
                derived_this_round.add("EPS (Diluted)")
                derivation_log.append("Derived EPS (Diluted) from EPS (Basic)")
            
            # Shares derivations
            if "Shares Outstanding (Basic)" not in metrics and "Shares Outstanding (Diluted)" in metrics:
                value = metrics["Shares Outstanding (Diluted)"] * 0.98  # Basic typically slightly less than diluted
                derived_metrics["Shares Outstanding (Basic)"] = {
                    "value": value,
                    "source": "estimated from Shares Outstanding (Diluted)",
                    "correlation": 0.99,
                    "confidence": 0.95
                }
                derived_this_round.add("Shares Outstanding (Basic)")
                derivation_log.append("Derived Shares Outstanding (Basic) from Shares Outstanding (Diluted)")
            
            if "Shares Outstanding (Diluted)" not in metrics and "Shares Outstanding (Basic)" in metrics:
                value = metrics["Shares Outstanding (Basic)"] * 1.02  # Diluted typically slightly more than basic
                derived_metrics["Shares Outstanding (Diluted)"] = {
                    "value": value,
                    "source": "estimated from Shares Outstanding (Basic)",
                    "correlation": 0.99,
                    "confidence": 0.95
                }
                derived_this_round.add("Shares Outstanding (Diluted)")
                derivation_log.append("Derived Shares Outstanding (Diluted) from Shares Outstanding (Basic)")
            
            # Income-related derivations
            if "Income Before Tax" not in metrics and "Net Income" in metrics:
                # Assume typical effective tax rate of 25%
                value = metrics["Net Income"] / 0.75
                derived_metrics["Income Before Tax"] = {
                    "value": value,
                    "source": "estimated from Net Income with assumed tax rate",
                    "correlation": 0.95,
                    "confidence": 0.90
                }
                derived_this_round.add("Income Before Tax")
                derivation_log.append("Derived Income Before Tax from Net Income")
            
            if "Net Income" not in metrics and "Income Before Tax" in metrics:
                # Assume typical effective tax rate of 25%
                value = metrics["Income Before Tax"] * 0.75
                derived_metrics["Net Income"] = {
                    "value": value,
                    "source": "estimated from Income Before Tax with assumed tax rate",
                    "correlation": 0.95,
                    "confidence": 0.90
                }
                derived_this_round.add("Net Income")
                derivation_log.append("Derived Net Income from Income Before Tax")
            
            # Operating Income related derivations
            if "Operating Income" not in metrics and "Income Before Tax" in metrics:
                # Assume operating income is slightly less than income before tax
                value = metrics["Income Before Tax"] * 0.95
                derived_metrics["Operating Income"] = {
                    "value": value,
                    "source": "estimated from Income Before Tax",
                    "correlation": 0.95,
                    "confidence": 0.90
                }
                derived_this_round.add("Operating Income")
                derivation_log.append("Derived Operating Income from Income Before Tax")
            
            if "Operating Income" not in metrics and "Net Income" in metrics:
                # Assume operating income is about 35% higher than net income (due to taxes and interest)
                value = metrics["Net Income"] * 1.35
                derived_metrics["Operating Income"] = {
                    "value": value,
                    "source": "estimated from Net Income",
                    "correlation": 0.90,
                    "confidence": 0.85
                }
                derived_this_round.add("Operating Income")
                derivation_log.append("Derived Operating Income from Net Income")
            
            # Update metrics with derived values for next iteration
            for metric, details in derived_metrics.items():
                if metric in derived_this_round:
                    metrics[metric] = details["value"]
            
            # If nothing was derived this round, break early
            if not derived_this_round:
                break
        
        # Calculate ratio metrics once all derivations are done
        if "Revenue" in metrics and metrics["Revenue"] > 0:
            # Gross Margin calculation
            if "Gross Profit" in metrics and "GrossMargin" not in metrics:
                metrics["GrossMargin"] = metrics["Gross Profit"] / metrics["Revenue"]
                derived_metrics["GrossMargin"] = {
                    "value": metrics["GrossMargin"],
                    "source": "calculated from Gross Profit / Revenue",
                    "correlation": 1.0,
                    "confidence": 1.0
                }
                
            # Operating Margin calculation
            if "Operating Income" in metrics and "OperatingMargin" not in metrics:
                metrics["OperatingMargin"] = metrics["Operating Income"] / metrics["Revenue"]
                derived_metrics["OperatingMargin"] = {
                    "value": metrics["OperatingMargin"],
                    "source": "calculated from Operating Income / Revenue",
                    "correlation": 1.0,
                    "confidence": 1.0
                }
                
            # Net Margin calculation
            if "Net Income" in metrics and "NetMargin" not in metrics:
                metrics["NetMargin"] = metrics["Net Income"] / metrics["Revenue"]
                derived_metrics["NetMargin"] = {
                    "value": metrics["NetMargin"],
                    "source": "calculated from Net Income / Revenue",
                    "correlation": 1.0,
                    "confidence": 1.0
                }
        
        # Return enhanced metrics and derivation info
        return metrics, derived_metrics, derivation_log
    
    def determine_tag_mapping(self, metadata):
        """
        Enhanced method to determine the appropriate tag mapping based on metadata.
        Includes better industry-specific mappings.
        """
        tag_mapping = {}
        
        # Start with standard tag mappings
        for metric, mapping in METRIC_TAG_MAPPING.items():
            tag_mapping[metric] = {
                "primary": mapping["primary"].copy(),
                "fallback": mapping["fallback"].copy()
            }
        
        # Apply industry-specific mappings if available
        industry = metadata.get('industry')
        sic = metadata.get('sic')
        
        # Try to determine industry from SIC if not provided
        if not industry and sic:
            industry = SIC_TO_INDUSTRY.get(sic, "OTHER")
            
        if industry and industry in INDUSTRY_TAG_MAPPINGS:
            industry_mappings = INDUSTRY_TAG_MAPPINGS[industry]
            
            for metric, mapping in industry_mappings.items():
                if metric in tag_mapping:
                    # Add industry-specific primary tags to the front of the list
                    tag_mapping[metric]["primary"] = mapping["primary"] + tag_mapping[metric]["primary"]
                    tag_mapping[metric]["fallback"] = mapping["fallback"] + tag_mapping[metric]["fallback"]
                else:
                    tag_mapping[metric] = mapping
        
        return tag_mapping
    
    def extract_value_from_fact(self, fact, filing_type):
        """
        Enhanced method to extract appropriate values from facts with better handling of 
        different units and time periods.
        """
        if not fact:
            return None
        
        # Handle various fact structures
        if isinstance(fact, dict):
            # Direct value access
            if 'val' in fact:
                return fact['val']
            
            # Handle units section
            if 'units' in fact:
                units = fact['units']
                
                # Try currency values first (USD, EUR, etc.)
                for currency_key in ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'AUD']:
                    if currency_key in units and units[currency_key]:
                        values = units[currency_key]
                        
                        # Try to find values matching the filing type
                        for value_obj in values:
                            form_match = False
                            
                            # Check if form matches
                            if 'form' in value_obj:
                                form_match = value_obj['form'] in (filing_type, '10-K', '10-Q', '20-F', '8-K')
                            
                            # If matched or if no form specified but value present, return it
                            if (form_match or 'form' not in value_obj) and 'val' in value_obj:
                                return value_obj['val']
                        
                        # If no match by form, take the first value with val
                        for value_obj in values:
                            if 'val' in value_obj:
                                return value_obj['val']
                
                # Try shares for share counts
                if 'shares' in units and units['shares']:
                    values = units['shares']
                    # First try to match by form
                    for value_obj in values:
                        if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                            if 'val' in value_obj:
                                return value_obj['val']
                    
                    # If no match, take first available
                    if values and 'val' in values[0]:
                        return values[0]['val']
                
                # Try per-share values
                for per_share_key in ['USD/shares', 'EUR/shares', 'JPY/shares', 'pure']:
                    if per_share_key in units and units[per_share_key]:
                        values = units[per_share_key]
                        # Try to match by form
                        for value_obj in values:
                            if 'form' in value_obj and value_obj['form'] in (filing_type, '10-K', '10-Q'):
                                if 'val' in value_obj:
                                    return value_obj['val']
                        
                        # If no match, take first available
                        if values and 'val' in values[0]:
                            return values[0]['val']
                
                # Fall back to any other unit type available
                for unit_type, values in units.items():
                    if values and isinstance(values, list) and len(values) > 0:
                        for value_obj in values:
                            if 'val' in value_obj:
                                return value_obj['val']
        
        # Handle simple value
        return fact
    
    def process_filing(self, file_path):
        """
        Enhanced method to process a single SEC filing file with better metadata extraction
        and error handling.
        """
        result = super().process_filing(file_path)
        
        # If we got a result with metrics, try to enhance it further
        if result and "metrics" in result:
            # Try more advanced derivation if needed
            if len(result["metrics"]) > 0:
                try:
                    enhanced_metrics, derived_metrics, derivation_log = self.derive_missing_metrics(result["metrics"])
                    result["metrics"] = enhanced_metrics
                    result["derived_metrics"] = derived_metrics
                    result["derivation_log"] = derivation_log
                except Exception as e:
                    self.logger.warning(f"Error during enhanced derivation for {file_path}: {e}")
        
        return result









def main():
    """
    Main entry point for SEC data processing with memory optimization.
    """
    # Configure more verbose logging for debugging
    logging.basicConfig(level=logging.INFO)
    
    logger = get_logger(script_name="Z-Macro")
    
    # Log important directory information
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Data directory exists: {os.path.exists('Data')}")
    logger.info(f"Filings directory exists: {os.path.exists('Data/Filings')}")
    
    # Check ticker file exists
    ticker_file = "Data/TickerCikData/TickerCIKs_20250325.parquet"
    logger.info(f"Ticker file exists: {os.path.exists(ticker_file)}")
    
    # Create processor instance with memory optimization
    processor = SECDataProcessor(
        base_dir="Data",
        filings_dir="Data/Filings",
        output_dir="Data/SEC_Data/Enhanced",
        max_workers=16,  # Reduced worker count
        logger=logger,
    )
    
    # Process all data with smaller batch size and fewer workers
    result = processor.process_all_data(
        ticker_file=ticker_file,
        batch_size=500,  # Smaller batch size
        max_workers=16  # Reduced worker count
    )
    
    return result








if __name__ == "__main__":
    main()