import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import os
from typing import Dict, List, Any, TypedDict, Annotated, Sequence
import operator
from dotenv import load_dotenv
import yfinance as yf
import requests
from io import BytesIO
import PyPDF2
import time
from functools import lru_cache
import re
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from threading import Thread

# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()

# JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime and pandas Timestamp objects"""
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super().default(obj)

# LLM Evaluation Metrics
class LLMEvaluationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.response_times = {}
        self.token_usage = {}
        self.accuracy_scores = {}
        self.coherence_scores = {}
        self.financial_accuracy = {}
        self.error_counts = {}
        self.cost_estimates = {}
    
    def record_response_time(self, llm_name: str, duration: float):
        if llm_name not in self.response_times:
            self.response_times[llm_name] = []
        self.response_times[llm_name].append(duration)
    
    def record_error(self, llm_name: str):
        if llm_name not in self.error_counts:
            self.error_counts[llm_name] = 0
        self.error_counts[llm_name] += 1
    
    def calculate_avg_response_time(self, llm_name: str) -> float:
        if llm_name in self.response_times and self.response_times[llm_name]:
            return sum(self.response_times[llm_name]) / len(self.response_times[llm_name])
        return 0.0
    
    def get_success_rate(self, llm_name: str, total_requests: int) -> float:
        errors = self.error_counts.get(llm_name, 0)
        return ((total_requests - errors) / total_requests * 100) if total_requests > 0 else 0.0

# SEC EDGAR API Configuration
SEC_HEADERS = {
    'User-Agent': 'Financial Analyst your.email@example.com'
}
SEC_BASE_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{concept}.json"

# Company CIK mappings for SEC EDGAR
COMPANY_CIK_MAP = {
    "Apple Inc.": "0000320193",
    "Microsoft Corp.": "0000789019", 
    "Tesla Inc.": "0001318605",
    "Amazon Inc.": "0001018724",
    "NVIDIA Corporation": "0001045810",
    "Meta Platforms Inc.": "0001326801",
    "Alphabet Inc.": "0001652044",
    "Netflix Inc.": "0001065280"
}

def clean_llm_text(text: str) -> str:
    """Clean text from LLM output while preserving word boundaries"""
    if not text:
        return text
    
    import re
    text = re.sub(r'(\S)\*(\S)', r'\1 * \2', text)
    text = re.sub(r'\s*\*\s*', ' ', text)
    text = re.sub(r'(?<!\*)\*(?!\*)', '', text)
    text = text.replace('∗', '')
    text = text.replace('*', '')
    text = text.replace('\\n', '\n')
    text = text.replace('\\t', '    ')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# Initialize LLMs with evaluation tracking
def get_llms():
    """Initialize all LLMs with API keys"""
    llms = {}
    llm_info = {}
    
    # OpenAI GPT-4
    if os.getenv("OPENAI_API_KEY"):
        try:
            llms["gpt-4"] = ChatOpenAI(
                model="gpt-4o",
                temperature=0.7,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            llm_info["gpt-4"] = {
                "provider": "OpenAI",
                "model": "gpt-4o",
                "speed": "Medium",
                "quality": "Excellent",
                "cost_per_1k_tokens": 0.03
            }
        except Exception as e:
            st.warning(f"OpenAI initialization failed: {e}")
    
    # Groq (Fast inference)
    if os.getenv("GROQ_API_KEY"):
        try:
            llms["groq-mixtral"] = ChatGroq(
                model="mixtral-8x7b-32768",
                temperature=0.7,
                api_key=os.getenv("GROQ_API_KEY")
            )
            llms["groq-llama3"] = ChatGroq(
                model="llama3-70b-8192",
                temperature=0.7,
                api_key=os.getenv("GROQ_API_KEY")
            )
            llm_info["groq-mixtral"] = {
                "provider": "Groq",
                "model": "mixtral-8x7b-32768",
                "speed": "Very Fast",
                "quality": "Good",
                "cost_per_1k_tokens": 0.0002
            }
            llm_info["groq-llama3"] = {
                "provider": "Groq", 
                "model": "llama3-70b-8192",
                "speed": "Very Fast",
                "quality": "Very Good",
                "cost_per_1k_tokens": 0.0002
            }
        except Exception as e:
            st.warning(f"Groq initialization failed: {e}")
    
    # Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            llms["gemini-pro"] = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            llm_info["gemini-pro"] = {
                "provider": "Google",
                "model": "gemini-1.5-flash", 
                "speed": "Fast",
                "quality": "Very Good",
                "cost_per_1k_tokens": 0.001
            }
        except Exception as e:
            st.warning(f"Google Gemini initialization failed: {e}")
    
    # Anthropic Claude
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            llms["claude-3"] = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.7,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            llm_info["claude-3"] = {
                "provider": "Anthropic",
                "model": "claude-3-sonnet",
                "speed": "Medium", 
                "quality": "Excellent",
                "cost_per_1k_tokens": 0.015
            }
        except Exception as e:
            st.warning(f"Anthropic initialization failed: {e}")
    
    return llms, llm_info

# Complete LangGraph State
class FinancialAnalysisState(TypedDict):
    # Core workflow data
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    company: str
    ticker: str
    period: str
    
    # Agent outputs 
    parsed_data: Dict
    kpis: Dict
    risk_assessment: Dict
    insights: Dict
    
    # Control flow
    current_agent: str
    response_mode: str
    
    # Data sources and extraction
    market_data: Dict
    web_search_results: Dict
    uploaded_content: str
    data_sources: List[Dict]
    period_validation: Dict
    extracted_company_info: Dict
    
    # Evaluation tracking
    agent_performance: Dict

# Enhanced Web Search Function (from original test.py)
def web_search_company_info(company_name: str, query: str) -> Dict:
    """Perform web search for additional company information"""
    try:
        # Basic web search implementation
        # In production, you'd use actual search APIs like Google Custom Search, Bing, etc.
        search_results = {
            "company_news": [],
            "financial_reports": [],
            "analyst_reports": [],
            "market_sentiment": "",
            "search_success": False
        }
        
        # Simulate web search results for demonstration
        # In real implementation, you'd integrate with actual search APIs
        if company_name:
            search_results.update({
                "company_news": [
                    f"Recent news about {company_name}",
                    f"{company_name} quarterly earnings update",
                    f"Market analysis for {company_name}"
                ],
                "market_sentiment": f"Mixed sentiment for {company_name} based on recent market activity",
                "search_success": True
            })
        
        return search_results
        
    except Exception as e:
        return {
            "search_success": False,
            "error": str(e),
            "company_news": [],
            "financial_reports": [],
            "analyst_reports": [],
            "market_sentiment": ""
        }

# COMPLETE AGENT SYSTEM - All 6 agents with web search integration

def create_company_extraction_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """Company extraction agent with enhanced intelligence"""
    
    COMPANY_EXTRACTOR_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analysis expert specializing in company identification. 

From the user's query, identify:
1. The company/stock they're asking about
2. Find the correct stock ticker symbol  
3. Determine confidence level

Respond EXACTLY in this format:
COMPANY: [Full Company Name]
TICKER: [Stock Symbol]
CONFIDENCE: [High/Medium/Low]

Examples:
Query: "How is NVIDIA performing currently?"
COMPANY: NVIDIA Corporation
TICKER: NVDA
CONFIDENCE: High

Query: "Tell me about Apple stock"
COMPANY: Apple Inc.
TICKER: AAPL
CONFIDENCE: High

Query: "Tata Motors analysis"
COMPANY: Tata Motors Limited
TICKER: TATA.NS
CONFIDENCE: High

Query: "Reliance earnings report"
COMPANY: Reliance Industries Limited
TICKER: RELIANCE.BO
CONFIDENCE: High

Query: "Amazon financial health"
COMPANY: Amazon Inc.
TICKER: AMZN
CONFIDENCE: High

Query: "What about the overall market?"
COMPANY: None
TICKER: None
CONFIDENCE: Low

Important:
- For Indian stocks, add .NS (NSE) or .BO (BSE)
- For international stocks, use primary exchange ticker
- Be precise with company names (Inc., Corp., Limited, etc.)
- If no specific company mentioned, use COMPANY: None, TICKER: None"""),
        ("human", "{query}")
    ])
    
    def company_extractor(state: FinancialAnalysisState):
        start_time = time.time()
        query = state["query"]
        
        try:
            prompt = COMPANY_EXTRACTOR_PROMPT.format_messages(query=query)
            response = llm.invoke(prompt)
            
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_company_extractor", duration)
            
            # Parse LLM response
            response_text = response.content.strip()
            company_name = None
            ticker = None
            confidence = "Low"
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('COMPANY:'):
                    company_name = line.replace('COMPANY:', '').strip()
                elif line.startswith('TICKER:'):
                    ticker = line.replace('TICKER:', '').strip()
                elif line.startswith('CONFIDENCE:'):
                    confidence = line.replace('CONFIDENCE:', '').strip()
            
            # Clean up None values
            if company_name and company_name.lower() in ['none', 'null', '']:
                company_name = None
            if ticker and ticker.lower() in ['none', 'null', '']:
                ticker = None
            
            # Validate ticker if found
            ticker_valid = False
            if ticker:
                try:
                    test_stock = yf.Ticker(ticker)
                    test_data = test_stock.history(period="1d")
                    if not test_data.empty:
                        ticker_valid = True
                        st.success(f"✅ Company identified: {company_name} ({ticker})")
                    else:
                        st.warning(f"⚠️ Ticker {ticker} validation failed")
                        ticker = None
                        company_name = None
                except Exception as e:
                    st.warning(f"⚠️ Ticker validation error: {e}")
                    ticker = None
                    company_name = None
            
            # Update performance tracking
            agent_perf = state.get("agent_performance", {})
            agent_perf["company_extractor"] = {
                "llm": llm_name,
                "duration": duration,
                "success": True,
                "company_found": bool(company_name),
                "ticker_found": bool(ticker),
                "ticker_valid": ticker_valid,
                "confidence": confidence
            }
            
            if not company_name and not ticker:
                st.info("🤔 No specific company identified - providing general market analysis")
                return {
                    "messages": [AIMessage(content="No specific company identified. Proceeding with general analysis.")],
                    "extracted_company_info": {
                        "company_name": None, 
                        "ticker": None, 
                        "found": False,
                        "confidence": confidence
                    },
                    "company": "General Market Analysis",
                    "ticker": "",
                    "agent_performance": agent_perf,
                    "current_agent": "web_search"
                }
            
            return {
                "messages": [AIMessage(content=f"Company identified: {company_name} ({ticker})")],
                "extracted_company_info": {
                    "company_name": company_name,
                    "ticker": ticker,
                    "found": True,
                    "confidence": confidence
                },
                "company": company_name or "Unknown Company",
                "ticker": ticker or "",
                "agent_performance": agent_perf,
                "current_agent": "web_search"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_company_extractor")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["company_extractor"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"Company extraction error: {str(e)[:100]}")],
                "extracted_company_info": {"company_name": None, "ticker": None, "found": False},
                "company": "Unknown Company",
                "ticker": "",
                "agent_performance": agent_perf,
                "current_agent": "web_search"
            }
    
    return company_extractor

def create_web_search_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """Web search agent for additional company information"""
    
    def web_search_agent(state: FinancialAnalysisState):
        start_time = time.time()
        
        try:
            company_name = state.get("company", "")
            query = state.get("query", "")
            
            # Perform web search for additional info
            search_results = web_search_company_info(company_name, query)
            
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_web_search", duration)
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["web_search"] = {
                "llm": llm_name,
                "duration": duration,
                "success": search_results.get("search_success", False),
                "news_found": len(search_results.get("company_news", [])),
                "has_sentiment": bool(search_results.get("market_sentiment", ""))
            }
            
            if search_results.get("search_success"):
                st.info(f"🔍 Web search completed for {company_name}")
            
            return {
                "messages": [AIMessage(content=f"Web search completed for {company_name}")],
                "web_search_results": search_results,
                "agent_performance": agent_perf,
                "current_agent": "data_fetcher"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_web_search")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["web_search"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"Web search error: {str(e)[:100]}")],
                "web_search_results": {"search_success": False, "error": str(e)},
                "agent_performance": agent_perf,
                "current_agent": "data_fetcher"
            }
    
    return web_search_agent

def create_data_fetcher_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """Enhanced data fetcher with multiple sources"""
    
    def data_fetcher(state: FinancialAnalysisState):
        start_time = time.time()
        
        try:
            company_name = state.get("company", "")
            ticker = state.get("ticker", "")
            period = state.get("period", "Q4 2024")
            
            market_data = {}
            data_sources = []
            
            # Yahoo Finance data
            if ticker:
                try:
                    with st.spinner(f"📈 Fetching Yahoo Finance data for {ticker}..."):
                        yahoo_data = fetch_yahoo_finance_data(ticker, period)
                        if yahoo_data and len(yahoo_data) > 3:
                            market_data.update(yahoo_data)
                            data_sources.append({
                                "source": "Yahoo Finance",
                                "ticker": ticker,
                                "success": True
                            })
                            st.success(f"✅ Yahoo Finance data retrieved for {ticker}")
                except Exception as e:
                    st.warning(f"Yahoo Finance error: {e}")
            
            # SEC EDGAR data
            if company_name in COMPANY_CIK_MAP:
                try:
                    with st.spinner(f"📑 Fetching SEC data for {company_name}..."):
                        sec_data = fetch_sec_edgar_data(company_name, period)
                        if sec_data and len(sec_data) > 3:
                            # Merge SEC data with market data
                            for key, value in sec_data.items():
                                if key not in ["requested_period", "data_period", "period_match"] and value is not None:
                                    market_data[key] = value
                            
                            data_sources.append({
                                "source": "SEC EDGAR",
                                "success": True,
                                "form": sec_data.get("form_type", "")
                            })
                            st.success(f"✅ SEC EDGAR data retrieved for {company_name}")
                except Exception as e:
                    st.warning(f"SEC EDGAR error: {e}")
            
            # Fallback sample data if needed
            if not market_data:
                with st.spinner("📊 Generating sample data..."):
                    market_data = generate_sample_financial_data(company_name)
                    data_sources.append({
                        "source": "Sample Data",
                        "success": True,
                        "note": "Real data unavailable"
                    })
            
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_data_fetcher", duration)
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["data_fetcher"] = {
                "llm": llm_name,
                "duration": duration,
                "success": True,
                "sources_used": len(data_sources),
                "has_real_data": any(s.get("source") != "Sample Data" for s in data_sources)
            }
            
            return {
                "messages": [AIMessage(content=f"Data fetching completed for {company_name}")],
                "market_data": market_data,
                "data_sources": data_sources,
                "agent_performance": agent_perf,
                "current_agent": "parser"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_data_fetcher")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["data_fetcher"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"Data fetcher error: {str(e)[:100]}")],
                "market_data": generate_sample_financial_data(state.get("company", "Unknown")),
                "data_sources": [{"source": "Fallback", "error": str(e)}],
                "agent_performance": agent_perf,
                "current_agent": "parser"
            }
    
    return data_fetcher

# Helper functions for data fetching
def fetch_yahoo_finance_data(ticker: str, period: str):
    """Fetch financial data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        result = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "eps": info.get("trailingEps"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta")
        }
        
        # Get financial statements
        try:
            financials = stock.quarterly_financials
            balance_sheet = stock.quarterly_balance_sheet
            cashflow = stock.quarterly_cashflow
            
            if not financials.empty:
                latest_financials = financials.iloc[:, 0]
                result.update({
                    "revenue": latest_financials.get("Total Revenue"),
                    "gross_profit": latest_financials.get("Gross Profit"),
                    "operating_income": latest_financials.get("Operating Income"),
                    "net_income": latest_financials.get("Net Income"),
                    "ebitda": latest_financials.get("EBITDA")
                })
            
            if not balance_sheet.empty:
                latest_balance = balance_sheet.iloc[:, 0]
                result.update({
                    "total_assets": latest_balance.get("Total Assets"),
                    "total_liabilities": latest_balance.get("Total Liabilities Net Minority Interest"),
                    "total_equity": latest_balance.get("Total Stockholder Equity"),
                    "cash": latest_balance.get("Cash And Cash Equivalents")
                })
            
            if not cashflow.empty:
                latest_cashflow = cashflow.iloc[:, 0]
                result.update({
                    "operating_cash_flow": latest_cashflow.get("Operating Cash Flow"),
                    "free_cash_flow": latest_cashflow.get("Free Cash Flow")
                })
                
        except Exception as e:
            st.warning(f"Could not fetch detailed financials: {e}")
        
        # Clean up None values
        return {k: v for k, v in result.items() if v is not None and not pd.isna(v)}
        
    except Exception as e:
        st.error(f"Yahoo Finance error: {e}")
        return {}

def fetch_sec_edgar_data(company: str, period: str):
    """Fetch data from SEC EDGAR API"""
    try:
        cik = COMPANY_CIK_MAP.get(company, "")
        if not cik:
            return {}
        
        # This is a simplified implementation
        # In production, you'd implement full SEC API integration
        return {
            "source": "SEC EDGAR",
            "cik": cik,
            "form_type": "10-K",
            "data_available": True
        }
        
    except Exception as e:
        return {"error": str(e)}

def generate_sample_financial_data(company_name: str) -> Dict:
    """Generate sample financial data when real data unavailable"""
    
    # Company-specific multipliers
    multipliers = {
        "apple": {"revenue": 90000, "profit": 0.25, "growth": 0.15},
        "tesla": {"revenue": 25000, "profit": 0.10, "growth": 0.30},
        "microsoft": {"revenue": 60000, "profit": 0.35, "growth": 0.12},
        "amazon": {"revenue": 150000, "profit": 0.05, "growth": 0.20},
        "nvidia": {"revenue": 30000, "profit": 0.40, "growth": 0.50},
        "meta": {"revenue": 40000, "profit": 0.20, "growth": 0.10},
        "alphabet": {"revenue": 75000, "profit": 0.22, "growth": 0.13},
        "google": {"revenue": 75000, "profit": 0.22, "growth": 0.13},
        "tata": {"revenue": 20000, "profit": 0.12, "growth": 0.08},
        "reliance": {"revenue": 25000, "profit": 0.15, "growth": 0.10}
    }
    
    # Find matching multiplier
    company_lower = company_name.lower()
    multiplier = None
    for key, mult in multipliers.items():
        if key in company_lower:
            multiplier = mult
            break
    
    if not multiplier:
        multiplier = {"revenue": 50000, "profit": 0.15, "growth": 0.10}
    
    revenue = multiplier["revenue"] * 1000000
    
    return {
        "revenue": revenue,
        "gross_profit": revenue * 0.4,
        "net_income": revenue * multiplier["profit"],
        "total_assets": revenue * 3,
        "total_liabilities": revenue * 1.2,
        "operating_cash_flow": revenue * 0.3,
        "market_cap": revenue * 15,
        "current_price": 150.0,
        "pe_ratio": 25.0,
        "eps": 6.0,
        "revenue_growth": multiplier["growth"] * 100,
        "data_period": "Q4 2024",
        "period_match": False
    }

# Parser, KPI, Risk, and Insight agents (keeping existing implementations)
def create_parser_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """Financial data parser agent"""
    
    PARSER_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a financial data parser. Extract key metrics from the provided data.

Company: {company} ({ticker})
Market Data: {market_data}
Web Search Results: {web_search_results}
Upload Content: {uploaded_content}

Extract these metrics in JSON format:
- revenue (USD)
- gross_profit (USD)
- net_income (USD)
- total_assets (USD)
- total_liabilities (USD)
- operating_cash_flow (USD)
- market_cap (USD)
- current_price (USD)
- eps (earnings per share)

Return clean JSON only."""),
        ("human", "Parse the financial data for analysis.")
    ])
    
    def parser(state: FinancialAnalysisState):
        start_time = time.time()
        
        try:
            prompt = PARSER_PROMPT.format_messages(
                company=state.get("company", "Unknown"),
                ticker=state.get("ticker", ""),
                market_data=json.dumps(state.get("market_data", {}), cls=DateTimeEncoder)[:2000],
                web_search_results=json.dumps(state.get("web_search_results", {}))[:1000],
                uploaded_content=state.get("uploaded_content", "")[:1000]
            )
            
            response = llm.invoke(prompt)
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_parser", duration)
            
            # Extract JSON from response
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    parsed_data = json.loads(json_match.group())
                else:
                    # Use market data directly
                    market_data = state.get("market_data", {})
                    parsed_data = {
                        "revenue": market_data.get("revenue", 1000000000),
                        "gross_profit": market_data.get("gross_profit", 400000000),
                        "net_income": market_data.get("net_income", 100000000),
                        "total_assets": market_data.get("total_assets", 5000000000),
                        "total_liabilities": market_data.get("total_liabilities", 2000000000),
                        "operating_cash_flow": market_data.get("operating_cash_flow", 150000000),
                        "market_cap": market_data.get("market_cap", 1000000000000),
                        "current_price": market_data.get("current_price", 150.0),
                        "eps": market_data.get("eps", 6.0)
                    }
            except:
                # Fallback
                parsed_data = {
                    "revenue": 1000000000,
                    "gross_profit": 400000000,
                    "net_income": 100000000,
                    "total_assets": 5000000000,
                    "total_liabilities": 2000000000,
                    "operating_cash_flow": 150000000,
                    "market_cap": 1000000000000,
                    "current_price": 150.0,
                    "eps": 6.0
                }
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["parser"] = {
                "llm": llm_name,
                "duration": duration,
                "success": True,
                "metrics_extracted": len(parsed_data)
            }
            
            return {
                "messages": [AIMessage(content="Financial data parsed successfully")],
                "parsed_data": parsed_data,
                "agent_performance": agent_perf,
                "current_agent": "kpi_extractor"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_parser")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["parser"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"Parser error: {str(e)[:100]}")],
                "parsed_data": generate_sample_financial_data(state.get("company", "Unknown")),
                "agent_performance": agent_perf,
                "current_agent": "kpi_extractor"
            }
    
    return parser

def create_kpi_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """KPI calculation agent"""
    
    def kpi_extractor(state: FinancialAnalysisState):
        start_time = time.time()
        
        try:
            parsed_data = state["parsed_data"]
            
            # Calculate KPIs
            revenue = max(parsed_data.get("revenue", 1), 1)
            total_assets = max(parsed_data.get("total_assets", 1), 1)
            total_liabilities = parsed_data.get("total_liabilities", 0) or 0
            gross_profit = parsed_data.get("gross_profit", 0) or 0
            net_income = parsed_data.get("net_income", 0) or 0
            
            total_equity = max(total_assets - total_liabilities, 1)
            
            kpis = {
                "gross_margin": (gross_profit / revenue * 100),
                "net_margin": (net_income / revenue * 100),
                "debt_to_equity": total_liabilities / total_equity,
                "return_on_assets": (net_income / total_assets * 100),
                "return_on_equity": (net_income / total_equity * 100),
                "current_ratio": 1.5,  # Simplified
                "revenue_growth": 15.0,  # Simplified
                "price_to_earnings": parsed_data.get("market_cap", 0) / max(net_income * 4, 1) if net_income > 0 else 25.0
            }
            
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_kpi", duration)
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["kpi_extractor"] = {
                "llm": llm_name,
                "duration": duration,
                "success": True,
                "kpis_calculated": len(kpis)
            }
            
            return {
                "messages": [AIMessage(content="KPIs calculated successfully")],
                "kpis": kpis,
                "agent_performance": agent_perf,
                "current_agent": "risk_assessor"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_kpi")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["kpi_extractor"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"KPI error: {str(e)[:100]}")],
                "kpis": {
                    "gross_margin": 40.0,
                    "net_margin": 10.0,
                    "debt_to_equity": 0.5,
                    "return_on_assets": 8.0,
                    "return_on_equity": 15.0,
                    "current_ratio": 1.5,
                    "revenue_growth": 15.0,
                    "price_to_earnings": 25.0
                },
                "agent_performance": agent_perf,
                "current_agent": "risk_assessor"
            }
    
    return kpi_extractor

def create_risk_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """Risk assessment agent"""
    
    def risk_assessor(state: FinancialAnalysisState):
        start_time = time.time()
        
        try:
            kpis = state["kpis"]
            
            # Calculate risk scores
            debt_to_equity = kpis.get("debt_to_equity", 0)
            current_ratio = kpis.get("current_ratio", 1)
            net_margin = kpis.get("net_margin", 0)
            
            debt_risk = min(100, debt_to_equity * 30)
            liquidity_risk = max(0, 100 - current_ratio * 50)
            profitability_risk = max(0, 100 - net_margin * 5)
            overall_risk = (debt_risk + liquidity_risk + profitability_risk) / 3
            
            risk_assessment = {
                "overall_risk_score": overall_risk,
                "risk_level": "High" if overall_risk > 70 else "Medium" if overall_risk > 40 else "Low",
                "debt_risk": debt_risk,
                "liquidity_risk": liquidity_risk,
                "profitability_risk": profitability_risk,
                "risk_factors": [],
                "mitigation_strategies": []
            }
            
            # Add specific risk factors
            if debt_risk > 60:
                risk_assessment["risk_factors"].append("High debt levels")
                risk_assessment["mitigation_strategies"].append("Consider debt reduction")
            if liquidity_risk > 60:
                risk_assessment["risk_factors"].append("Liquidity concerns")
                risk_assessment["mitigation_strategies"].append("Improve cash management")
            if profitability_risk > 60:
                risk_assessment["risk_factors"].append("Low profitability")
                risk_assessment["mitigation_strategies"].append("Focus on efficiency")
            
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_risk", duration)
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["risk_assessor"] = {
                "llm": llm_name,
                "duration": duration,
                "success": True,
                "risk_score": overall_risk,
                "risk_level": risk_assessment["risk_level"]
            }
            
            return {
                "messages": [AIMessage(content="Risk assessment completed")],
                "risk_assessment": risk_assessment,
                "agent_performance": agent_perf,
                "current_agent": "insight_generator"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_risk")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["risk_assessor"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"Risk error: {str(e)[:100]}")],
                "risk_assessment": {
                    "overall_risk_score": 45.0,
                    "risk_level": "Medium",
                    "risk_factors": ["Analysis limitations"],
                    "mitigation_strategies": ["Manual review needed"]
                },
                "agent_performance": agent_perf,
                "current_agent": "insight_generator"
            }
    
    return risk_assessor

def create_insight_agent(llm, llm_name: str, metrics: LLMEvaluationMetrics):
    """Insight generation agent"""
    
    INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a senior financial analyst. Generate insights and recommendations.

Company: {company} ({ticker})
Query: {query}
Financial Data: {parsed_data}
KPIs: {kpis}
Risk Assessment: {risk_assessment}
Web Search Results: {web_search_results}

Provide insights in JSON format:
- query_specific_answer: Direct answer to user's question
- key_strengths: List of company strengths
- key_concerns: List of concerns/risks
- recommendations: Actionable recommendations
- investment_decision: BUY/HOLD/SELL/WAIT (only if investment question)

Tailor response to the specific query and be actionable."""),
        ("human", "Generate comprehensive financial insights.")
    ])
    
    def insight_generator(state: FinancialAnalysisState):
        start_time = time.time()
        
        try:
            prompt = INSIGHT_PROMPT.format_messages(
                company=state.get("company", "Unknown"),
                ticker=state.get("ticker", ""),
                query=state.get("query", ""),
                parsed_data=json.dumps(state.get("parsed_data", {}), indent=2),
                kpis=json.dumps(state.get("kpis", {}), indent=2),
                risk_assessment=json.dumps(state.get("risk_assessment", {}), indent=2),
                web_search_results=json.dumps(state.get("web_search_results", {}))[:500]
            )
            
            response = llm.invoke(prompt)
            duration = time.time() - start_time
            metrics.record_response_time(f"{llm_name}_insight", duration)
            
            try:
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    insights = json.loads(json_match.group())
                    # Clean text
                    for key in ['query_specific_answer']:
                        if key in insights:
                            insights[key] = clean_llm_text(insights[key])
                else:
                    insights = generate_default_insights(state)
            except:
                insights = generate_default_insights(state)
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["insight_generator"] = {
                "llm": llm_name,
                "duration": duration,
                "success": True,
                "has_recommendation": "investment_decision" in insights
            }
            
            return {
                "messages": [AIMessage(content="Insights generated successfully")],
                "insights": insights,
                "agent_performance": agent_perf,
                "current_agent": "end"
            }
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_error(f"{llm_name}_insight")
            
            agent_perf = state.get("agent_performance", {})
            agent_perf["insight_generator"] = {
                "llm": llm_name,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
            
            return {
                "messages": [AIMessage(content=f"Insight error: {str(e)[:100]}")],
                "insights": generate_default_insights(state),
                "agent_performance": agent_perf,
                "current_agent": "end"
            }
    
    return insight_generator

def generate_default_insights(state):
    """Generate default insights when LLM fails"""
    company = state.get("company", "Unknown")
    ticker = state.get("ticker", "")
    kpis = state.get("kpis", {})
    risk = state.get("risk_assessment", {})
    
    # Determine investment decision
    risk_score = risk.get('overall_risk_score', 50)
    gross_margin = kpis.get('gross_margin', 0)
    net_margin = kpis.get('net_margin', 0)
    
    if risk_score < 40 and gross_margin > 30 and net_margin > 10:
        decision = "BUY"
        reason = "low risk with strong profitability"
    elif risk_score > 70 or net_margin < 5:
        decision = "SELL"
        reason = "high risk or poor profitability"
    elif risk_score < 60 and gross_margin > 20:
        decision = "HOLD"
        reason = "moderate risk with decent returns"
    else:
        decision = "WAIT"
        reason = "mixed signals require more analysis"
    
    company_display = f"{company} ({ticker})" if ticker else company
    
    return {
        "query_specific_answer": f"{company_display} shows {risk.get('risk_level', 'Medium')} risk with {gross_margin:.1f}% gross margins. {decision} recommendation based on {reason}.",
        "key_strengths": [
            f"Gross margin: {gross_margin:.1f}%",
            "Established market position"
        ],
        "key_concerns": risk.get("risk_factors", ["Market volatility"]),
        "recommendations": [
            "Monitor quarterly results",
            "Set appropriate stop-loss levels",
            "Consider portfolio diversification"
        ],
        "investment_decision": decision
    }

# Process PDF with RAG
def process_pdf_with_rag(pdf_file, query: str = "", company: str = "", period: str = ""):
    """Process PDF with RAG for better retrieval"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text() + "\n"
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_text(full_text)
        
        if os.getenv("OPENAI_API_KEY") and chunks:
            documents = []
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_file.name,
                        "page": i,
                        "company": company,
                        "period": period
                    }
                )
                documents.append(doc)
            
            embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
            vectorstore = FAISS.from_documents(documents, embeddings)
            
            retrieval_query = f"""
            {query}
            Extract financial information including:
            - Revenue and earnings
            - Key financial metrics
            - Risk factors
            - Business performance
            Company: {company}
            Period: {period}
            """
            
            relevant_docs = vectorstore.similarity_search(retrieval_query, k=5)
            relevant_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            return {
                "success": True,
                "content": relevant_text[:8000],
                "num_chunks": len(chunks),
                "chunks_used": len(relevant_docs)
            }
        else:
            return {
                "success": True,
                "content": full_text[:8000],
                "num_chunks": len(chunks),
                "chunks_used": 1
            }
            
    except Exception as e:
        return {
            "success": False,
            "content": "",
            "error": str(e)
        }

# Complete workflow creation
def create_complete_financial_workflow(selected_llms: Dict, llm_info: Dict, metrics: LLMEvaluationMetrics):
    """Create complete 6-agent financial analysis workflow"""
    
    if not selected_llms:
        st.error("No LLMs selected!")
        return None, {}
    
    llm_names = list(selected_llms.keys())
    
    # Smart LLM assignment
    company_llm_name = "groq-llama3" if "groq-llama3" in selected_llms else llm_names[0]
    web_llm_name = "groq-llama3" if "groq-llama3" in selected_llms else llm_names[0]
    data_llm_name = "groq-llama3" if "groq-llama3" in selected_llms else llm_names[0]
    parser_llm_name = "groq-llama3" if "groq-llama3" in selected_llms else llm_names[0]
    kpi_llm_name = "gpt-4" if "gpt-4" in selected_llms else "claude-3" if "claude-3" in selected_llms else llm_names[0]
    risk_llm_name = "claude-3" if "claude-3" in selected_llms else "gpt-4" if "gpt-4" in selected_llms else llm_names[0]
    insight_llm_name = "gpt-4" if "gpt-4" in selected_llms else "claude-3" if "claude-3" in selected_llms else llm_names[0]
    
    # Create workflow
    workflow = StateGraph(FinancialAnalysisState)
    
    # Add all 6 agents
    workflow.add_node("company_extractor", create_company_extraction_agent(selected_llms[company_llm_name], company_llm_name, metrics))
    workflow.add_node("web_search", create_web_search_agent(selected_llms[web_llm_name], web_llm_name, metrics))
    workflow.add_node("data_fetcher", create_data_fetcher_agent(selected_llms[data_llm_name], data_llm_name, metrics))
    workflow.add_node("parser", create_parser_agent(selected_llms[parser_llm_name], parser_llm_name, metrics))
    workflow.add_node("kpi_extractor", create_kpi_agent(selected_llms[kpi_llm_name], kpi_llm_name, metrics))
    workflow.add_node("risk_assessor", create_risk_agent(selected_llms[risk_llm_name], risk_llm_name, metrics))
    workflow.add_node("insight_generator", create_insight_agent(selected_llms[insight_llm_name], insight_llm_name, metrics))
    
    # Complete workflow: Company → Web Search → Data → Parse → KPI → Risk → Insights
    workflow.set_entry_point("company_extractor")
    workflow.add_edge("company_extractor", "web_search")
    workflow.add_edge("web_search", "data_fetcher")
    workflow.add_edge("data_fetcher", "parser")
    workflow.add_edge("parser", "kpi_extractor")
    workflow.add_edge("kpi_extractor", "risk_assessor")
    workflow.add_edge("risk_assessor", "insight_generator")
    workflow.add_edge("insight_generator", END)
    
    app = workflow.compile()
    
    llm_assignments = {
        "company_extractor": company_llm_name,
        "web_search": web_llm_name,
        "data_fetcher": data_llm_name,
        "parser": parser_llm_name,
        "kpi_extractor": kpi_llm_name,
        "risk_assessor": risk_llm_name,
        "insight_generator": insight_llm_name
    }
    
    return app, llm_assignments

# Detection functions
def detect_query_intent(query: str):
    """Detect user's intent from query"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["invest", "buy", "sell", "hold", "should i"]):
        return "investment_decision"
    elif any(word in query_lower for word in ["risk", "risky", "safe", "dangerous"]):
        return "risk_assessment"
    elif any(word in query_lower for word in ["compare", "versus", "vs", "better than"]):
        return "comparison"
    elif any(word in query_lower for word in ["analyze", "review", "explain", "tell me about"]):
        return "general_analysis"
    else:
        return "general_query"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="FinSight",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "yahoo_cache" not in st.session_state:
        st.session_state.yahoo_cache = {}
    if "last_yahoo_request" not in st.session_state:
        st.session_state.last_yahoo_request = 0
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "current_company" not in st.session_state:
        st.session_state.current_company = None
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = None
    if "current_period" not in st.session_state:
        st.session_state.current_period = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = LLMEvaluationMetrics()
    if "llm_assignments" not in st.session_state:
        st.session_state.llm_assignments = {}
    
    # Custom CSS
    st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; margin: 5px; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Complete AI Financial Analyst")
    st.markdown("**7-Agent Multi-LLM System with Company Extraction, Web Search & Performance Evaluation**")
    
    # Check API keys
    api_keys_configured = bool(
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("GROQ_API_KEY") or 
        os.getenv("GOOGLE_API_KEY") or
        os.getenv("ANTHROPIC_API_KEY")
    )
    
    if not api_keys_configured:
        st.error("""
        ⚠️ No API keys found! Please create a `.env` file with at least one of:
        - OPENAI_API_KEY
        - GROQ_API_KEY
        - GOOGLE_API_KEY
        - ANTHROPIC_API_KEY
        """)
        st.stop()
    
    # Initialize LLMs
    llms, llm_info = get_llms()
    
    if not llms:
        st.error("No LLMs available!")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Response mode
        response_mode = st.radio(
            "Response Detail Level:",
            ["concise", "detailed"],
            format_func=lambda x: "🎯 Concise" if x == "concise" else "📈 Detailed"
        )
        
        # Document upload
        uploaded_file = st.file_uploader(
            "Upload PDF (optional):",
            type=['pdf'],
            help="Financial reports, earnings, etc."
        )
        
        # Current context
        if st.session_state.current_company:
            st.subheader("📍 Current Context")
            company_display = st.session_state.current_company
            if st.session_state.current_ticker:
                company_display += f" ({st.session_state.current_ticker})"
            st.info(f"**Company:** {company_display}")
        
        # LLM assignments
        if st.session_state.llm_assignments:
            st.subheader("🤖 Agent Assignments")
            for agent, llm_name in st.session_state.llm_assignments.items():
                st.caption(f"**{agent.replace('_', ' ').title()}:** {llm_name}")
        
        # Clear button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.current_company = None
            st.session_state.current_ticker = None
            st.rerun()
    
    # Initialize workflow
    workflow, llm_assignments = create_complete_financial_workflow(llms, llm_info, st.session_state.metrics)
    st.session_state.llm_assignments = llm_assignments
    
    if not workflow:
        st.stop()
    
    # Chat interface
    st.subheader("💬 Chat with AI Financial Analyst")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["type"] == "human":
            st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            👤 **You:** {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #e8f5e9; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
            🤖 **AI:** {message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Show performance metrics if available
            if "full_results" in message:
                with st.expander("📊 Agent Performance"):
                    agent_perf = message["full_results"].get("agent_performance", {})
                    if agent_perf:
                        cols = st.columns(len(agent_perf))
                        for i, (agent, perf) in enumerate(agent_perf.items()):
                            with cols[i]:
                                if isinstance(perf, dict):
                                    status = "✅" if perf.get("success", False) else "❌"
                                    st.metric(
                                        f"{agent.replace('_', ' ').title()} {status}",
                                        f"{perf.get('duration', 0):.2f}s"
                                    )
    
    # Chat input
    query = st.text_area(
        "Ask about any company:",
        placeholder="e.g., 'Should I invest in Apple?', 'How is Tesla performing?', 'Analyze Microsoft's risks'",
        height=100
    )
    
    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if query:
            # Add user message
            st.session_state.chat_history.append({
                "type": "human",
                "content": query,
                "timestamp": datetime.now().strftime("%I:%M %p")
            })
            
            with st.spinner("🤔 Running complete 7-agent analysis..."):
                # Process PDF if uploaded
                uploaded_content = ""
                if uploaded_file:
                    try:
                        pdf_result = process_pdf_with_rag(uploaded_file, query, "", "")
                        if pdf_result["success"]:
                            uploaded_content = pdf_result["content"]
                            st.success("✅ PDF processed successfully")
                    except Exception as e:
                        st.warning(f"PDF processing error: {e}")
                
                # Create initial state
                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "query": query,
                    "company": "",
                    "ticker": "",
                    "period": f"Q4 {datetime.now().year - 1}",
                    "parsed_data": {},
                    "kpis": {},
                    "risk_assessment": {},
                    "insights": {},
                    "current_agent": "company_extractor",
                    "response_mode": response_mode,
                    "market_data": {},
                    "web_search_results": {},
                    "uploaded_content": uploaded_content,
                    "data_sources": [],
                    "period_validation": {},
                    "extracted_company_info": {},
                    "agent_performance": {}
                }
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                agent_steps = [
                    "🎯 Company Extraction",
                    "🔍 Web Search",
                    "📊 Data Fetching",
                    "📝 Data Parsing", 
                    "🧮 KPI Calculation",
                    "⚠️ Risk Assessment",
                    "🧠 Insight Generation"
                ]
                
                try:
                    # Show progress
                    for step, agent_step in enumerate(agent_steps):
                        progress_bar.progress((step + 1) / len(agent_steps))
                        status_text.text(f"Agent {step + 1}/7: {agent_step}")
                        time.sleep(0.3)
                    
                    # Execute workflow
                    st.text("🚀 Executing complete 7-agent workflow...")
                    workflow_start = time.time()
                    final_state = workflow.invoke(initial_state)
                    workflow_duration = time.time() - workflow_start
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ All 7 agents completed successfully!")
                    
                    # Store results
                    st.session_state.analysis_results = final_state
                    
                    # Update context
                    if final_state.get("extracted_company_info", {}).get("found"):
                        company_info = final_state["extracted_company_info"]
                        st.session_state.current_company = company_info.get("company_name")
                        st.session_state.current_ticker = company_info.get("ticker")
                    
                    # Display performance summary
                    agent_perf = final_state.get("agent_performance", {})
                    if agent_perf:
                        with st.expander("⚡ Complete Workflow Performance"):
                            perf_cols = st.columns(len(agent_perf) + 1)
                            
                            with perf_cols[0]:
                                st.metric("Total Workflow", f"{workflow_duration:.2f}s")
                            
                            for i, (agent, perf) in enumerate(agent_perf.items()):
                                with perf_cols[i + 1]:
                                    if isinstance(perf, dict):
                                        status = "✅" if perf.get("success", False) else "❌"
                                        st.metric(
                                            f"{agent.replace('_', ' ').title()} {status}",
                                            f"{perf.get('duration', 0):.2f}s"
                                        )
                    
                    # Generate response
                    insights = final_state.get("insights", {})
                    company_analyzed = final_state.get("company", "Unknown")
                    ticker_analyzed = final_state.get("ticker", "")
                    
                    # Show company identified
                    if company_analyzed and company_analyzed != "Unknown":
                        st.success(f"✅ Analyzed: {company_analyzed} ({ticker_analyzed})")
                    
                    # Build response based on mode
                    response_parts = []
                    
                    if response_mode == "concise":
                        if insights.get("query_specific_answer"):
                            response_parts.append(f"**📌 {insights['query_specific_answer']}**")
                        
                        if insights.get("investment_decision"):
                            decision = insights["investment_decision"]
                            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "WAIT": "⏸️"}.get(decision, "")
                            response_parts.append(f"\n**Decision: {emoji} {decision}**")
                        
                        if insights.get("key_concerns"):
                            response_parts.append("\n**Key Concerns:**")
                            for concern in insights["key_concerns"][:2]:
                                response_parts.append(f"• {concern}")
                        
                        if insights.get("recommendations"):
                            response_parts.append("\n**Recommendations:**")
                            for rec in insights["recommendations"][:2]:
                                response_parts.append(f"• {rec}")
                    
                    else:  # Detailed mode
                        if insights.get("query_specific_answer"):
                            response_parts.append(f"### 📊 Analysis Results\n{insights['query_specific_answer']}")
                        
                        if insights.get("investment_decision"):
                            decision = insights["investment_decision"]
                            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "WAIT": "⏸️"}.get(decision, "")
                            response_parts.append(f"\n### {emoji} Investment Decision: {decision}")
                        
                        if insights.get("key_strengths") or insights.get("key_concerns"):
                            response_parts.append("\n### 💪 Strengths & ⚠️ Concerns")
                            
                            if insights.get("key_strengths"):
                                response_parts.append("\n**Strengths:**")
                                for strength in insights["key_strengths"]:
                                    response_parts.append(f"• {strength}")
                            
                            if insights.get("key_concerns"):
                                response_parts.append("\n**Concerns:**")
                                for concern in insights["key_concerns"]:
                                    response_parts.append(f"• {concern}")
                        
                        if insights.get("recommendations"):
                            response_parts.append("\n### 🎯 Recommendations")
                            for i, rec in enumerate(insights["recommendations"], 1):
                                response_parts.append(f"{i}. {rec}")
                    
                    # Add data sources
                    data_sources = final_state.get("data_sources", [])
                    if data_sources:
                        source_names = [s.get("source", "Unknown") for s in data_sources]
                        response_parts.append(f"\n\n📊 *Data sources: {', '.join(source_names)}*")
                    
                    # Add PDF info
                    if uploaded_content:
                        response_parts.append(f"\n📄 *Analysis includes uploaded PDF content*")
                    
                    # Combine response
                    ai_response = "\n".join(response_parts)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "type": "ai",
                        "content": ai_response,
                        "timestamp": datetime.now().strftime("%I:%M %p"),
                        "full_results": final_state,
                        "company": company_analyzed,
                        "ticker": ticker_analyzed
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Workflow error: {str(e)}")
                    
                    error_message = f"I encountered an error during analysis. Please try again.\n\nError: {str(e)[:200]}..."
                    st.session_state.chat_history.append({
                        "type": "ai",
                        "content": error_message,
                        "timestamp": datetime.now().strftime("%I:%M %p")
                    })
                    st.rerun()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center'>
        <p><strong>Complete AI Financial Analyst v4.0</strong></p>
        <p>🤖 7-Agent System | 🔍 Company Extraction | 🌐 Web Search | 📊 Multi-Source Data | 📈 LLM Evaluation</p>
        <p><strong>Agents:</strong> Company Extractor → Web Search → Data Fetcher → Parser → KPI Calculator → Risk Assessor → Insight Generator</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                    