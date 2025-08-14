"""
Shariah Compliant Stocks Database

This module contains a curated list of publicly traded companies that are
generally considered Shariah compliant based on AAOIFI standards and Islamic
finance principles. Companies are selected based on:

1. Business Activities: No involvement in alcohol, gambling, tobacco, pork,
   interest-based banking, adult entertainment, or weapons manufacturing
2. Financial Ratios: Debt-to-equity and interest income ratios within Islamic limits
3. Ethical Standards: Business practices aligned with Islamic values

Sources: AAOIFI guidelines, Zoya Finance, Islamicly, and other halal investment platforms.

IMPORTANT: This list is for educational purposes. For actual investing, always:
- Use certified Shariah screening services (Zoya, Islamicly, Musaffa)
- Verify current compliance status (companies can lose compliance)
- Consult with Islamic finance scholars
- Purify dividends if required
"""

def get_shariah_compliant_stocks():
    """
    Returns a dictionary of Shariah compliant stocks with their information
    
    Returns:
        dict: Dictionary with stock symbols as keys and company info as values
    """
    
    shariah_stocks = {
        # Technology Companies - AAOIFI Approved
        'AAPL': {
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'market_cap': '$3.0T+',
            'description': 'Consumer electronics and software - AAOIFI compliant'
        },
        'NVDA': {
            'name': 'NVIDIA Corporation',
            'sector': 'Technology',
            'market_cap': '$1.8T+',
            'description': 'Graphics processing and AI chips - semiconductor manufacturing'
        },
        'TSLA': {
            'name': 'Tesla Inc.',
            'sector': 'Automotive/Technology',
            'market_cap': '$800B+',
            'description': 'Electric vehicles and clean energy - halal business model'
        },
        'ORCL': {
            'name': 'Oracle Corporation',
            'sector': 'Technology',
            'market_cap': '$300B+',
            'description': 'Database software and cloud infrastructure'
        },
        'ADBE': {
            'name': 'Adobe Inc.',
            'sector': 'Technology',
            'market_cap': '$200B+',
            'description': 'Creative software and digital solutions'
        },
        'AMD': {
            'name': 'Advanced Micro Devices',
            'sector': 'Technology',
            'market_cap': '$200B+',
            'description': 'Semiconductor processors and graphics'
        },
        'INTC': {
            'name': 'Intel Corporation',
            'sector': 'Technology',
            'market_cap': '$150B+',
            'description': 'Semiconductor chip manufacturing'
        },
        'QCOM': {
            'name': 'Qualcomm Inc.',
            'sector': 'Technology',
            'market_cap': '$180B+',
            'description': 'Mobile chip technology and wireless communications'
        },
        'AVGO': {
            'name': 'Broadcom Inc.',
            'sector': 'Technology',
            'market_cap': '$600B+',
            'description': 'Semiconductor and infrastructure software'
        },
        
        # Healthcare & Pharmaceuticals - Halal Business
        'LLY': {
            'name': 'Eli Lilly and Company',
            'sector': 'Healthcare',
            'market_cap': '$700B+',
            'description': 'Diabetes and obesity medications - pharmaceutical research'
        },
        'ABT': {
            'name': 'Abbott Laboratories',
            'sector': 'Healthcare',
            'market_cap': '$180B+',
            'description': 'Medical devices and diagnostics'
        },
        'AZN': {
            'name': 'AstraZeneca PLC',
            'sector': 'Healthcare',
            'market_cap': '$200B+',
            'description': 'Oncology and rare disease treatments'
        },
        'SNY': {
            'name': 'Sanofi S.A.',
            'sector': 'Healthcare',
            'market_cap': '$120B+',
            'description': 'Global pharmaceutical company'
        },
        'TMO': {
            'name': 'Thermo Fisher Scientific',
            'sector': 'Healthcare',
            'market_cap': '$200B+',
            'description': 'Scientific instruments and laboratory equipment'
        },
        'DXCM': {
            'name': 'DexCom Inc.',
            'sector': 'Healthcare',
            'market_cap': '$30B+',
            'description': 'Continuous glucose monitoring systems'
        },
        
        # Industrial & Manufacturing - Halal Operations
        'HD': {
            'name': 'The Home Depot Inc.',
            'sector': 'Retail/Industrial',
            'market_cap': '$350B+',
            'description': 'Home improvement retail - building materials'
        },
        'CAT': {
            'name': 'Caterpillar Inc.',
            'sector': 'Industrial',
            'market_cap': '$150B+',
            'description': 'Construction and mining equipment manufacturing'
        },
        'HON': {
            'name': 'Honeywell International',
            'sector': 'Industrial',
            'market_cap': '$150B+',
            'description': 'Aerospace and building technologies'
        },
        'ASML': {
            'name': 'ASML Holding N.V.',
            'sector': 'Technology/Industrial',
            'market_cap': '$300B+',
            'description': 'Semiconductor equipment manufacturing'
        },
        'SE': {
            'name': 'Schneider Electric',
            'sector': 'Industrial',
            'market_cap': '$80B+',
            'description': 'Energy management and automation'
        },
        
        # Consumer Goods - Strictly Halal
        'COST': {
            'name': 'Costco Wholesale Corporation',
            'sector': 'Retail',
            'market_cap': '$400B+',
            'description': 'Membership-only warehouse club'
        },
        'NKE': {
            'name': 'Nike Inc.',
            'sector': 'Consumer Goods',
            'market_cap': '$150B+',
            'description': 'Athletic footwear and apparel'
        },
        'LULU': {
            'name': 'Lululemon Athletica',
            'sector': 'Consumer Goods',
            'market_cap': '$40B+',
            'description': 'Athletic and yoga apparel'
        },
        'ETSY': {
            'name': 'Etsy Inc.',
            'sector': 'E-commerce',
            'market_cap': '$6B+',
            'description': 'Online marketplace for handmade and vintage items'
        },
        
        # Clean Energy & Utilities - Halal Business Model
        'NEE': {
            'name': 'NextEra Energy Inc.',
            'sector': 'Utilities',
            'market_cap': '$150B+',
            'description': 'Electric utilities and renewable energy'
        },
        'ENPH': {
            'name': 'Enphase Energy Inc.',
            'sector': 'Clean Energy',
            'market_cap': '$15B+',
            'description': 'Solar energy technology and microinverters'
        },
        'SEDG': {
            'name': 'SolarEdge Technologies',
            'sector': 'Clean Energy',
            'market_cap': '$5B+',
            'description': 'Solar power optimization and monitoring'
        },
        
        # Telecommunications - Halal Services
        'T': {
            'name': 'AT&T Inc.',
            'sector': 'Telecommunications',
            'market_cap': '$150B+',
            'description': 'Telecommunications and wireless services'
        },
        'VZ': {
            'name': 'Verizon Communications',
            'sector': 'Telecommunications',
            'market_cap': '$180B+',
            'description': 'Wireless and broadband communications'
        },
        
        # Software & Services - Compliant Operations
        'SHOP': {
            'name': 'Shopify Inc.',
            'sector': 'Technology/E-commerce',
            'market_cap': '$80B+',
            'description': 'E-commerce platform and merchant services'
        },
        'ZM': {
            'name': 'Zoom Video Communications',
            'sector': 'Technology',
            'market_cap': '$20B+',
            'description': 'Video conferencing and communications software'
        },
        'DDOG': {
            'name': 'Datadog Inc.',
            'sector': 'Technology',
            'market_cap': '$35B+',
            'description': 'Cloud monitoring and analytics platform'
        },
        
        # Food & Agriculture - Halal Certified
        'ADM': {
            'name': 'Archer-Daniels-Midland Company',
            'sector': 'Agriculture',
            'market_cap': '$30B+',
            'description': 'Agricultural processing and food ingredients'
        },
        'MDLZ': {
            'name': 'Mondelez International',
            'sector': 'Food & Beverages',
            'market_cap': '$90B+',
            'description': 'Global snacks and confectionery (halal-certified products)'
        }
    }
    
    return shariah_stocks

def is_stock_shariah_compliant(symbol):
    """
    Check if a stock symbol is in our Shariah compliant list
    
    Args:
        symbol (str): Stock symbol to check
        
    Returns:
        bool: True if the stock is considered Shariah compliant
    """
    shariah_stocks = get_shariah_compliant_stocks()
    return symbol.upper() in shariah_stocks

def get_stock_info(symbol):
    """
    Get information about a specific Shariah compliant stock
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Stock information or None if not found
    """
    shariah_stocks = get_shariah_compliant_stocks()
    return shariah_stocks.get(symbol.upper())

def get_stocks_by_sector(sector):
    """
    Get all Shariah compliant stocks in a specific sector
    
    Args:
        sector (str): Sector name
        
    Returns:
        dict: Dictionary of stocks in the specified sector
    """
    shariah_stocks = get_shariah_compliant_stocks()
    sector_stocks = {}
    
    for symbol, info in shariah_stocks.items():
        if sector.lower() in info['sector'].lower():
            sector_stocks[symbol] = info
    
    return sector_stocks

def get_all_sectors():
    """
    Get list of all sectors represented in our Shariah compliant stocks
    
    Returns:
        list: List of unique sectors
    """
    shariah_stocks = get_shariah_compliant_stocks()
    sectors = set()
    
    for info in shariah_stocks.values():
        sectors.add(info['sector'])
    
    return sorted(list(sectors))
