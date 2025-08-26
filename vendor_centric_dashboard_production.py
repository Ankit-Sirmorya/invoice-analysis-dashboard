#!/usr/bin/env python3
"""
Vendor-Centric Invoice Analysis Dashboard - Production Version
Optimized for deployment with better error handling and configuration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import os
import re

# Page configuration
st.set_page_config(
    page_title="Vendor-Centric Analysis",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .vendor-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .insight-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .deployment-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class VendorCentricDashboard:
    def __init__(self):
        # Try multiple database locations for deployment flexibility
        self.db_paths = [
            "invoice_line_items.db",
            "/app/invoice_line_items.db",  # Docker path
            "./invoice_line_items.db"
        ]
        self.conn = None
        
    def connect_db(self):
        """Connect to database with multiple path fallbacks."""
        for db_path in self.db_paths:
            try:
                if os.path.exists(db_path):
                    self.conn = sqlite3.connect(db_path)
                    st.success(f"✅ Connected to database: {db_path}")
                    return True
            except Exception as e:
                st.warning(f"⚠️ Failed to connect to {db_path}: {e}")
                continue
        
        st.error("❌ Could not connect to any database. Please ensure invoice_line_items.db is available.")
        return False
    
    def disconnect_db(self):
        if self.conn:
            self.conn.close()
    
    def get_available_vendors(self):
        """Get list of available vendors."""
        try:
            if not self.conn:
                return []
            
            query = """
            SELECT DISTINCT vendor 
            FROM invoices 
            WHERE vendor IS NOT NULL AND vendor != '' 
            ORDER BY vendor
            """
            df = pd.read_sql_query(query, self.conn)
            return df['vendor'].tolist()
        except Exception as e:
            st.error(f"Error getting vendors: {e}")
            return []
    
    def get_vendor_overview(self, vendor_name):
        """Get overview data for a specific vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            # Use separate queries to avoid JOIN inflation
            query = """
            SELECT 
                COUNT(DISTINCT filename) as invoice_count,
                SUM(total_amount) as total_spending,
                AVG(total_amount) as avg_invoice_value,
                MIN(date) as first_order,
                MAX(date) as last_order
            FROM invoices
            WHERE vendor = ?
            GROUP BY vendor
            """
            
            # Get invoice data
            invoice_df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            
            # Get line item count separately
            line_query = """
            SELECT COUNT(*) as line_item_count
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE i.vendor = ?
            """
            
            line_df = pd.read_sql_query(line_query, self.conn, params=(vendor_name,))
            
            # Combine the results
            if not invoice_df.empty and not line_df.empty:
                invoice_df['line_item_count'] = line_df.iloc[0]['line_item_count']
            
            return invoice_df
        except Exception as e:
            st.error(f"Error getting vendor overview: {e}")
            return pd.DataFrame()
    
    def get_vendor_duplicates(self, vendor_name):
        """Get duplicate analysis for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                filename,
                total_amount,
                date,
                COUNT(*) as duplicate_count
            FROM invoices 
            WHERE vendor = ?
            GROUP BY filename
            HAVING COUNT(*) > 1
            ORDER BY duplicate_count DESC, filename
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            return df
        except Exception as e:
            st.error(f"Error getting vendor duplicates: {e}")
            return pd.DataFrame()
    
    def get_vendor_monthly_trends(self, vendor_name):
        """Get monthly spending trends for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                strftime('%Y-%m', date) as month,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(i.total_amount) as total_spending,
                AVG(i.total_amount) as avg_invoice_value
            FROM invoices i
            WHERE i.vendor = ? AND i.date IS NOT NULL AND i.date != ''
            GROUP BY month
            ORDER BY month
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            
            if not df.empty:
                df['month'] = pd.to_datetime(df['month'] + '-01')
                df['month_name'] = df['month'].dt.strftime('%b %Y')
            
            return df
        except Exception as e:
            st.error(f"Error getting monthly trends: {e}")
            return pd.DataFrame()
    
    def get_vendor_daily_analysis(self, vendor_name):
        """Get daily spending analysis for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                date,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(i.total_amount) as total_spending
            FROM invoices i
            WHERE i.vendor = ? AND i.date IS NOT NULL AND i.date != ''
            GROUP BY date
            ORDER BY date
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_week'] = df['date'].dt.day_name()
                df['month'] = df['date'].dt.month_name()
            
            return df
        except Exception as e:
            st.error(f"Error getting daily analysis: {e}")
            return pd.DataFrame()
    
    def get_vendor_weekly_patterns(self, vendor_name):
        """Get weekly spending patterns for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                strftime('%W', date) as week_number,
                strftime('%Y', date) as year,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(i.total_amount) as total_spending
            FROM invoices i
            WHERE i.vendor = ? AND i.date IS NOT NULL AND i.date != ''
            GROUP BY year, week_number
            ORDER BY year, week_number
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            
            if not df.empty:
                df['week_label'] = df['year'] + ' W' + df['week_number']
            
            return df
        except Exception as e:
            st.error(f"Error getting weekly patterns: {e}")
            return pd.DataFrame()
    
    def get_vendor_seasonal_analysis(self, vendor_name):
        """Get seasonal spending analysis for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                CASE 
                    WHEN strftime('%m', date) IN ('12', '01', '02') THEN 'Winter'
                    WHEN strftime('%m', date) IN ('03', '04', '05') THEN 'Spring'
                    WHEN strftime('%m', date) IN ('06', '07', '08') THEN 'Summer'
                    WHEN strftime('%m', date) IN ('09', '10', '11') THEN 'Fall'
                    ELSE 'Unknown'
                END as season,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(i.total_amount) as total_spending
            FROM invoices i
            WHERE i.vendor = ? AND i.date IS NOT NULL AND i.date != ''
            GROUP BY season
            ORDER BY 
                CASE season
                    WHEN 'Winter' THEN 1
                    WHEN 'Spring' THEN 2
                    WHEN 'Summer' THEN 3
                    WHEN 'Fall' THEN 4
                    ELSE 5
                END
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            return df
        except Exception as e:
            st.error(f"Error getting seasonal analysis: {e}")
            return pd.DataFrame()
    
    def get_vendor_categories(self, vendor_name):
        """Get category breakdown for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                l.category,
                COUNT(*) as item_count,
                SUM(l.total_price) as total_value,
                AVG(l.total_price) as avg_price
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE i.vendor = ? AND l.category IS NOT NULL AND l.category != ''
            GROUP BY l.category
            ORDER BY total_value DESC
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            return df
        except Exception as e:
            st.error(f"Error getting vendor categories: {e}")
            return pd.DataFrame()
    
    def get_vendor_pricing(self, vendor_name):
        """Get pricing analysis for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                l.description,
                l.category,
                COUNT(*) as purchase_count,
                AVG(l.unit_price) as avg_unit_price,
                MIN(l.unit_price) as min_unit_price,
                MAX(l.unit_price) as max_unit_price,
                SUM(l.total_price) as total_value
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE i.vendor = ? AND l.description IS NOT NULL AND l.description != ''
            GROUP BY l.description, l.category
            HAVING purchase_count > 1
            ORDER BY total_value DESC
            LIMIT 20
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            return df
        except Exception as e:
            st.error(f"Error getting vendor pricing: {e}")
            return pd.DataFrame()
    
    def get_vendor_details(self, vendor_name):
        """Get detailed invoice information for a vendor."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                filename,
                date,
                total_amount
            FROM invoices 
            WHERE vendor = ?
            ORDER BY date DESC
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            
            # Convert date column to datetime for better handling
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            return df
        except Exception as e:
            st.error(f"Error getting vendor details: {e}")
            return pd.DataFrame()

    def get_all_vendors_monthly_trends(self):
        """Get monthly spending trends across all vendors."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                strftime('%Y-%m', date) as month,
                vendor,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(total_amount) as total_spending,
                AVG(total_amount) as avg_invoice_value
            FROM invoices 
            WHERE date IS NOT NULL AND date != '' AND vendor IS NOT NULL AND vendor != ''
            GROUP BY month, vendor
            ORDER BY month, total_spending DESC
            """
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                df['month'] = pd.to_datetime(df['month'] + '-01')
                df['month_name'] = df['month'].dt.strftime('%b %Y')
            
            return df
        except Exception as e:
            st.error(f"Error getting all vendors monthly trends: {e}")
            return pd.DataFrame()

    def get_all_vendors_seasonal_analysis(self):
        """Get seasonal spending patterns across all vendors."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                vendor,
                CASE 
                    WHEN strftime('%m', date) IN ('12', '01', '02') THEN 'Winter'
                    WHEN strftime('%m', date) IN ('03', '04', '05') THEN 'Spring'
                    WHEN strftime('%m', date) IN ('06', '07', '08') THEN 'Summer'
                    WHEN strftime('%m', date) IN ('09', '10', '11') THEN 'Fall'
                    ELSE 'Unknown'
                END as season,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(total_amount) as total_spending,
                AVG(total_amount) as avg_invoice_value
            FROM invoices 
            WHERE date IS NOT NULL AND date != '' AND vendor IS NOT NULL AND vendor != ''
            GROUP BY vendor, season
            ORDER BY vendor, 
                CASE season
                    WHEN 'Winter' THEN 1
                    WHEN 'Spring' THEN 2
                    WHEN 'Summer' THEN 3
                    WHEN 'Fall' THEN 4
                    ELSE 5
                END
            """
            df = pd.read_sql_query(query, self.conn)
            return df
        except Exception as e:
            st.error(f"Error getting all vendors seasonal analysis: {e}")
            return pd.DataFrame()

    def get_all_vendors_weekly_patterns(self):
        """Get weekly spending patterns across all vendors."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                strftime('%W', date) as week_number,
                strftime('%Y', date) as year,
                vendor,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(total_amount) as total_spending
            FROM invoices 
            WHERE date IS NOT NULL AND date != '' AND vendor IS NOT NULL AND vendor != ''
            GROUP BY week_number, year, vendor
            ORDER BY year, week_number, total_spending DESC
            """
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                df['week_label'] = df['year'] + ' W' + df['week_number']
            
            return df
        except Exception as e:
            st.error(f"Error getting all vendors weekly patterns: {e}")
            return pd.DataFrame()

    def get_all_vendors_daily_analysis(self):
        """Get daily spending analysis across all vendors."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                date,
                vendor,
                COUNT(DISTINCT filename) as invoice_count,
                SUM(total_amount) as total_spending
            FROM invoices 
            WHERE date IS NOT NULL AND date != '' AND vendor IS NOT NULL AND vendor != ''
            GROUP BY date, vendor
            ORDER BY date DESC, total_spending DESC
            """
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_week'] = df['date'].dt.day_name()
                df['month'] = df['date'].dt.month_name()
            
            return df
        except Exception as e:
            st.error(f"Error getting all vendors daily analysis: {e}")
            return pd.DataFrame()

    def get_time_based_optimization_insights(self):
        """Get time-based optimization insights across all vendors."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            # Get monthly trends for optimization analysis
            monthly_data = self.get_all_vendors_monthly_trends()
            
            if monthly_data.empty:
                return pd.DataFrame()
            
            insights = []
            
            # Analyze spending volatility
            vendor_volatility = monthly_data.groupby('vendor')['total_spending'].agg(['mean', 'std']).reset_index()
            vendor_volatility['coefficient_of_variation'] = vendor_volatility['std'] / vendor_volatility['mean']
            
            # High volatility vendors (opportunity for bulk ordering)
            high_volatility = vendor_volatility[vendor_volatility['coefficient_of_variation'] > 0.5]
            if not high_volatility.empty:
                for _, row in high_volatility.iterrows():
                    insights.append({
                        'vendor': row['vendor'],
                        'insight_type': 'High Spending Volatility',
                        'recommendation': 'Consider bulk ordering and inventory management',
                        'metric': f"CV: {row['coefficient_of_variation']:.2f}"
                    })
            
            # Seasonal patterns
            seasonal_data = self.get_all_vendors_seasonal_analysis()
            if not seasonal_data.empty:
                seasonal_spending = seasonal_data.groupby('season')['total_spending'].sum().reset_index()
                peak_season = seasonal_spending.loc[seasonal_spending['total_spending'].idxmax()]
                insights.append({
                    'vendor': 'All Vendors',
                    'insight_type': 'Seasonal Peak',
                    'recommendation': f'Peak spending in {peak_season["season"]} - plan inventory accordingly',
                    'metric': f"${peak_season['total_spending']:,.0f}"
                })
            
            return pd.DataFrame(insights)
        except Exception as e:
            st.error(f"Error getting time-based optimization insights: {e}")
            return pd.DataFrame()

    def get_product_price_variability_insights(self):
        """Get insights on products with high price variability across vendors."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                l.description,
                l.category,
                COUNT(DISTINCT i.vendor) as vendor_count,
                AVG(l.unit_price) as avg_price,
                MIN(l.unit_price) as min_price,
                MAX(l.unit_price) as max_price,
                (MAX(l.unit_price) - MIN(l.unit_price)) as price_range,
                (MAX(l.unit_price) - MIN(l.unit_price)) / AVG(l.unit_price) * 100 as price_variability_pct,
                GROUP_CONCAT(DISTINCT i.vendor || ' ($' || l.unit_price || ')') as vendor_prices
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE l.description IS NOT NULL AND l.description != '' 
                AND l.unit_price > 0
                AND i.vendor IS NOT NULL AND i.vendor != ''
            GROUP BY l.description, l.category
            HAVING vendor_count > 1 AND price_variability_pct > 20
            ORDER BY price_variability_pct DESC
            LIMIT 20
            """
            
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                # Add detailed insights and restaurant-specific recommendations
                insights = []
                for _, row in df.iterrows():
                    # Calculate potential savings
                    potential_savings = row['price_range'] * 10  # Assuming 10 units per month
                    annual_savings = potential_savings * 12
                    
                    if row['price_variability_pct'] > 50:
                        insight_type = "🚨 High Price Variability - Immediate Action Required"
                        recommendation = f"Switch to lower-priced vendor or negotiate better rates. You could save ${annual_savings:.2f} annually on this product."
                        business_impact = f"High impact: ${annual_savings:.2f} annual savings potential"
                        example = f"Example: If you buy 10 {row['description']} per month, switching from ${row['max_price']:.2f} to ${row['min_price']:.2f} saves ${row['price_range']:.2f} per order = ${annual_savings:.2f} annually"
                    elif row['price_variability_pct'] > 30:
                        insight_type = "⚠️ Moderate Price Variability - Monitor Closely"
                        recommendation = f"Monitor pricing and consider vendor consolidation. Potential annual savings: ${annual_savings:.2f}"
                        business_impact = f"Medium impact: ${annual_savings:.2f} annual savings potential"
                        example = f"Example: Current price range ${row['min_price']:.2f} - ${row['max_price']:.2f} suggests room for negotiation"
                    else:
                        insight_type = "📊 Price Variability Detected - Review Strategy"
                        recommendation = f"Review pricing strategy for this product. Consider bulk ordering for better rates."
                        business_impact = f"Low impact: ${annual_savings:.2f} annual savings potential"
                        example = f"Example: Price varies by ${row['price_range']:.2f} across vendors - opportunity for bulk discounts"
                    
                    insights.append({
                        'product': row['description'],
                        'category': row['category'],
                        'vendor_count': row['vendor_count'],
                        'avg_price': row['avg_price'],
                        'price_range': row['price_range'],
                        'variability_pct': row['price_variability_pct'],
                        'vendor_prices': row['vendor_prices'],
                        'insight_type': insight_type,
                        'recommendation': recommendation,
                        'business_impact': business_impact,
                        'example': example,
                        'annual_savings_potential': annual_savings
                    })
                
                return pd.DataFrame(insights)
            
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error getting product price variability insights: {e}")
            return pd.DataFrame()

    def get_vendor_switching_recommendations(self):
        """Get recommendations for vendor switching based on price analysis."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                l.description,
                l.category,
                i.vendor,
                l.unit_price,
                l.total_price,
                COUNT(*) as purchase_count,
                AVG(l.unit_price) OVER (PARTITION BY l.description) as avg_market_price,
                (l.unit_price - AVG(l.unit_price) OVER (PARTITION BY l.description)) / AVG(l.unit_price) OVER (PARTITION BY l.description) * 100 as price_premium_pct
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE l.description IS NOT NULL AND l.description != '' 
                AND l.unit_price > 0
                AND i.vendor IS NOT NULL AND i.vendor != ''
            ORDER BY l.description, price_premium_pct DESC
            """
            
            df = pd.read_sql_query(query, self.conn)
            
            if not df.empty:
                # Find products where current vendor has high price premium
                high_premium = df[df['price_premium_pct'] > 15].copy()
                
                if not high_premium.empty:
                    # Get alternative vendors for each product
                    recommendations = []
                    for _, row in high_premium.iterrows():
                        alternatives = df[
                            (df['description'] == row['description']) & 
                            (df['vendor'] != row['vendor'])
                        ]
                        
                        if not alternatives.empty:
                            best_alternative = alternatives.loc[alternatives['unit_price'].idxmin()]
                            potential_savings = (row['unit_price'] - best_alternative['unit_price']) * row['purchase_count']
                            
                            # Calculate annual impact
                            monthly_frequency = row['purchase_count'] / 3  # Assuming 3 months of data
                            annual_savings = potential_savings * 4 * 12  # Extrapolate to annual
                            
                            # Create detailed recommendation with examples
                            if row['price_premium_pct'] > 30:
                                priority = "🚨 High Priority - Immediate Action"
                                impact_level = "High Impact"
                                action_required = "Switch vendors immediately or negotiate aggressively"
                            elif row['price_premium_pct'] > 20:
                                priority = "⚠️ Medium Priority - Plan Switch"
                                impact_level = "Medium Impact"
                                action_required = "Plan vendor switch within 30 days"
                            else:
                                priority = "📊 Low Priority - Monitor"
                                impact_level = "Low Impact"
                                action_required = "Monitor pricing and consider switch"
                            
                            # Restaurant-specific examples
                            if "coffee" in row['description'].lower() or "bean" in row['description'].lower():
                                example = f"Example: You're paying ${row['unit_price']:.2f} for {row['description']} from {row['vendor']}, but {best_alternative['vendor']} offers it for ${best_alternative['unit_price']:.2f}. For a coffee shop using 50 lbs/month, this saves ${(row['unit_price'] - best_alternative['unit_price']) * 50:.2f} monthly = ${annual_savings:.2f} annually"
                            elif "produce" in row['description'].lower() or "vegetable" in row['description'].lower():
                                example = f"Example: {row['description']} costs ${row['unit_price']:.2f} from {row['vendor']} vs ${best_alternative['unit_price']:.2f} from {best_alternative['vendor']}. For a restaurant using 20 units/week, switching saves ${(row['unit_price'] - best_alternative['unit_price']) * 20 * 52:.2f} annually"
                            elif "linen" in row['description'].lower() or "napkin" in row['description'].lower():
                                example = f"Example: Linen service from {row['vendor']} costs ${row['unit_price']:.2f} vs ${best_alternative['unit_price']:.2f} from {best_alternative['vendor']}. For weekly service, switching saves ${(row['unit_price'] - best_alternative['unit_price']) * 52:.2f} annually"
                            else:
                                example = f"Example: {row['description']} from {row['vendor']} costs ${row['unit_price']:.2f} vs ${best_alternative['unit_price']:.2f} from {best_alternative['vendor']}. Switching saves ${(row['unit_price'] - best_alternative['unit_price']) * row['purchase_count']:.2f} per order"
                            
                            recommendations.append({
                                'product': row['description'],
                                'category': row['category'],
                                'current_vendor': row['vendor'],
                                'current_price': row['unit_price'],
                                'alternative_vendor': best_alternative['vendor'],
                                'alternative_price': best_alternative['unit_price'],
                                'price_premium_pct': row['price_premium_pct'],
                                'potential_savings': potential_savings,
                                'annual_savings': annual_savings,
                                'priority': priority,
                                'impact_level': impact_level,
                                'action_required': action_required,
                                'example': example,
                                'recommendation': f"Switch from {row['vendor']} to {best_alternative['vendor']} for {row['description']}"
                            })
                    
                    return pd.DataFrame(recommendations)
            
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error getting vendor switching recommendations: {e}")
            return pd.DataFrame()

    def clean_product_name(self, raw_name):
        """Clean up product names by removing vendor names, dates, and other metadata."""
        if not raw_name:
            return "Unknown Product"
        
        # Check if this looks like a real product name (contains actual product info)
        if any(keyword in raw_name.upper() for keyword in ['CHEESE', 'COFFEE', 'BAGEL', 'CREAM', 'MILK', 'BREAD', 'PRODUCE', 'LINEN', 'SERVICE']):
            # This is likely a real product, clean it up
            name = raw_name.strip()
            
            # Remove common prefixes that aren't product names
            name = re.sub(r'^\d+\s+\d+\.\d+\s+\d+\s+', '', name)  # Remove "0 0.00 671778" type prefixes
            name = re.sub(r'^\d+\s+\d+\.\d+\s+', '', name)  # Remove "0 0.00" type prefixes
            
            # Clean up extra whitespace
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()
            
            return name if name else "Product"
        
        # For metadata descriptions, try to extract meaningful information
        name = raw_name.strip()
        
        # Remove vendor names that commonly appear in descriptions
        vendor_patterns = [
            r'ankit\s+sirmorya\s*', r'westman\'s?\s*', r'bagel\s*and?\s*coffee\s*',
            r'chemmark\s*of\s*washington\s*', r'costco\s*business\s*', r'costco\s*',
            r'franks?\s*quality\s*produce\s*', r'over\s*the\s*moon\s*coffee\s*roasters\s*',
            r'pacific\s*food\s*importers\s*', r'pfi\s*', r'tomlinson\s*linen\s*service\s*',
            r'villa\s*jerada\s*', r'linen\s*service\s*', r'coffee\s*roasters\s*'
        ]
        
        for pattern in vendor_patterns:
            name = re.sub(pattern, '', name, flags=re.IGNORECASE)
        
        # Remove date patterns
        name = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', name)
        name = re.sub(r'\d{1,2}-\d{1,2}-\d{2,4}', '', name)
        name = re.sub(r'DATE', '', name, flags=re.IGNORECASE)
        
        # Remove invoice numbers and batch numbers
        name = re.sub(r'INVOICE\s*\d+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'BATCH\s*\d+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'ARPFIINVOEBTCHLASER\s*\(\d+\)', '', name, flags=re.IGNORECASE)
        
        # Remove common metadata
        name = re.sub(r'from\s+[^,]+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\([^)]*\)', '', name)  # Remove parentheses content
        
        # Clean up extra whitespace and punctuation
        name = re.sub(r'\s+', ' ', name)  # Multiple spaces to single space
        name = name.strip(' ,.-_')
        
        # If the name is too short or generic after cleaning, use a more descriptive version
        if len(name) < 3 or name.lower() in ['', 'product', 'item', 'goods', 'supplies']:
            # Extract meaningful parts from the original name
            parts = raw_name.split()
            meaningful_parts = [part for part in parts if len(part) > 2 and part.lower() not in 
                              ['ankit', 'sirmorya', 'westman', 'bagel', 'coffee', 'date', 'invoice', 'batch']]
            if meaningful_parts:
                name = ' '.join(meaningful_parts[:3])  # Take first 3 meaningful words
            else:
                name = raw_name[:50]  # Use first 50 characters if no meaningful parts
        
        return name if name else "Product"
    
    def get_strategic_cost_optimization_insights(self):
        """Get comprehensive strategic cost optimization insights based on actual line items data."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            insights = []
            
            # 1. Top Spending Products - Most Expensive Line Items
            try:
                top_spending_products = pd.read_sql_query("""
                    SELECT 
                        l.description,
                        l.category,
                        i.vendor,
                        l.unit_price,
                        l.quantity,
                        l.total_price,
                        COUNT(*) as order_frequency
                    FROM line_items l
                    JOIN invoices i ON l.filename = i.filename
                    WHERE l.description IS NOT NULL AND l.description != ''
                        AND l.total_price > 0
                    GROUP BY l.description, l.category, i.vendor, l.unit_price, l.quantity, l.total_price
                    ORDER BY l.total_price DESC
                    LIMIT 10
                """, self.conn)
                
                if not top_spending_products.empty:
                    # Group by product to find total spending across all vendors
                    product_totals = top_spending_products.groupby('description').agg({
                        'total_price': 'sum',
                        'order_frequency': 'sum',
                        'vendor': lambda x: list(set(x)),
                        'unit_price': 'mean'
                    }).reset_index()
                    
                    # Find products with highest total spending
                    top_products = product_totals.nlargest(5, 'total_price')
                    
                    for _, product in top_products.iterrows():
                        raw_product_name = str(product['description'])
                        # Clean up product name by removing vendor names and metadata
                        product_name = self.clean_product_name(raw_product_name)
                        total_spending = float(product['total_price'])
                        frequency = int(product['order_frequency'])
                        vendors = product['vendor']
                        avg_price = float(product['unit_price'])
                        
                        if len(vendors) > 1:
                            insight_type = '🚨 High-Spending Product - Multi-Vendor Opportunity'
                            recommendation = f'Consolidate {product_name} purchases to 1 vendor for bulk pricing'
                            potential_savings = total_spending * 0.15  # 15% savings from consolidation
                            business_example = f"Example: {product_name} costs you ${total_spending:,.2f} annually across {len(vendors)} vendors. Consolidating to 1 vendor could save ${potential_savings:,.2f} through bulk pricing and reduced admin costs."
                            action_items = [
                                f'Analyze pricing from all {len(vendors)} vendors for {product_name}',
                                'Negotiate bulk pricing with preferred vendor',
                                'Plan gradual transition over 2-3 months',
                                'Monitor quality and service during transition'
                            ]
                        else:
                            insight_type = '💰 High-Spending Product - Negotiation Opportunity'
                            recommendation = f'Leverage high spending on {product_name} for better rates'
                            potential_savings = total_spending * 0.10  # 10% savings from negotiation
                            business_example = f"Example: {product_name} costs you ${total_spending:,.2f} annually from {vendors[0]}. You can negotiate: 1) 10-15% bulk discount, 2) Better payment terms, 3) Priority service, 4) Free delivery."
                            action_items = [
                                f'Request bulk pricing tiers for {product_name}',
                                'Negotiate extended payment terms (Net 30-45)',
                                'Ask for free delivery and priority service',
                                'Request quarterly business reviews for pricing optimization'
                            ]
                        
                        insights.append({
                            'insight_type': insight_type,
                            'vendor': f"Product: {product_name}",
                            'metric': f"${total_spending:,.2f} annually, {frequency} orders",
                            'recommendation': recommendation,
                            'potential_impact': 'High',
                            'business_example': business_example,
                            'action_items': action_items
                        })
            except Exception as e:
                st.error(f"Error processing top spending products: {e}")
            
            # 2. Most Frequently Ordered Products - Volume Optimization
            try:
                frequent_products = pd.read_sql_query("""
                    SELECT 
                        l.description,
                        l.category,
                        COUNT(*) as order_count,
                        SUM(l.quantity) as total_quantity,
                        SUM(l.total_price) as total_spending,
                        AVG(l.unit_price) as avg_unit_price,
                        COUNT(DISTINCT i.vendor) as vendor_count
                    FROM line_items l
                    JOIN invoices i ON l.filename = i.filename
                    WHERE l.description IS NOT NULL AND l.description != ''
                        AND l.quantity > 0
                    GROUP BY l.description, l.category
                    HAVING order_count >= 3
                    ORDER BY order_count DESC, total_spending DESC
                    LIMIT 8
                """, self.conn)
                
                if not frequent_products.empty:
                    for _, product in frequent_products.iterrows():
                        raw_product_name = str(product['description'])
                        product_name = self.clean_product_name(raw_product_name)
                        order_count = int(product['order_count'])
                        total_quantity = float(product['total_quantity'])
                        total_spending = float(product['total_spending'])
                        vendor_count = int(product['vendor_count'])
                        avg_price = float(product['avg_unit_price'])
                        
                        if vendor_count > 1:
                            consolidation_savings = total_spending * 0.12  # 12% savings from consolidation
                            insight_type = '🔄 Frequently Ordered - Multi-Vendor Consolidation'
                            recommendation = f'Consolidate {product_name} orders to reduce costs and improve efficiency'
                            business_example = f"Example: {product_name} is ordered {order_count} times annually (${total_spending:,.2f}) from {vendor_count} vendors. Consolidating to 1 vendor could save ${consolidation_savings:,.2f} through: 1) Volume pricing, 2) Consistent delivery scheduling, 3) Reduced ordering complexity, 4) Better inventory management."
                            action_items = [
                                f'Evaluate all {vendor_count} vendors for {product_name}',
                                'Select vendor with best pricing and service',
                                'Negotiate volume discounts based on {order_count} annual orders',
                                'Implement regular ordering schedule'
                            ]
                        else:
                            volume_savings = total_spending * 0.08  # 8% savings from volume pricing
                            insight_type = '📦 Frequently Ordered - Volume Pricing Opportunity'
                            recommendation = f'Negotiate volume pricing for {product_name} based on high order frequency'
                            business_example = f"Example: {product_name} is ordered {order_count} times annually (${total_spending:,.2f}) from current vendor. You can negotiate: 1) Volume pricing tiers, 2) Regular delivery scheduling, 3) Bulk ordering discounts, 4) Loyalty program benefits."
                            action_items = [
                                f'Request volume pricing for {product_name}',
                                'Negotiate regular delivery schedule',
                                'Ask for bulk ordering incentives',
                                'Request loyalty program enrollment'
                            ]
                        
                        insights.append({
                            'insight_type': insight_type,
                            'vendor': f"Product: {product_name}",
                            'metric': f"{order_count} orders, {total_quantity:.0f} units, ${total_spending:,.2f}",
                            'recommendation': recommendation,
                            'potential_impact': 'Medium',
                            'business_example': business_example,
                            'action_items': action_items
                        })
            except Exception as e:
                st.error(f"Error processing frequent products: {e}")
            
            # 3. High Unit Price Products - Price Optimization
            try:
                high_unit_price_products = pd.read_sql_query("""
                    SELECT 
                        l.description,
                        l.category,
                        i.vendor,
                        l.unit_price,
                        l.quantity,
                        l.total_price,
                        COUNT(*) as order_frequency
                    FROM line_items l
                    JOIN invoices i ON l.filename = i.filename
                    WHERE l.description IS NOT NULL AND l.description != ''
                        AND l.unit_price > 10  -- Focus on products with unit price > $10
                        AND l.quantity > 0
                    GROUP BY l.description, l.category, i.vendor, l.unit_price, l.quantity, l.total_price
                    ORDER BY l.unit_price DESC
                    LIMIT 8
                """, self.conn)
                
                if not high_unit_price_products.empty:
                    for _, product in high_unit_price_products.iterrows():
                        raw_product_name = str(product['description'])
                        product_name = self.clean_product_name(raw_product_name)
                        unit_price = float(product['unit_price'])
                        quantity = float(product['quantity'])
                        total_price = float(product['total_price'])
                        vendor = str(product['vendor'])
                        frequency = int(product['order_frequency'])
                        
                        # Check if this product is available from other vendors
                        alternative_vendors = pd.read_sql_query("""
                            SELECT DISTINCT i.vendor, AVG(l.unit_price) as avg_price
                            FROM line_items l
                            JOIN invoices i ON l.filename = i.filename
                            WHERE l.description = ? AND i.vendor != ?
                            GROUP BY i.vendor
                        """, self.conn, params=(product_name, vendor))
                        
                        if not alternative_vendors.empty:
                            best_alternative = alternative_vendors.loc[alternative_vendors['avg_price'].idxmin()]
                            alternative_vendor = str(best_alternative['vendor'])
                            alternative_price = float(best_alternative['avg_price'])
                            price_difference = unit_price - alternative_price
                            potential_savings = price_difference * quantity * frequency
                            
                            if price_difference > 0:
                                insight_type = '💸 High Unit Price - Better Alternative Available'
                                recommendation = f'Switch {product_name} to {alternative_vendor} for better pricing'
                                business_example = f"Example: {product_name} costs ${unit_price:.2f} from {vendor} vs ${alternative_price:.2f} from {alternative_vendor}. Switching could save ${potential_savings:.2f} annually based on your ordering pattern."
                                action_items = [
                                    f'Test {product_name} from {alternative_vendor}',
                                    'Compare quality and service between vendors',
                                    'Plan gradual transition if quality is acceptable',
                                    'Negotiate even better pricing with new vendor'
                                ]
                            else:
                                insight_type = '✅ High Unit Price - Best Price Available'
                                recommendation = f'Current pricing for {product_name} is competitive'
                                business_example = f"Example: {product_name} at ${unit_price:.2f} from {vendor} is already the best price available. Focus optimization efforts on other products."
                                action_items = [
                                    'Monitor pricing trends for this product',
                                    'Consider bulk ordering for additional discounts',
                                    'Explore alternative products or suppliers',
                                    'Maintain current vendor relationship'
                                ]
                        else:
                            insight_type = '🔍 High Unit Price - Research Alternative Suppliers'
                            recommendation = f'Research alternative suppliers for {product_name} to find better pricing'
                            business_example = f"Example: {product_name} costs ${unit_price:.2f} from {vendor} with no alternatives found. Research new suppliers or negotiate better rates."
                            action_items = [
                                'Research new suppliers for this product',
                                'Negotiate better rates with current vendor',
                                'Consider alternative product options',
                                'Join industry groups for supplier recommendations'
                            ]
                        
                        insights.append({
                            'insight_type': insight_type,
                            'vendor': f"Product: {product_name}",
                            'metric': f"${unit_price:.2f} unit price, {quantity:.0f} quantity, ${total_price:.2f} total",
                            'recommendation': recommendation,
                            'potential_impact': 'Medium',
                            'business_example': business_example,
                            'action_items': action_items
                        })
            except Exception as e:
                st.error(f"Error processing high unit price products: {e}")
            
            # 4. Category Spending Analysis - Strategic Sourcing
            try:
                category_spending = pd.read_sql_query("""
                    SELECT 
                        l.category,
                        COUNT(DISTINCT i.vendor) as vendor_count,
                        SUM(l.total_price) as total_spending,
                        AVG(l.unit_price) as avg_unit_price,
                        COUNT(*) as total_orders
                    FROM line_items l
                    JOIN invoices i ON l.filename = i.filename
                    WHERE l.category IS NOT NULL AND l.category != ''
                        AND l.total_price > 0
                    GROUP BY l.category
                    HAVING total_spending > 1000  -- Focus on categories with significant spending
                    ORDER BY total_spending DESC
                """, self.conn)
                
                if not category_spending.empty:
                    for _, category in category_spending.iterrows():
                        category_name = str(category['category'])
                        vendor_count = int(category['vendor_count'])
                        total_spending = float(category['total_spending'])
                        total_orders = int(category['total_orders'])
                        
                        if vendor_count >= 3:
                            consolidation_savings = total_spending * 0.15  # 15% savings from consolidation
                            insight_type = '🔄 Multi-Vendor Category - Strategic Consolidation'
                            recommendation = f'Consolidate {category_name} sourcing to 1-2 strategic vendors'
                            business_example = f"Example: {category_name} costs ${total_spending:,.2f} annually across {vendor_count} vendors. Consolidating to 1-2 vendors could save ${consolidation_savings:,.2f} through: 1) Volume pricing, 2) Consistent quality standards, 3) Coordinated delivery schedules, 4) Better relationship management."
                            action_items = [
                                'Evaluate all vendors for quality, price, and service',
                                'Select top 2 vendors based on comprehensive criteria',
                                'Negotiate volume discounts with selected vendors',
                                'Plan gradual transition over 3-6 months'
                            ]
                        elif vendor_count == 2:
                            optimization_savings = total_spending * 0.08  # 8% savings from optimization
                            insight_type = '⚖️ Dual-Vendor Category - Optimization Opportunity'
                            recommendation = f'Optimize {category_name} sourcing between 2 vendors for best pricing'
                            business_example = f"Example: {category_name} costs ${total_spending:,.2f} annually from 2 vendors. You can optimize by: 1) Playing vendors against each other for better rates, 2) Splitting orders strategically, 3) Negotiating volume discounts from both, 4) Maintaining backup supplier relationships."
                            action_items = [
                                'Analyze pricing patterns from both vendors',
                                'Negotiate competitive rates using dual-vendor leverage',
                                'Develop strategic ordering plan',
                                'Monitor performance and adjust strategy quarterly'
                            ]
                        else:
                            relationship_savings = total_spending * 0.05  # 5% savings from relationship building
                            insight_type = '🤝 Single-Vendor Category - Relationship Building'
                            recommendation = f'Strengthen relationship with {category_name} vendor for better pricing'
                            business_example = f"Example: {category_name} costs ${total_spending:,.2f} annually from single vendor. You can optimize by: 1) Building stronger relationship, 2) Negotiating volume discounts, 3) Requesting better payment terms, 4) Exploring exclusive pricing arrangements."
                            action_items = [
                                'Schedule quarterly business reviews with vendor',
                                'Negotiate volume-based pricing tiers',
                                'Request extended payment terms',
                                'Explore exclusive or preferred customer programs'
                            ]
                        
                        insights.append({
                            'insight_type': insight_type,
                            'vendor': f"Category: {category_name}",
                            'metric': f"{vendor_count} vendors, ${total_spending:,.2f} spending, {total_orders} orders",
                            'recommendation': recommendation,
                            'potential_impact': 'Medium',
                            'business_example': business_example,
                            'action_items': action_items
                        })
            except Exception as e:
                st.error(f"Error processing category spending: {e}")
            
            return pd.DataFrame(insights)
        except Exception as e:
            st.error(f"Error getting strategic cost optimization insights: {e}")
            return pd.DataFrame()

def show_combined_analysis(dashboard):
    """Show combined analysis across all vendors."""
    st.subheader("🏪 Restaurant Supply Chain Overview")
    st.markdown("**Comprehensive vendor performance analysis for Westman's Bagels & Coffee operations**")
    
    try:
        conn = dashboard.conn
        if not conn:
            st.error("Database connection not available")
            return
            
        cursor = conn.cursor()
        
        # 1. Vendor Summary
        st.subheader("📊 Supply Chain Partner Summary")
        summary_query = """
        SELECT 
            vendor,
            COUNT(DISTINCT filename) as invoice_count,
            SUM(total_amount) as total_spending,
            AVG(total_amount) as avg_invoice_value
        FROM invoices 
        WHERE vendor IS NOT NULL AND vendor != '' AND date IS NOT NULL AND date != ''
        GROUP BY vendor
        ORDER BY total_spending DESC
        """
        summary_df = pd.read_sql_query(summary_query, conn)
        
        if not summary_df.empty:
            total_spending = summary_df['total_spending'].sum()
            total_invoices = summary_df['invoice_count'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Total Spending", f"${total_spending:,.0f}")
            with col2: st.metric("Total Invoices", total_invoices)
            with col3: st.metric("Active Vendors", len(summary_df))
            with col4: st.metric("Avg Invoice", f"${total_spending/total_invoices:,.0f}")
            
            fig = px.pie(summary_df, values='total_spending', names='vendor', title="Spending Distribution by Vendor")
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Detailed Vendor Breakdown:**")
            summary_df['percentage'] = (summary_df['total_spending'] / total_spending * 100).round(1)
            st.dataframe(summary_df, use_container_width=True)
        
        # 2. Quick Strategic Insights (High Priority)
        st.subheader("🎯 Quick Strategic Insights - Immediate Action Items")
        
        # Get key insights for quick display
        price_variability = dashboard.get_product_price_variability_insights()
        switching_recs = dashboard.get_vendor_switching_recommendations()
        strategic_insights = dashboard.get_strategic_cost_optimization_insights()
        
        # Display high-priority insights immediately
        col1, col2 = st.columns(2)
        
        with col1:
            if not price_variability.empty:
                high_var_count = len(price_variability[price_variability['variability_pct'] > 50])
                if high_var_count > 0:
                    st.error(f"🚨 **{high_var_count} High Price Variability Products**")
                    st.write("Products with >50% price difference across vendors")
                    st.write("**Action**: Review pricing strategy immediately")
                else:
                    st.success("✅ **Price Variability Under Control**")
            else:
                st.info("📊 **Price Variability**: No data available")
        
        with col2:
            if not switching_recs.empty:
                total_savings = switching_recs['potential_savings'].sum()
                if total_savings > 500:
                    st.error(f"💰 **${total_savings:,.0f} Potential Savings**")
                    st.write("From vendor switching opportunities")
                    st.write("**Action**: Review vendor switching recommendations")
                else:
                    st.success("✅ **Vendor Optimization**: Good pricing across vendors")
            else:
                st.info("🔄 **Vendor Switching**: No recommendations available")
        
        # 3. Risk Analysis (High Concentration Risk)
        st.subheader("🚨 Supply Chain Risk Assessment")
        if not summary_df.empty:
            top_vendor = summary_df.iloc[0]
            top_vendor_pct = (top_vendor['total_spending'] / total_spending) * 100
            if top_vendor_pct > 40:
                st.warning(f"⚠️ **High Concentration Risk**: {top_vendor['vendor']} represents {top_vendor_pct:.1f}% of total spending")
                st.info("💡 **Recommendation**: Consider diversifying suppliers to reduce dependency")
            else:
                st.success("✅ **Good Vendor Diversity**: No single vendor represents more than 40% of spending")
        
        # 4. Optimization Opportunities
        st.subheader("💰 Restaurant Cost Optimization Opportunities")
        high_spending = summary_df[summary_df['total_spending'] > 1000]
        if not high_spending.empty:
            st.write("**High-Spending Vendors (Negotiation Priority):**")
            for _, row in high_spending.iterrows():
                st.write(f"   • **{row['vendor']}**: ${row['total_spending']:,.2f} across {row['invoice_count']} invoices")
        
        # 4. Strategic Recommendations
        st.subheader("🎯 Strategic Business Recommendations")
        recommendations = []
        if not summary_df.empty:
            if len(summary_df) > 5: 
                recommendations.append("🏢 **Vendor Consolidation**: Consider reducing from {} to 4-5 strategic partners".format(len(summary_df)))
            if total_spending > 10000:
                recommendations.append("💰 **Volume Leverage**: Total spending of ${:,.0f} provides negotiation power".format(total_spending))
                recommendations.append("   → Negotiate better rates, bulk discounts, or payment terms")
        
        for rec in recommendations:
            st.write(rec)
        
        # 5. Savings Potential
        st.subheader("💡 Estimated Annual Savings for Westman's Bagels")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Conservative Estimates:**")
            st.write("• Volume discounts: 5-10% = **$2,400 - $4,800**")
            st.write("• Price standardization: 15-20% = **$7,200 - $9,600**")
            st.write("• Bulk ordering: 8-12% = **$3,800 - $5,700**")
            st.write("**Total**: **$13,400 - $20,100 annually**")
        with col2:
            st.write("**Aggressive Estimates:**")
            st.write("• Contract renegotiation: 15-25% = **$7,200 - $12,000**")
            st.write("• Vendor consolidation: 10-15% = **$4,800 - $7,200**")
            st.write("• Strategic sourcing: 20-30% = **$9,600 - $14,400**")
            st.write("**Total**: **$21,600 - $33,600 annually**")
        
        # 6. Strategic Business Insights & Recommendations
        st.subheader("🎯 Strategic Business Insights & Cost Optimization")
        st.markdown("**Actionable insights for vendor optimization, price variability, and strategic recommendations**")
        
        # Strategic insights tabs
        insights_tab1, insights_tab2, insights_tab3, insights_tab4 = st.tabs([
            "🚨 Price Variability", 
            "🔄 Vendor Switching", 
            "💰 Strategic Insights", 
            "📊 Time Analysis"
        ])
        
        with insights_tab1:
            st.markdown("### 🚨 Products with High Price Variability")
            price_variability = dashboard.get_product_price_variability_insights()
            if not price_variability.empty:
                # High priority insights
                high_variability = price_variability[price_variability['variability_pct'] > 50]
                if not high_variability.empty:
                    st.warning("**🚨 High Priority - Immediate Action Required**")
                    for _, insight in high_variability.iterrows():
                        st.markdown(f"""
                        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #856404;">{insight['insight_type']}</h4>
                            <p><strong>Product:</strong> {insight['product']}</p>
                            <p><strong>Category:</strong> {insight['category']}</p>
                            <p><strong>Price Variability:</strong> {insight['variability_pct']:.1f}%</p>
                            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show detailed insights with examples
                st.write("**🚨 High Priority Products (Immediate Action Required):**")
                high_priority = price_variability[price_variability['variability_pct'] > 50]
                if not high_priority.empty:
                    for _, insight in high_priority.iterrows():
                        st.markdown(f"""
                        <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                            <h4 style="margin: 0 0 1rem 0; color: #721c24;">{insight['insight_type']}</h4>
                            <p><strong>Product:</strong> {insight['product']}</p>
                            <p><strong>Category:</strong> {insight['category']}</p>
                            <p><strong>Price Variability:</strong> {insight['variability_pct']:.1f}%</p>
                            <p><strong>Vendor Prices:</strong> {insight['vendor_prices']}</p>
                            <p><strong>Business Impact:</strong> {insight['business_impact']}</p>
                            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                                <strong>💡 Restaurant Example:</strong><br>
                                {insight['example']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show all insights in a comprehensive table
                st.write("**📊 All Price Variability Insights:**")
                display_df = price_variability[['product', 'category', 'vendor_count', 'variability_pct', 'annual_savings_potential', 'insight_type']].copy()
                display_df['variability_pct'] = display_df['variability_pct'].round(1)
                display_df['annual_savings_potential'] = display_df['annual_savings_potential'].round(2)
                st.dataframe(display_df, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Variability Products", len(price_variability[price_variability['variability_pct'] > 50]))
                with col2:
                    st.metric("Moderate Variability", len(price_variability[(price_variability['variability_pct'] > 30) & (price_variability['variability_pct'] <= 50)]))
                with col3:
                    st.metric("Total Products Analyzed", len(price_variability))
            else:
                st.info("No significant price variability detected across vendors")
        
        with insights_tab2:
            st.markdown("### 🔄 Vendor Switching Recommendations")
            switching_recs = dashboard.get_vendor_switching_recommendations()
            if not switching_recs.empty:
                # High savings opportunities
                high_savings = switching_recs[switching_recs['potential_savings'] > 100]
                if not high_savings.empty:
                    st.success("**💰 High Savings Opportunities**")
                    for _, rec in high_savings.iterrows():
                        st.markdown(f"""
                        <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #155724;">🔄 Vendor Switch Recommendation</h4>
                            <p><strong>Product:</strong> {rec['product']}</p>
                            <p><strong>Current Vendor:</strong> {rec['current_vendor']} (${rec['current_price']:.2f})</p>
                            <p><strong>Alternative:</strong> {rec['alternative_vendor']} (${rec['alternative_price']:.2f})</p>
                            <p><strong>Potential Savings:</strong> ${rec['potential_savings']:.2f}</p>
                            <p><strong>Price Premium:</strong> {rec['price_premium_pct']:.1f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show detailed recommendations with examples
                st.write("**💰 High Savings Opportunities (Immediate Action):**")
                high_savings = switching_recs[switching_recs['annual_savings'] > 500]
                if not high_savings.empty:
                    for _, rec in high_savings.iterrows():
                        st.markdown(f"""
                        <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                            <h4 style="margin: 0 0 1rem 0; color: #155724;">🔄 {rec['priority']}</h4>
                            <p><strong>Product:</strong> {rec['product']}</p>
                            <p><strong>Category:</strong> {rec['category']}</p>
                            <p><strong>Current Vendor:</strong> {rec['current_vendor']} (${rec['current_price']:.2f})</p>
                            <p><strong>Alternative:</strong> {rec['alternative_vendor']} (${rec['alternative_price']:.2f})</p>
                            <p><strong>Price Premium:</strong> {rec['price_premium_pct']:.1f}%</p>
                            <p><strong>Annual Savings:</strong> ${rec['annual_savings']:.2f}</p>
                            <p><strong>Action Required:</strong> {rec['action_required']}</p>
                            <div style="background: #e8f5e8; border: 1px solid #d4edda; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                                <strong>💡 Restaurant Example:</strong><br>
                                {rec['example']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show all recommendations in a comprehensive table
                st.write("**📊 All Vendor Switching Recommendations:**")
                display_df = switching_recs[['product', 'category', 'current_vendor', 'alternative_vendor', 'annual_savings', 'price_premium_pct', 'priority']].copy()
                display_df['annual_savings'] = display_df['annual_savings'].round(2)
                display_df['price_premium_pct'] = display_df['price_premium_pct'].round(1)
                st.dataframe(display_df, use_container_width=True)
                
                # Total potential savings
                total_savings = switching_recs['potential_savings'].sum()
                st.success(f"**Total Potential Savings from Vendor Switching: ${total_savings:,.2f}**")
            else:
                st.info("No vendor switching recommendations available")
        
        with insights_tab3:
            st.markdown("### 💰 Data-Driven Strategic Insights from Line Items")
            st.info("🎯 **These insights are derived from your actual line items data, showing real spending patterns and optimization opportunities**")
            
            strategic_insights = dashboard.get_strategic_cost_optimization_insights()
            if not strategic_insights.empty:
                # Summary metrics for quick overview
                st.markdown("**📊 Quick Impact Summary**")
                col1, col2, col3, col4 = st.columns(4)
                
                high_impact_count = len(strategic_insights[strategic_insights['potential_impact'] == 'High'])
                medium_impact_count = len(strategic_insights[strategic_insights['potential_impact'] == 'Medium'])
                
                with col1:
                    st.metric("High Impact Items", high_impact_count, help="Requires immediate attention - highest potential savings")
                with col2:
                    st.metric("Medium Impact Items", medium_impact_count, help="Plan for optimization - good savings potential")
                with col3:
                    st.metric("Total Insights", len(strategic_insights), help="All optimization opportunities identified")
                with col4:
                    # Calculate estimated total potential savings
                    high_savings = high_impact_count * 0.15  # 15% average savings for high impact
                    medium_savings = medium_impact_count * 0.08  # 8% average savings for medium impact
                    total_potential_savings = high_savings + medium_savings
                    st.metric("Estimated Total Savings", f"{total_potential_savings:.1f}%", help="Combined potential savings from all insights")
                
                st.markdown("---")
                
                # Insights by Type Summary
                st.subheader("📈 Insights by Optimization Type")
                
                # Create a summary by insight type
                insight_summary = strategic_insights.groupby('insight_type').agg({
                    'vendor': 'count',
                    'potential_impact': lambda x: list(x)
                }).reset_index()
                insight_summary.columns = ['Insight Type', 'Count', 'Impact Levels']
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(insight_summary, use_container_width=True)
                with col2:
                    # Quick stats
                    st.metric("Total Products Analyzed", len(strategic_insights), 
                            help="Number of products/categories with optimization opportunities")
                    
                    # Calculate estimated total potential savings
                    high_savings = len(strategic_insights[strategic_insights['potential_impact'] == 'High']) * 0.15  # 15% average savings
                    medium_savings = len(strategic_insights[strategic_insights['potential_impact'] == 'Medium']) * 0.08  # 8% average savings
                    total_potential_savings = high_savings + medium_savings
                    
                    st.metric("Estimated Total Potential Savings", f"{total_potential_savings:.1f}%", 
                            help="Combined potential savings from all insights")
                
                st.markdown("---")
                
                # High impact insights
                high_impact = strategic_insights[strategic_insights['potential_impact'] == 'High']
                if not high_impact.empty:
                    st.error("**🚨 High Impact Insights - Immediate Action Required**")
                    st.info("These insights represent your biggest cost optimization opportunities based on actual spending data.")
                    
                    for _, insight in high_impact.iterrows():
                        # Safety check for optional columns
                        business_example = insight.get('business_example', 'No business example available')
                        action_items = insight.get('action_items', ['No specific action items available'])
                        
                        st.markdown(f"""
                        <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                            <h4 style="margin: 0 0 1rem 0; color: #721c24;">{insight['insight_type']}</h4>
                            <p><strong>Product/Category:</strong> {insight['vendor']}</p>
                            <p><strong>Spending Metric:</strong> {insight['metric']}</p>
                            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                                <strong>💡 Business Example:</strong><br>
                                {business_example}
                            </div>
                            <div style="background: #e8f5e8; border: 1px solid #d4edda; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                                <strong>📋 Action Items:</strong><br>
                                {chr(10).join([f"• {item}" for item in action_items])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Medium impact insights
                medium_impact = strategic_insights[strategic_insights['potential_impact'] == 'Medium']
                if not medium_impact.empty:
                    st.warning("**⚠️ Medium Impact Insights - Strategic Planning**")
                    st.info("These insights offer good optimization opportunities that should be planned over the next quarter.")
                    
                    for _, insight in medium_impact.iterrows():
                        # Safety check for optional columns
                        business_example = insight.get('business_example', 'No business example available')
                        action_items = insight.get('action_items', ['No specific action items available'])
                        
                        st.markdown(f"""
                        <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                            <h4 style="margin: 0 0 1rem 0; color: #856404;">{insight['insight_type']}</h4>
                            <p><strong>Product/Category:</strong> {insight['vendor']}</p>
                            <p><strong>Spending Metric:</strong> {insight['metric']}</p>
                            <p><strong>Recommendation:</strong> {insight['recommendation']}</p>
                            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                                <strong>💡 Business Example:</strong><br>
                                {business_example}
                            </div>
                            <div style="background: #e8f5e8; border: 1px solid #d4edda; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                                <strong>📋 Action Items:</strong><br>
                                {chr(10).join([f"• {item}" for item in action_items])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Complete Strategic Insights Summary
                st.subheader("📋 Complete Strategic Insights Summary")
                display_df = strategic_insights[['insight_type', 'vendor', 'metric', 'recommendation', 'potential_impact']].copy()
                st.dataframe(display_df, use_container_width=True)
                
                # Download option
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Strategic Insights as CSV",
                    data=csv,
                    file_name="strategic_insights_summary.csv",
                    mime="text/csv"
                )
            else:
                st.info("No strategic insights available. Check your data and try again.")
        
        with insights_tab4:
            st.markdown("### 📊 Time-Based Spending Analysis & Trends")
            st.markdown("**Restaurant spending patterns, seasonal trends, and optimization opportunities**")
            
            # Time-based analysis content
            time_insights = dashboard.get_time_based_optimization_insights()
            if not time_insights.empty:
                st.write("**Key Time-Based Insights:**")
                for _, insight in time_insights.iterrows():
                    st.info(f"**{insight['vendor']}**: {insight['insight_type']}")
                    st.write(f"   • **Recommendation**: {insight['recommendation']}")
                    st.write(f"   • **Metric**: {insight['metric']}")
                    st.write("---")
                
                # Additional time-based recommendations
                st.markdown("**🕒 Time-Based Business Recommendations:**")
                st.write("• **Inventory Planning**: Use seasonal patterns to optimize stock levels")
                st.write("• **Bulk Ordering**: High volatility periods are opportunities for bulk purchases")
                st.write("• **Vendor Scheduling**: Coordinate deliveries based on spending patterns")
                st.write("• **Cash Flow Management**: Plan for seasonal spending variations")
            else:
                st.info("No time-based insights available")
        
        # Time-based analysis is now integrated into the Strategic Insights tab above
        st.info("💡 **Time-based analysis has been integrated into the Strategic Insights section above for better organization and actionable insights.**")
        
        # Download combined report
        st.subheader("📥 Download Restaurant Supply Chain Report")
        report_data = {
            'Vendor': summary_df['vendor'],
            'Invoice_Count': summary_df['invoice_count'],
            'Total_Spending': summary_df['total_spending'],
            'Avg_Invoice_Value': summary_df['avg_invoice_value'],
            'Percentage_of_Total': summary_df.get('percentage', [0] * len(summary_df))
        }
        report_df = pd.DataFrame(report_data)
        csv = report_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Restaurant Supply Chain Report",
            data=csv,
            file_name="westmans_bagels_supply_chain_analysis.csv",
            mime="text/csv"
        )
            
    except Exception as e:
        st.error(f"Error generating combined analysis: {e}")
        st.info("Please ensure all vendor data is properly loaded.")

def main():
    st.set_page_config(
        page_title="Westman's Bagels - Vendor Management Dashboard",
        page_icon="🥯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Enhanced Custom CSS for restaurant dashboard
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #8B0000 0%, #DC143C 25%, #FF4500 50%, #DC143C 75%, #8B0000 100%);
        padding: 4rem 2rem;
        border-radius: 20px;
        margin: 2rem 0 3rem 0;
        text-align: center;
        color: white;
        box-shadow: 0 15px 50px rgba(139, 0, 0, 0.4), 0 8px 25px rgba(0, 0, 0, 0.3);
        border: 3px solid rgba(255, 255, 255, 0.25);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%, rgba(255, 255, 255, 0.1) 100%);
        pointer-events: none;
    }
    .main-header h1 {
        margin: 0;
        font-size: 3.5rem;
        font-weight: 900;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.5), 1px 1px 2px rgba(0, 0, 0, 0.3);
        font-family: 'Georgia', serif;
        color: #ffffff;
        position: relative;
        z-index: 2;
    }
    .main-header .restaurant-name {
        font-size: 2rem;
        font-weight: 700;
        margin: 1rem 0;
        opacity: 1;
        font-family: 'Georgia', serif;
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.4);
        position: relative;
        z-index: 2;
    }
    .main-header .dashboard-subtitle {
        margin: 1.5rem 0 0 0;
        font-size: 1.4rem;
        opacity: 1;
        font-weight: 400;
        color: #e8f4fd;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 2;
    }
    .main-header .business-info {
        background: rgba(255, 255, 255, 0.25);
        padding: 1.5rem;
        border-radius: 15px;
        margin-top: 2rem;
        display: inline-block;
        backdrop-filter: blur(15px);
        border: 2px solid rgba(255, 255, 255, 0.4);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2), 0 4px 15px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 2;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        text-align: center;
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #27ae60, #3498db);
    }
    .metric-card h3 {
        color: #495057;
        margin: 0 0 1rem 0;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-card h2 {
        color: #2c3e50;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .vendor-header {
        background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 12px 35px rgba(139, 0, 0, 0.4), 0 6px 20px rgba(0, 0, 0, 0.3);
        border: 3px solid rgba(255, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    .vendor-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, transparent 50%, rgba(255, 255, 255, 0.1) 100%);
        pointer-events: none;
    }
    .vendor-header h2 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        font-family: 'Georgia', serif;
        color: #ffffff;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7), 1px 1px 2px rgba(0, 0, 0, 0.5);
        position: relative;
        z-index: 2;
    }
    .vendor-header p {
        color: #ffffff;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        position: relative;
        z-index: 2;
        font-weight: 500;
    }
    .business-summary {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 50%, #e9ecef 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 2px solid #dee2e6;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08), 0 4px 15px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
    }
    .business-summary::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #8B0000, #DC143C, #FF4500);
    }
    .business-summary h3 {
        color: #2c3e50;
        margin: 0 0 1.5rem 0;
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Georgia', serif;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .business-summary p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #495057;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 8px;
        gap: 1rem;
        padding: 15px 20px;
        font-weight: 600;
        border: 1px solid #dee2e6;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(139, 0, 0, 0.3);
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        border-color: #2c3e50;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        color: white;
    }
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
        border: 2px solid #dee2e6;
    }
    .stSelectbox > div > div:hover {
        border-color: #2c3e50;
    }
    
    /* Enhanced visibility and animations */
    .main-header h1, .main-header .restaurant-name, .main-header .dashboard-subtitle {
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Ensure proper contrast and visibility */
    .main-header {
        position: relative;
        z-index: 10;
    }
    
    /* Enhanced text readability */
    .main-header * {
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.5rem;
        }
        .main-header .restaurant-name {
            font-size: 1.5rem;
        }
        .main-header .dashboard-subtitle {
            font-size: 1.1rem;
        }
        .vendor-header h2 {
            font-size: 2.2rem;
        }
        .vendor-header p {
            font-size: 1.1rem;
        }
    }
    
    /* Enhanced vendor header readability */
    .vendor-header h2, .vendor-header p {
        text-rendering: optimizeLegibility;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        letter-spacing: 0.5px;
    }
    
    /* Ensure vendor name stands out */
    .vendor-header h2 {
        background: linear-gradient(45deg, #ffffff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(3px 3px 6px rgba(0, 0, 0, 0.8));
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🥯 Westman's Bagels & Coffee</h1>
        <div class="restaurant-name">Vendor Management & Cost Analysis Dashboard</div>
        <div class="dashboard-subtitle">Professional restaurant supply chain optimization platform</div>
        <div class="business-info">
            <strong>🏪 Restaurant Operations Dashboard</strong> | 
            <strong>📊 Real-time Vendor Analytics</strong> | 
            <strong>💰 Cost Optimization Insights</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Business context and dashboard overview
    st.markdown("""
    <div class="business-summary">
        <h3>🏪 Restaurant Operations Overview</h3>
        <p><strong>Welcome to your comprehensive vendor management dashboard!</strong> This platform provides real-time insights into your restaurant's supply chain costs, helping you make data-driven decisions to optimize spending and improve profitability.</p>
        <p><strong>Key Features:</strong> Vendor performance analysis, cost optimization opportunities, spending trends, and strategic recommendations for Westman's Bagels & Coffee operations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    dashboard = VendorCentricDashboard()
    
    if not dashboard.connect_db():
        st.error("❌ Failed to connect to database.")
        return
    
    try:
        # Get available vendors
        vendors = dashboard.get_available_vendors()
        
        if not vendors:
            st.error("No vendors found in database.")
            return
        
        # Enhanced sidebar for restaurant operations
        st.sidebar.markdown("""
        <div style="background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem; box-shadow: 0 6px 20px rgba(139, 0, 0, 0.3);">
            <h2 style="margin: 0; text-align: center; font-family: 'Georgia', serif;">🥯 Westman's Bagels</h2>
            <p style="text-align: center; margin: 0.5rem 0; opacity: 0.9;">Vendor Management Dashboard</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("### 🎯 Vendor Selection")
        selected_vendor = st.sidebar.selectbox(
            "Choose a vendor to analyze:",
            vendors,
            index=0,
            help="Select any vendor to view detailed analysis, spending patterns, and optimization opportunities"
        )
        
        # Quick stats in sidebar
        if selected_vendor:
            vendor_stats = dashboard.get_vendor_overview(selected_vendor)
            if not vendor_stats.empty:
                stats = vendor_stats.iloc[0]
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 📊 Quick Stats")
                st.sidebar.metric("Total Spending", f"${stats['total_spending']:,.0f}")
                st.sidebar.metric("Invoice Count", stats['invoice_count'])
                st.sidebar.metric("Avg Invoice", f"${stats['avg_invoice_value']:,.0f}")
        
        # Restaurant operations info
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🏪 Restaurant Info")
        st.sidebar.info("""
        **Westman's Bagels & Coffee**  
        Professional vendor management platform  
        Real-time cost optimization insights
        """)
        
        if selected_vendor:
            show_vendor_analysis(dashboard, selected_vendor)
    
    finally:
        dashboard.disconnect_db()
    
    # Professional restaurant dashboard footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 3rem;">
        <h3 style="color: #2c3e50; font-family: Georgia, serif; margin-bottom: 1rem;">🥯 Westman's Bagels & Coffee</h3>
        <p style="color: #495057; margin: 0;"><strong>Professional Vendor Management Dashboard</strong> | 
        <strong>Real-time Cost Optimization</strong> | 
        <strong>Strategic Supply Chain Insights</strong></p>
        <p style="color: #6c757d; margin: 0.5rem 0 0 0; font-size: 0.9rem;">Powered by advanced analytics for restaurant operations excellence</p>
    </div>
    """, unsafe_allow_html=True)

def show_vendor_analysis(dashboard, vendor_name):
    """Show comprehensive analysis for the selected vendor."""
    
    try:
        # Enhanced Combined Analysis Section for Restaurant Operations
        st.markdown("---")
        with st.expander("🏪 **Restaurant-Wide Vendor Analysis** - Click to expand/collapse", expanded=True):
            st.markdown('<h3 style="text-align: center; color: #2c3e50; font-family: Georgia, serif;">🏪 Restaurant-Wide Vendor Analysis</h3>', unsafe_allow_html=True)
            st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #495057;"><strong>Comprehensive supply chain insights for Westman\'s Bagels & Coffee operations</strong></p>', unsafe_allow_html=True)
            
            # Show combined analysis
            show_combined_analysis(dashboard)
        
        st.markdown("---")
        
        # Enhanced vendor header for restaurant operations with better readability
        st.markdown(f"""
        <div class="vendor-header">
            <h2 style="color: #ffffff; text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7), 1px 1px 2px rgba(0, 0, 0, 0.5); font-size: 2.8rem; font-weight: 800; margin: 0;">🏢 {vendor_name}</h2>
            <p style="margin: 1rem 0 0 0; font-size: 1.3rem; color: #ffffff; opacity: 1; font-weight: 500; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);">Supply Chain Partner Analysis & Cost Optimization</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get vendor data
        vendor_overview = dashboard.get_vendor_overview(vendor_name)
        
        if vendor_overview.empty:
            st.error(f"No data found for {vendor_name}")
            return
        
        vendor_data = vendor_overview.iloc[0]
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📄 Total Invoices</h3>
                <h2>{vendor_data['invoice_count']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💰 Total Spending</h3>
                <h2>${vendor_data['total_spending']:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📦 Line Items</h3>
                <h2>{vendor_data['line_item_count']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_invoice = vendor_data['avg_invoice_value'] if pd.notna(vendor_data['avg_invoice_value']) else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Avg Invoice</h3>
                <h2>${avg_invoice:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Date range info with error handling
        try:
            if vendor_data['first_order'] and vendor_data['last_order'] and pd.notna(vendor_data['first_order']) and pd.notna(vendor_data['last_order']):
                st.info(f"**📅 Order Period**: {vendor_data['first_order']} to {vendor_data['last_order']}")
        except Exception as e:
            st.warning("**📅 Order Period**: Date information not available")
        
        st.markdown("---")
        
        # Enhanced analysis tabs for restaurant operations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Business Overview", 
            "🚨 Duplicate Detection", 
            "📅 Spending Trends", 
            "📦 Product Categories", 
            "💰 Cost Analysis", 
            "📋 Invoice Details",
            "🕒 Time Analysis"
        ])
        
        with tab1:
            show_vendor_overview(dashboard, vendor_name, vendor_data)
        
        with tab2:
            show_vendor_duplicates(dashboard, vendor_name)
        
        with tab3:
            show_vendor_trends(dashboard, vendor_name)
        
        with tab4:
            show_vendor_categories(dashboard, vendor_name)
        
        with tab5:
            show_vendor_pricing(dashboard, vendor_name)
        
        with tab6:
            show_vendor_details(dashboard, vendor_name)
        
        with tab7:
            show_vendor_time_analysis(dashboard, vendor_name)
            
    except Exception as e:
        st.error(f"Error displaying vendor analysis: {str(e)}")
        st.info("Please try refreshing the page or selecting a different vendor.")

def show_vendor_overview(dashboard, vendor_name, vendor_data):
    """Show vendor overview and insights."""
    st.subheader("📊 Vendor Overview & Insights")
    
    # Vendor summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Vendor Information**")
        st.write(f"• **Name**: {vendor_name}")
        st.write(f"• **Total Invoices**: {vendor_data['invoice_count']}")
        st.write(f"• **Total Spending**: ${vendor_data['total_spending']:,.2f}")
        st.write(f"• **Line Items**: {vendor_data['line_item_count']}")
        
        avg_invoice = vendor_data['avg_invoice_value'] if pd.notna(vendor_data['avg_invoice_value']) else 0
        st.write(f"• **Average Invoice**: ${avg_invoice:,.2f}")
    
    with col2:
        st.write("**Performance Metrics**")
        
        # Calculate efficiency metrics
        if vendor_data['invoice_count'] > 0:
            items_per_invoice = vendor_data['line_item_count'] / vendor_data['invoice_count']
            st.write(f"• **Items per Invoice**: {items_per_invoice:.1f}")
        
        if vendor_data['total_spending'] > 0:
            spending_efficiency = vendor_data['total_spending'] / vendor_data['invoice_count']
            st.write(f"• **Spending per Invoice**: ${spending_efficiency:,.2f}")
    
    # Vendor-specific insights
    st.subheader("🔍 Vendor-Specific Insights")
    
    if vendor_name == "Costco":
        st.markdown("""
        <div class="insight-card">
            <h4>🏆 Costco - Major Business Supplier</h4>
            <p><strong>Key Strengths:</strong> Bulk purchasing, wide product range, competitive pricing</p>
            <p><strong>Optimization Areas:</strong> Duplicate removal ✅, delivery consolidation, bulk ordering</p>
            <p><strong>Special Features:</strong> Multi-location delivery, route optimization opportunities</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif vendor_name == "Villa Jerada":
        st.markdown("""
        <div class="insight-card">
            <h4>🏪 Villa Jerada - Major Strategic Partner</h4>
            <p><strong>Key Strengths:</strong> High volume supplier, consistent delivery</p>
            <p><strong>Optimization Areas:</strong> Contract negotiations, volume pricing</p>
            <p><strong>Special Features:</strong> Second largest vendor, significant business impact</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif vendor_name == "Pacific Food Importers":
        st.markdown("""
        <div class="insight-card">
            <h4>🏪 PFI - Specialized Food Supplier</h4>
            <p><strong>Key Strengths:</strong> Specialized products, food industry expertise</p>
            <p><strong>Optimization Areas:</strong> Price volatility management, supplier negotiations</p>
            <p><strong>Special Features:</strong> Multi-invoice processing, specialized food products</p>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        st.markdown(f"""
        <div class="insight-card">
            <h4>🏢 {vendor_name} - Vendor Analysis</h4>
            <p><strong>Total Spending:</strong> ${vendor_data['total_spending']:,.2f}</p>
            <p><strong>Invoice Count:</strong> {vendor_data['invoice_count']}</p>
            <p><strong>Average Invoice:</strong> ${avg_invoice:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

def show_vendor_duplicates(dashboard, vendor_name):
    """Show duplicate analysis for a vendor."""
    st.subheader("🚨 Duplicate Analysis")
    
    duplicates_df = dashboard.get_vendor_duplicates(vendor_name)
    
    if duplicates_df.empty:
        st.success("✅ No duplicates found for this vendor!")
        return
    
    st.warning(f"⚠️ Found {len(duplicates_df)} potential duplicate invoices")
    
    # Show duplicates table
    st.dataframe(duplicates_df, use_container_width=True)
    
    # Summary metrics
    total_duplicate_value = duplicates_df['total_amount'].sum()
    st.info(f"**Total Value in Duplicates**: ${total_duplicate_value:,.2f}")
    
    # Recommendations
    st.subheader("💡 Recommendations")
    st.write("• Review duplicate invoices for accuracy")
    st.write("• Consider consolidating similar orders")
    st.write("• Implement duplicate prevention measures")

def show_vendor_trends(dashboard, vendor_name):
    """Show trend analysis for a vendor."""
    st.subheader("📅 Trend Analysis")
    
    # Monthly trends
    monthly_df = dashboard.get_vendor_monthly_trends(vendor_name)
    if not monthly_df.empty:
        st.write("**📊 Monthly Spending Trends**")
        
        # Create trend chart
        fig = px.line(monthly_df, x='month', y='total_spending', 
                     title=f"Monthly Spending - {vendor_name}",
                     markers=True)
        fig.update_layout(xaxis_title="Month", yaxis_title="Total Spending ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show data table
        st.dataframe(monthly_df, use_container_width=True)
    
    # Daily analysis
    daily_df = dashboard.get_vendor_daily_analysis(vendor_name)
    if not daily_df.empty:
        st.write("**📈 Daily Spending Analysis**")
        
        # Create daily chart
        fig = px.scatter(daily_df, x='date', y='total_spending', 
                        title=f"Daily Spending - {vendor_name}",
                        size='invoice_count')
        fig.update_layout(xaxis_title="Date", yaxis_title="Total Spending ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Weekly patterns
    weekly_df = dashboard.get_vendor_weekly_patterns(vendor_name)
    if not weekly_df.empty:
        st.write("**📅 Weekly Patterns**")
        
        # Create weekly chart
        fig = px.bar(weekly_df, x='week_number', y='total_spending',
                    title=f"Weekly Spending - {vendor_name}")
        fig.update_layout(xaxis_title="Week Number", yaxis_title="Total Spending ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    seasonal_df = dashboard.get_vendor_seasonal_analysis(vendor_name)
    if not seasonal_df.empty:
        st.write("**🌤️ Seasonal Analysis**")
        
        # Create seasonal chart
        fig = px.pie(seasonal_df, values='total_spending', names='season',
                    title=f"Seasonal Spending - {vendor_name}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show seasonal data
        st.dataframe(seasonal_df, use_container_width=True)

def show_vendor_categories(dashboard, vendor_name):
    """Show category breakdown for a vendor."""
    st.subheader("📦 Category Analysis")
    
    categories_df = dashboard.get_vendor_categories(vendor_name)
    
    if categories_df.empty:
        st.info("No category data available for this vendor")
        return
    
    # Category breakdown chart
    fig = px.pie(categories_df, values='total_value', names='category',
                title=f"Spending by Category - {vendor_name}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Category table
    st.write("**📊 Category Breakdown**")
    st.dataframe(categories_df, use_container_width=True)
    
    # Top categories
    if not categories_df.empty:
        top_category = categories_df.iloc[0]
        st.success(f"**Top Category**: {top_category['category']} - ${top_category['total_value']:,.2f}")

def show_vendor_pricing(dashboard, vendor_name):
    """Show pricing analysis for a vendor."""
    st.subheader("💰 Pricing Analysis")
    
    pricing_df = dashboard.get_vendor_pricing(vendor_name)
    
    if pricing_df.empty:
        st.info("No pricing data available for this vendor")
        return
    
    # Price variability analysis
    st.write("**📊 Price Variability Analysis**")
    
    # Show pricing data
    st.dataframe(pricing_df, use_container_width=True)
    
    # Price range insights
    if not pricing_df.empty:
        avg_price = pricing_df['avg_unit_price'].mean()
        price_range = pricing_df['max_unit_price'].max() - pricing_df['min_unit_price'].min()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Unit Price", f"${avg_price:.2f}")
        with col2:
            st.metric("Price Range", f"${price_range:.2f}")
        
        # Recommendations
        st.subheader("💡 Pricing Insights")
        if price_range > avg_price * 2:
            st.warning("⚠️ High price variability detected - consider price negotiations")
        else:
            st.success("✅ Price stability maintained")

def show_vendor_details(dashboard, vendor_name):
    """Show detailed invoice information for a vendor."""
    st.subheader("📋 Invoice Details")
    
    details_df = dashboard.get_vendor_details(vendor_name)
    
    if details_df.empty:
        st.error("No invoice details available")
        return
    
    # Show invoice details
    st.dataframe(details_df, use_container_width=True)
    
    # Summary statistics
    total_invoices = len(details_df)
    total_value = details_df['total_amount'].sum()
    avg_value = details_df['total_amount'].mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Invoices", total_invoices)
    with col2:
        st.metric("Total Value", f"${total_value:,.2f}")
    with col3:
        st.metric("Average Value", f"${avg_value:,.2f}")
    
    # Date range with proper error handling
    if not details_df.empty:
        try:
            # Convert date column to datetime if it's not already
            if details_df['date'].dtype == 'object':
                # Try to convert string dates to datetime
                details_df['date'] = pd.to_datetime(details_df['date'], errors='coerce')
            
            # Check if we have valid dates
            valid_dates = details_df['date'].dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                
                if pd.notna(min_date) and pd.notna(max_date):
                    date_range = max_date - min_date
                    st.info(f"**📅 Date Range**: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({date_range.days} days)")
                else:
                    st.info("**📅 Date Range**: Available but some dates may be invalid")
            else:
                st.info("**📅 Date Range**: No valid dates available")
                
        except Exception as e:
            st.warning(f"**📅 Date Range**: Could not calculate date range: {str(e)}")
            st.info("**📅 Date Range**: Check date format in database")


def show_vendor_time_analysis(dashboard, vendor_name):
    """Show comprehensive time-based analysis for a specific vendor."""
    st.subheader("🕒 Time-Based Analysis & Trends")
    st.markdown(f"**Detailed time analysis for {vendor_name} spending patterns**")
    
    # Time analysis tabs
    time_tab1, time_tab2, time_tab3, time_tab4 = st.tabs([
        "📅 Monthly Trends", 
        "🌤️ Seasonal Patterns", 
        "📊 Weekly Patterns", 
        "💡 Time Insights"
    ])
    
    with time_tab1:
        st.markdown("### 📅 Monthly Spending Trends")
        monthly_data = dashboard.get_vendor_monthly_trends(vendor_name)
        if not monthly_data.empty:
            # Monthly spending line chart
            fig_monthly = px.line(monthly_data, x='month', y='total_spending',
                                title=f"Monthly Spending Trends - {vendor_name}",
                                labels={'total_spending': 'Total Spending ($)', 'month': 'Month'})
            fig_monthly.update_layout(height=500)
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Monthly summary
            monthly_summary = monthly_data.groupby('month').agg({
                'total_spending': 'sum',
                'invoice_count': 'sum'
            }).reset_index()
            monthly_summary['avg_spending'] = monthly_summary['total_spending'] / monthly_summary['invoice_count']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Monthly Spending Summary:**")
                st.dataframe(monthly_summary, use_container_width=True)
            with col2:
                st.write("**Monthly Insights:**")
                peak_month = monthly_summary.loc[monthly_summary['total_spending'].idxmax()]
                st.success(f"**Peak Month**: {peak_month['month']} - ${peak_month['total_spending']:,.0f}")
                st.info(f"**Average Monthly Spending**: ${monthly_summary['avg_spending'].mean():,.0f}")
        else:
            st.info("No monthly trend data available")
    
    with time_tab2:
        st.markdown("### 🌤️ Seasonal Spending Patterns")
        seasonal_data = dashboard.get_vendor_seasonal_analysis(vendor_name)
        if not seasonal_data.empty:
            # Seasonal spending chart
            fig_seasonal = px.bar(seasonal_data, x='season', y='total_spending',
                                title=f"Seasonal Spending Patterns - {vendor_name}",
                                labels={'total_spending': 'Total Spending ($)', 'season': 'Season'})
            fig_seasonal.update_layout(height=500)
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Seasonal insights
            if len(seasonal_data) > 1:
                seasonal_variation = seasonal_data['total_spending'].max() / seasonal_data['total_spending'].min()
                st.info(f"**Seasonal Variation**: {seasonal_variation:.1f}x difference between highest and lowest seasons")
                
                peak_season = seasonal_data.loc[seasonal_data['total_spending'].idxmax()]
                st.success(f"**Peak Season**: {peak_season['season']} - ${peak_season['total_spending']:,.0f}")
        else:
            st.info("No seasonal data available")
    
    with time_tab3:
        st.markdown("### 📊 Weekly Spending Patterns")
        weekly_data = dashboard.get_vendor_weekly_patterns(vendor_name)
        if not weekly_data.empty:
            # Weekly spending trends
            fig_weekly = px.line(weekly_data, x='week_label', y='total_spending',
                               title=f"Weekly Spending Patterns - {vendor_name}",
                               labels={'total_spending': 'Total Spending ($)', 'week_label': 'Week'})
            fig_weekly.update_layout(height=500)
            st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Weekly statistics
            weekly_stats = weekly_data.groupby('year').agg({
                'total_spending': ['mean', 'std', 'min', 'max']
            }).round(2)
            weekly_stats.columns = ['Avg Weekly Spending', 'Std Dev', 'Min Weekly', 'Max Weekly']
            weekly_stats = weekly_stats.reset_index()
            
            st.write("**Weekly Spending Statistics by Year:**")
            st.dataframe(weekly_stats, use_container_width=True)
        else:
            st.info("No weekly pattern data available")
    
    with time_tab4:
        st.markdown("### 💡 Time-Based Business Insights")
        
        # Spending volatility analysis
        monthly_data = dashboard.get_vendor_monthly_trends(vendor_name)
        if not monthly_data.empty and len(monthly_data) > 1:
            spending_values = monthly_data['total_spending']
            volatility = spending_values.std() / spending_values.mean()
            
            st.write("**📊 Spending Volatility Analysis:**")
            if volatility > 0.5:
                st.warning(f"⚠️ **High Volatility**: Coefficient of variation = {volatility:.2f}")
                st.write("• **Recommendation**: Consider bulk ordering and inventory management")
                st.write("• **Opportunity**: High volatility periods are opportunities for bulk purchases")
            elif volatility > 0.3:
                st.info(f"📈 **Moderate Volatility**: Coefficient of variation = {volatility:.2f}")
                st.write("• **Recommendation**: Monitor spending patterns for optimization opportunities")
            else:
                st.success(f"✅ **Low Volatility**: Coefficient of variation = {volatility:.2f}")
                st.write("• **Recommendation**: Stable spending allows for predictable planning")
        
        # Time-based recommendations
        st.markdown("**🕒 Time-Based Recommendations for {vendor_name}:**")
        st.write("• **Inventory Planning**: Use seasonal patterns to optimize stock levels")
        st.write("• **Order Timing**: Schedule orders based on spending patterns")
        st.write("• **Cash Flow**: Plan for seasonal spending variations")
        st.write("• **Vendor Relations**: Coordinate with vendor based on usage patterns")


if __name__ == "__main__":
    main()
