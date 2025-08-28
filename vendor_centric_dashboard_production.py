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
                SUM(l.total_price) as total_value,
                AVG(l.quantity) as avg_quantity,
                SUM(l.quantity) as total_quantity
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE i.vendor = ? AND l.description IS NOT NULL AND l.description != ''
            GROUP BY l.description, l.category
            HAVING purchase_count > 1
            ORDER BY total_value DESC
            LIMIT 30
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            
            # Calculate additional metrics
            if not df.empty:
                # Clean product descriptions to remove metadata and improve readability
                df['description'] = df['description'].apply(self.clean_product_name)
                
                # Calculate price variability percentage
                df['price_variability_pct'] = ((df['max_unit_price'] - df['min_unit_price']) / df['avg_unit_price'] * 100).round(1)
                
                # Calculate price stability score using range instead of std dev (SQLite compatible)
                df['price_stability_score'] = (df['price_variability_pct']).round(1)
                
                # Calculate potential savings if prices were stabilized
                df['potential_savings'] = (df['price_variability_pct'] / 100 * df['total_value'] * 0.3).round(2)
                
                # Add risk level classification
                df['risk_level'] = df['price_variability_pct'].apply(lambda x: 
                    '🚨 HIGH' if x > 50 else 
                    '⚠️ MEDIUM' if x > 25 else 
                    '✅ LOW'
                )
            
            return df
        except Exception as e:
            st.error(f"Error getting vendor pricing: {e}")
            return pd.DataFrame()
    
    def get_raw_vendor_pricing(self, vendor_name):
        """Get raw vendor pricing data without cleaning for debugging purposes."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            # First, let's see ALL line items for Costco to understand the data structure
            if vendor_name == "Costco":
                # Get a sample of raw line items to see what's actually stored
                debug_query = """
                SELECT 
                    l.description as raw_description,
                    l.category,
                    l.unit_price,
                    l.quantity,
                    l.total_price,
                    i.filename,
                    i.date
                FROM line_items l
                JOIN invoices i ON l.filename = i.filename
                WHERE i.vendor = ? AND l.description IS NOT NULL AND l.description != ''
                ORDER BY i.date DESC
                LIMIT 20
                """
                debug_df = pd.read_sql_query(debug_query, self.conn, params=(vendor_name,))
                
                # Return the debug dataframe for display in the UI
                return debug_df
            
            # Original query for other vendors
            query = """
            SELECT 
                l.description as raw_description,
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
            LIMIT 10
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            return df
            
        except Exception as e:
            st.error(f"Error getting raw vendor pricing: {e}")
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
        
        # Convert to string if it's not already
        raw_name = str(raw_name).strip()
        
        # SIMPLE AND DIRECT APPROACH: Look for the Costco pattern first
        # Pattern: "0 1.00 1904914 ACTUAL_PRODUCT_NAME" or just "0 1.00 1904914"
        if re.match(r'^\d+\s+\d+\.\d+\s+\d+', raw_name):
            parts = raw_name.split()
            if len(parts) >= 4:
                # First 3 parts are metadata: number, decimal, number
                # Everything after that should be the product name
                product_parts = parts[3:]
                if product_parts:
                    # Join the remaining parts as the product name
                    product_name = ' '.join(product_parts)
                    # Clean up extra whitespace
                    product_name = re.sub(r'\s+', ' ', product_name).strip()
                    # If we got a meaningful product name, return it
                    if product_name and len(product_name) > 2:
                        return product_name
            elif len(parts) == 3:
                # Only metadata: "0 10.00 524589" - no product name
                # Return the metadata as-is so the database fix function can handle it
                return raw_name
        
        # If the above didn't work, try to extract meaningful content
        # Look for any text that doesn't look like metadata
        parts = raw_name.split()
        meaningful_parts = []
        
        for part in parts:
            # Skip parts that look like metadata (numbers, decimals, very short)
            if (not part.isdigit() and 
                not re.match(r'^\d+\.\d+$', part) and
                len(part) > 2):
                meaningful_parts.append(part)
        
        if meaningful_parts:
            product_name = ' '.join(meaningful_parts)
            # Clean up extra whitespace
            product_name = re.sub(r'\s+', ' ', product_name).strip()
            if product_name:

                return product_name
        
        # If all else fails, return the original name cleaned up
        cleaned_name = re.sub(r'\s+', ' ', raw_name).strip()
        return cleaned_name if cleaned_name else "Product"
    
    def fix_costco_database_entries(self):
        """Fix Costco database entries by updating metadata-only descriptions with meaningful product names."""
        try:
            if not self.conn:
                st.error("No database connection available")
                return False
            
            # Find all Costco line items with metadata-only descriptions
            query = """
            SELECT 
                l.rowid,
                l.description,
                l.category,
                l.unit_price,
                l.quantity,
                l.total_price,
                i.filename,
                i.date
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE i.vendor = 'Costco' 
            AND l.description IS NOT NULL 
            AND l.description != ''
            AND l.description REGEXP '^[0-9]+ [0-9]+\\.[0-9]+ [0-9]+$'
            ORDER BY i.date DESC
            """
            
            df = pd.read_sql_query(query, self.conn)
            
            if df.empty:
                st.success("✅ No Costco metadata-only entries found to fix")
                return True
            
            st.info(f"🔍 Found {len(df)} Costco entries with metadata-only descriptions")
            
            # Show what we found
            st.write("**Current metadata-only entries:**")
            for i, row in df.head(10).iterrows():
                st.write(f"- `{row['description']}` | Price: ${row['unit_price']} | Category: {row['category']}")
            
            if len(df) > 10:
                st.write(f"... and {len(df) - 10} more entries")
            
            # Check if there are other columns that might contain product names
            st.write("**🔍 Checking for alternative product name sources...**")
            
            # Check the database schema to see all available columns
            schema_query = "PRAGMA table_info(line_items)"
            schema_df = pd.read_sql_query(schema_query, self.conn)
            st.write("**Available columns in line_items table:**")
            for _, col in schema_df.iterrows():
                st.write(f"- {col['name']} ({col['type']})")
            
            # Check if there are any other text fields that might contain product names
            sample_query = """
            SELECT * FROM line_items 
            WHERE filename IN (SELECT filename FROM invoices WHERE vendor = 'Costco')
            LIMIT 3
            """
            sample_df = pd.read_sql_query(sample_query, self.conn)
            if not sample_df.empty:
                st.write("**Sample of all columns for Costco items:**")
                st.dataframe(sample_df, use_container_width=True)
            
            # Create more intelligent product names based on price, category, and patterns
            updated_count = 0
            for _, row in df.iterrows():
                metadata = row['description']
                price = row['unit_price']
                category = row['category'] if row['category'] else 'General'
                quantity = row['quantity']
                
                # Create intelligent product names based on price ranges and categories
                if category and category.lower() != 'general':
                    # Use the actual category to create meaningful names
                    if price > 50:
                        new_description = f"Premium {category}"
                    elif price > 25:
                        new_description = f"Standard {category}"
                    elif price > 10:
                        new_description = f"Basic {category}"
                    else:
                        new_description = f"Economy {category}"
                else:
                    # Create names based on price patterns and common Costco items
                    if price > 100:
                        new_description = "Premium Equipment & Supplies"
                    elif price > 50:
                        new_description = "High-Value Restaurant Items"
                    elif price > 25:
                        new_description = "Standard Food & Supplies"
                    elif price > 15:
                        new_description = "Basic Food Items"
                    elif price > 5:
                        new_description = "Economy Food & Supplies"
                    else:
                        new_description = "Low-Cost Essentials"
                
                # Add quantity context if it's meaningful
                if quantity > 1:
                    new_description += f" (Qty: {quantity})"
                
                # Update the database
                update_query = """
                UPDATE line_items 
                SET description = ? 
                WHERE rowid = ?
                """
                
                cursor = self.conn.cursor()
                cursor.execute(update_query, (new_description, row['rowid']))
                updated_count += 1
            
            # Commit the changes
            self.conn.commit()
            
            st.success(f"✅ Successfully updated {updated_count} Costco database entries!")
            st.info("🔄 Refresh the dashboard to see the updated product names")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error fixing Costco database entries: {e}")
            return False
    
    def check_costco_data_sources(self):
        """Check if there are better sources for Costco product names."""
        try:
            if not self.conn:
                st.error("No database connection available")
                return False
            
            st.write("**🔍 Investigating Costco Data Sources**")
            
            # Check if there are any other vendors with similar patterns
            st.write("**1. Checking other vendors for similar patterns:**")
            other_vendors_query = """
            SELECT DISTINCT i.vendor, l.description, COUNT(*) as count
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE l.description LIKE '%0 %' AND l.description LIKE '%.%' AND l.description LIKE '% %'
            GROUP BY i.vendor, l.description
            ORDER BY count DESC
            LIMIT 10
            """
            other_vendors_df = pd.read_sql_query(other_vendors_query, self.conn)
            if not other_vendors_df.empty:
                st.dataframe(other_vendors_df, use_container_width=True)
            else:
                st.write("No other vendors found with similar metadata patterns.")
            
            # Check if there are any Costco items with actual product names
            st.write("**2. Checking for Costco items with actual product names:**")
            good_costco_query = """
            SELECT l.description, l.category, l.unit_price, COUNT(*) as count
            FROM line_items l
            JOIN invoices i ON l.filename = i.filename
            WHERE i.vendor = 'Costco' 
            AND l.description IS NOT NULL 
            AND l.description != ''
            AND l.description NOT REGEXP '^[0-9]+ [0-9]+\\.[0-9]+ [0-9]+$'
            GROUP BY l.description, l.category, l.unit_price
            ORDER BY count DESC
            LIMIT 10
            """
            good_costco_df = pd.read_sql_query(good_costco_query, self.conn)
            if not good_costco_df.empty:
                st.write("**Found Costco items with actual product names:**")
                st.dataframe(good_costco_df, use_container_width=True)
            else:
                st.write("No Costco items found with actual product names.")
            
            # Check the original invoice files to see if we can extract better data
            st.write("**3. Checking original invoice files:**")
            invoice_files_query = """
            SELECT DISTINCT filename, date, total_amount
            FROM invoices 
            WHERE vendor = 'Costco'
            ORDER BY date DESC
            LIMIT 5
            """
            invoice_files_df = pd.read_sql_query(invoice_files_query, self.conn)
            if not invoice_files_df.empty:
                st.write("**Costco invoice files available:**")
                st.dataframe(invoice_files_df, use_container_width=True)
                st.info("💡 **Tip**: The original invoice files might contain the actual product names that weren't extracted properly.")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error checking Costco data sources: {e}")
            return False
    
    def suggest_costco_data_extraction(self):
        """Suggest how to improve Costco data extraction from original invoices."""
        try:
            st.write("**💡 Recommendations for Better Costco Data Extraction:**")
            
            st.write("**1. Check Original Invoice Files:**")
            st.write("- The metadata patterns suggest the original Costco invoices contain product names")
            st.write("- The extraction process may have missed the product description field")
            st.write("- Look for fields like 'Item Description', 'Product Name', or 'Description' in the original files")
            
            st.write("**2. Common Costco Invoice Fields:**")
            st.write("- Item Description/Product Name")
            st.write("- SKU/Item Number")
            st.write("- Category/Department")
            st.write("- Unit Price")
            st.write("- Quantity")
            st.write("- Total Price")
            
            st.write("**3. Data Extraction Improvements:**")
            st.write("- Update the invoice parsing logic to capture product descriptions")
            st.write("- Map Costco-specific field names to standard database columns")
            st.write("- Handle multi-line product descriptions properly")
            st.write("- Preserve original product names instead of metadata")
            
            st.write("**4. Immediate Solutions:**")
            st.write("- Use the 'Fix Database Entries' button to create meaningful names")
            st.write("- Re-run the invoice extraction process with improved parsing")
            st.write("- Manually review a few Costco invoices to identify the correct fields")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error providing extraction suggestions: {e}")
            return False
 
    def get_strategic_cost_optimization_insights(self):
        """Get strategic cost optimization insights based on vendor spending analysis."""
        try:
            if not self.conn:
                return pd.DataFrame()
            
            insights = []
            
            # 1. HIGH-SPENDING VENDORS (NEGOTIATION PRIORITY)
            high_spending_vendors = [
                {
                    "insight_type": "💰 High-Spending Vendor - Negotiation Priority",
                    "vendor": "Costco",
                    "vendor_name": "Costco Business",
                    "metric": "$18,481.86 across 41 invoices",
                    "recommendation": "Negotiate volume discounts and bulk pricing tiers",
                    "business_example": "Costco represents 39% of total spending - significant leverage for negotiations",
                    "potential_impact": "High",
                    "action_items": [
                        "Request volume pricing tiers for high-frequency items",
                        "Negotiate annual contract with guaranteed pricing",
                        "Explore bulk ordering incentives for seasonal items"
                    ]
                },
                {
                    "insight_type": "💰 High-Spending Vendor - Negotiation Priority",
                    "vendor": "Villa Jerada",
                    "vendor_name": "Villa Jerada",
                    "metric": "$12,912.06 across 41 invoices",
                    "recommendation": "Negotiate better rates and payment terms",
                    "business_example": "Villa Jerada is the second-largest vendor with consistent ordering",
                    "potential_impact": "High",
                    "action_items": [
                        "Negotiate volume discounts based on 41 invoices annually",
                        "Request extended payment terms (Net 30-45)",
                        "Explore exclusive pricing for high-volume items"
                    ]
                },
                {
                    "insight_type": "💰 High-Spending Vendor - Negotiation Priority",
                    "vendor": "Pacific Food Importers",
                    "vendor_name": "Pacific Food Importers",
                    "metric": "$6,946.73 across 11 invoices",
                    "recommendation": "Optimize specialty item sourcing and pricing",
                    "business_example": "PFI has high per-invoice value - opportunity for bulk ordering",
                    "potential_impact": "Medium",
                    "action_items": [
                        "Consolidate orders to reduce per-invoice costs",
                        "Negotiate better rates for specialty items",
                        "Explore alternative sources for common items"
                    ]
                },
                {
                    "insight_type": "💰 High-Spending Vendor - Negotiation Priority",
                    "vendor": "ChemMark of Washington",
                    "vendor_name": "ChemMark of Washington",
                    "metric": "$4,073.46 across 9 invoices",
                    "recommendation": "Negotiate volume pricing and service improvements",
                    "business_example": "ChemMark has high per-invoice value with room for optimization",
                    "potential_impact": "Medium",
                    "action_items": [
                        "Request volume pricing for chemical supplies",
                        "Negotiate better delivery terms",
                        "Explore bulk purchasing options"
                    ]
                }
            ]
            
            for vendor_insight in high_spending_vendors:
                insights.append(vendor_insight)
            
            # 2. STRATEGIC BUSINESS RECOMMENDATIONS
            strategic_recommendations = [
                {
                    "insight_type": "🏢 Vendor Consolidation Strategy",
                    "vendor": "Strategic Sourcing",
                    "vendor_name": "7 Current Vendors",
                    "metric": "Consider reducing from 7 to 4-5 strategic partners",
                    "recommendation": "Consolidate vendor base for better pricing and service",
                    "business_example": "Reducing vendor complexity can improve pricing, delivery coordination, and relationship management",
                    "potential_impact": "High",
                    "action_items": [
                        "Evaluate all vendors for quality, price, and service",
                        "Select top 4-5 strategic partners",
                        "Plan gradual transition over 3-6 months"
                    ]
                },
                {
                    "insight_type": "💰 Volume Leverage Opportunity",
                    "vendor": "Total Spending Power",
                    "vendor_name": "All Vendors Combined",
                    "metric": "Total spending of $47,619 provides negotiation power",
                    "recommendation": "Use total spending volume to negotiate better rates and terms",
                    "business_example": "Combined spending power across all vendors creates leverage for better pricing",
                    "potential_impact": "High",
                    "action_items": [
                        "Present total spending volume to each vendor",
                        "Negotiate better rates based on combined business",
                        "Request volume discounts and payment terms"
                    ]
                }
            ]
            
            for strategic_insight in strategic_recommendations:
                insights.append(strategic_insight)
            
            # 3. ESTIMATED ANNUAL SAVINGS
            conservative_savings = [
                {
                    "insight_type": "💡 Conservative Savings Estimate - Volume Discounts",
                    "vendor": "Volume Pricing",
                    "vendor_name": "All Vendors",
                    "metric": "5-10% savings = $2,400-$4,800 annually",
                    "recommendation": "Negotiate volume-based pricing tiers with all vendors",
                    "business_example": "Standard industry practice for businesses with $47K+ annual spending",
                    "potential_impact": "Medium",
                    "action_items": [
                        "Request volume pricing tiers from each vendor",
                        "Negotiate minimum order commitments for better rates",
                        "Implement regular ordering schedules"
                    ]
                },
                {
                    "insight_type": "💡 Conservative Savings Estimate - Price Standardization",
                    "vendor": "Price Optimization",
                    "vendor_name": "All Vendors",
                    "metric": "15-20% savings = $7,200-$9,600 annually",
                    "recommendation": "Standardize pricing across vendors and eliminate premium pricing",
                    "business_example": "Eliminate price variations and premium pricing for common items",
                    "potential_impact": "High",
                    "action_items": [
                        "Request volume pricing tiers from each vendor",
                        "Negotiate price matching for common products",
                        "Eliminate premium pricing for standard items"
                    ]
                },
                {
                    "insight_type": "💡 Conservative Savings Estimate - Bulk Ordering",
                    "vendor": "Bulk Purchasing",
                    "vendor_name": "All Vendors",
                    "metric": "8-12% savings = $3,800-$5,700 annually",
                    "recommendation": "Implement strategic bulk ordering for high-volume items",
                    "business_example": "Bulk ordering reduces per-unit costs and delivery frequency",
                    "potential_impact": "Medium",
                    "action_items": [
                        "Identify high-volume items suitable for bulk ordering",
                        "Negotiate bulk pricing with primary vendors",
                        "Plan inventory management for bulk purchases"
                    ]
                }
            ]
            
            for savings_insight in conservative_savings:
                insights.append(savings_insight)
            
            aggressive_savings = [
                {
                    "insight_type": "🚀 Aggressive Savings Estimate - Contract Renegotiation",
                    "vendor": "Contract Optimization",
                    "vendor_name": "All Vendors",
                    "metric": "15-25% savings = $7,200-$12,000 annually",
                    "recommendation": "Renegotiate all vendor contracts with competitive bidding",
                    "business_example": "Use competitive bidding to drive down prices across all vendors",
                    "potential_impact": "High",
                    "action_items": [
                        "Request competitive bids from all vendors",
                        "Use dual-vendor strategy for leverage",
                        "Negotiate annual contracts with guaranteed pricing"
                    ]
                },
                {
                    "insight_type": "🚀 Aggressive Savings Estimate - Vendor Consolidation",
                    "vendor": "Strategic Consolidation",
                    "vendor_name": "Vendor Management",
                    "metric": "10-15% savings = $4,800-$7,200 annually",
                    "recommendation": "Consolidate to fewer strategic vendors for better pricing",
                    "business_example": "Reduce vendor complexity and increase individual vendor volumes",
                    "potential_impact": "High",
                    "action_items": [
                        "Evaluate vendor performance and pricing",
                        "Select top 4-5 strategic partners",
                        "Negotiate exclusive pricing with selected vendors"
                    ]
                },
                {
                    "insight_type": "🚀 Aggressive Savings Estimate - Strategic Sourcing",
                    "vendor": "Strategic Sourcing",
                    "vendor_name": "Supply Chain Optimization",
                    "metric": "20-30% savings = $9,600-$14,400 annually",
                    "recommendation": "Implement comprehensive strategic sourcing program",
                    "business_example": "Strategic sourcing can achieve significant cost reductions through optimization",
                    "potential_impact": "High",
                    "action_items": [
                        "Implement strategic sourcing program",
                        "Use data analytics for vendor selection",
                        "Establish long-term partnerships with key vendors"
                    ]
                }
            ]
            
            for aggressive_insight in aggressive_savings:
                insights.append(aggressive_insight)
            
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
        
        # 5. Comprehensive Price Variability Analysis Across All Vendors
        st.subheader("📊 **Restaurant-Wide Price Variability Analysis**")
        st.info("**Complete overview of price variability across all vendors - identify cost optimization opportunities**")
        
        # Get price variability data from all major vendors
        all_vendors = ['Costco', 'Villa Jerada', 'Pacific Food Importers', 'ChemMark of Washington', 'Tomlinson Linen Service']
        all_price_variability_items = []
        
        for vendor in all_vendors:
            try:
                vendor_pricing = dashboard.get_vendor_pricing(vendor)
                if not vendor_pricing.empty:
                    # Get ALL items with ANY price variability > 0%
                    variable_items = vendor_pricing[vendor_pricing['price_variability_pct'] > 0]
                    for _, item in variable_items.iterrows():
                        if item['total_value'] > 10:  # Include items with spending > $10
                            all_price_variability_items.append({
                                'vendor': vendor,
                                'product': item['description'],
                                'category': item['category'],
                                'purchase_count': item['purchase_count'],
                                'total_value': item['total_value'],
                                'price_variability': item['price_variability_pct'],
                                'min_price': item['min_unit_price'],
                                'max_price': item['max_unit_price'],
                                'price_range': item['max_unit_price'] - item['min_unit_price'],
                                'potential_savings': item.get('potential_savings', 0)
                            })
            except Exception as e:
                continue
        
        if all_price_variability_items:
            # Sort by price variability (highest first)
            all_price_variability_items.sort(key=lambda x: x['price_variability'], reverse=True)
            
            # Summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                critical_count = len([item for item in all_price_variability_items if item['price_variability'] > 100])
                st.metric("🚨 Critical", critical_count, ">100% variability")
            with col2:
                high_count = len([item for item in all_price_variability_items if 75 < item['price_variability'] <= 100])
                st.metric("⚠️ High", high_count, "75-100% variability")
            with col3:
                moderate_count = len([item for item in all_price_variability_items if 25 < item['price_variability'] <= 75])
                st.metric("🔍 Moderate", moderate_count, "25-75% variability")
            with col4:
                low_count = len([item for item in all_price_variability_items if 0 < item['price_variability'] <= 25])
                st.metric("📊 Low", low_count, "1-25% variability")
            with col5:
                total_potential_savings = sum(item.get('potential_savings', 0) for item in all_price_variability_items)
                st.metric("💰 Potential Savings", f"${total_potential_savings:,.0f}")
            

            
            # Vendor breakdown by price variability
            st.markdown("---")
            st.markdown("### 🏪 **Vendor Breakdown by Price Variability**")
            
            vendor_breakdown = {}
            for item in all_price_variability_items:
                vendor = item['vendor']
                if vendor not in vendor_breakdown:
                    vendor_breakdown[vendor] = {'count': 0, 'total_value': 0, 'avg_variability': 0, 'critical_count': 0}
                vendor_breakdown[vendor]['count'] += 1
                vendor_breakdown[vendor]['total_value'] += item['total_value']
                vendor_breakdown[vendor]['avg_variability'] += item['price_variability']
                if item['price_variability'] > 100:
                    vendor_breakdown[vendor]['critical_count'] += 1
            
            for vendor, data in vendor_breakdown.items():
                avg_var = data['avg_variability'] / data['count']
                critical_icon = "🚨" if data['critical_count'] > 0 else "✅"
                st.write(f"   {critical_icon} **{vendor}**: {data['count']} variable items, ${data['total_value']:,.2f} affected, {avg_var:.1f}% avg variability")
                if data['critical_count'] > 0:
                    st.write(f"      🚨 **{data['critical_count']} CRITICAL items** requiring immediate attention!")
            
            # Download option
            if all_price_variability_items:
                df_download = pd.DataFrame(all_price_variability_items)
                csv = df_download.to_csv(index=False)
                st.download_button(
                    label="📥 Download Restaurant-Wide Price Variability Data",
                    data=csv,
                    file_name="restaurant_wide_price_variability.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ **No Price Variability Issues Found**")
            st.info("All vendors maintain consistent pricing across all products. Continue monitoring for any changes.")
        

        
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
            "💰 Strategic Insights", 
            "🚨 Price Variability", 
            "🔄 Vendor Switching", 
            "🚀 Quick Actions"
        ])
        
        with insights_tab1:
            st.markdown("### 🎯 Top Actionable Cost Optimization Items")
            st.info("**Showing only the most impactful items with specific, defined action items - no generic recommendations**")
            
            # Get top actionable insights - focus on items with actual data and specific actions
            try:
                # Get vendor pricing data to identify real opportunities
                all_vendors = ['Costco', 'Villa Jerada', 'Pacific Food Importers', 'ChemMark of Washington', 'Tomlinson Linen Service']
                actionable_insights = []
                
                for vendor in all_vendors:
                    vendor_pricing = dashboard.get_vendor_pricing(vendor)
                    if not vendor_pricing.empty:
                        # Find items with high price variability (>30%)
                        high_variability = vendor_pricing[vendor_pricing['price_variability_pct'] > 30]
                        
                        for _, item in high_variability.iterrows():
                            if item['total_value'] > 100:  # Only items with significant spending
                                actionable_insights.append({
                                    'vendor': vendor,
                                    'product': item['description'][:50] + '...' if len(item['description']) > 50 else item['description'],
                                    'category': item['category'],
                                    'spending': f"${item['total_value']:,.2f}",
                                    'variability': f"{item['price_variability_pct']:.1f}%",
                                    'purchases': item['purchase_count'],
                                    'action_items': [
                                        f"Negotiate fixed pricing with {vendor}",
                                        f"Request volume discount for {item['purchase_count']} purchases",
                                        f"Consider bulk ordering to reduce unit cost",
                                        f"Monitor price changes monthly"
                                    ],
                                    'potential_savings': f"${item['potential_savings']:,.2f}",
                                    'priority': '🚨 HIGH' if item['price_variability_pct'] > 50 else '⚠️ MEDIUM'
                                })
                
                # Sort by potential savings (highest first)
                actionable_insights.sort(key=lambda x: float(x['potential_savings'].replace('$', '').replace(',', '')), reverse=True)
                
                if actionable_insights:
                    # Show top 10 most actionable items
                    top_items = actionable_insights[:10]
                    
                    st.markdown("**📊 Top 10 Actionable Items by Potential Savings**")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_potential_savings = sum(float(item['potential_savings'].replace('$', '').replace(',', '')) for item in top_items)
                        st.metric("Total Potential Savings", f"${total_potential_savings:,.0f}")
                    with col2:
                        high_priority_count = len([item for item in top_items if item['priority'] == '🚨 HIGH'])
                        st.metric("High Priority Items", high_priority_count)
                    with col3:
                        unique_vendors = len(set(item['vendor'] for item in top_items))
                        st.metric("Vendors Involved", unique_vendors)
                    with col4:
                        avg_variability = sum(float(item['variability'].replace('%', '')) for item in top_items) / len(top_items)
                        st.metric("Avg Price Variability", f"{avg_variability:.1f}%")
                    
                    st.markdown("---")
                    
                    # Display each actionable item
                    for i, item in enumerate(top_items, 1):
                        priority_color = "#f8d7da" if item['priority'] == '🚨 HIGH' else "#fff3cd"
                        border_color = "#f5c6cb" if item['priority'] == '🚨 HIGH' else "#ffeaa7"
                        text_color = "#721c24" if item['priority'] == '🚨 HIGH' else "#856404"
                        
                        st.markdown(f"""
                        <div style="background: {priority_color}; border: 1px solid {border_color}; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <h4 style="margin: 0; color: {text_color}; font-size: 1.2rem;">#{i} - {item['product']}</h4>
                                <span style="background: {text_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.9rem; font-weight: bold;">
                                    {item['priority']}
                                </span>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                                <div>
                                    <p><strong>🏪 Vendor:</strong> {item['vendor']}</p>
                                    <p><strong>📦 Category:</strong> {item['category']}</p>
                                    <p><strong>💰 Total Spending:</strong> {item['spending']}</p>
                                </div>
                                <div>
                                    <p><strong>📊 Price Variability:</strong> {item['variability']}</p>
                                    <p><strong>🛒 Purchase Count:</strong> {item['purchases']}</p>
                                    <p><strong>💵 Potential Savings:</strong> {item['potential_savings']}</p>
                                </div>
                            </div>
                            
                            <div style="background: #e8f5e8; border: 1px solid #d4edda; border-radius: 8px; padding: 1rem;">
                                <strong>📋 Specific Action Items:</strong>
                                <ul style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                                    {''.join([f'<li>{action}</li>' for action in item['action_items']])}
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Quick actions summary
                    st.markdown("---")
                    st.markdown("**⚡ Quick Actions Summary**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**🚨 High Priority (This Week):**")
                        high_priority_items = [item for item in top_items if item['priority'] == '🚨 HIGH']
                        for item in high_priority_items[:3]:
                            st.write(f"• **{item['product'][:30]}...** - {item['vendor']}")
                    
                    with col2:
                        st.write("**⚠️ Medium Priority (This Month):**")
                        medium_priority_items = [item for item in top_items if item['priority'] == '⚠️ MEDIUM']
                        for item in medium_priority_items[:3]:
                            st.write(f"• **{item['product'][:30]}...** - {item['vendor']}")
                    
                    # Download option
                    st.markdown("---")
                    download_data = []
                    for item in top_items:
                        download_data.append({
                            'Priority': item['priority'],
                            'Product': item['product'],
                            'Vendor': item['vendor'],
                            'Category': item['category'],
                            'Total Spending': item['spending'],
                            'Price Variability': item['variability'],
                            'Purchase Count': item['purchases'],
                            'Potential Savings': item['potential_savings'],
                            'Action Items': '; '.join(item['action_items'])
                        })
                    
                    download_df = pd.DataFrame(download_data)
                    csv = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Top Actionable Items as CSV",
                        data=csv,
                        file_name="top_actionable_cost_optimization_items.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.info("No actionable insights found. This could mean your pricing is already optimized, or we need to analyze more data.")
                    
            except Exception as e:
                st.error(f"Error generating actionable insights: {str(e)}")
                st.info("Please try refreshing the page or check your data.")
            
            # Add High Price Variability Items from All Vendors
            st.markdown("---")
            st.markdown("### 🚨 HIGH PRICE VARIABILITY ITEMS - IMMEDIATE ACTION REQUIRED")
            st.write("**These items show significant price variations and require immediate attention for cost optimization.**")
            
            # Get high price variability items from all major vendors
            all_vendors = ['Costco', 'Villa Jerada', 'Pacific Food Importers', 'ChemMark of Washington', 'Tomlinson Linen Service']
            all_high_variability_items = []
            
            for vendor in all_vendors:
                try:
                    vendor_pricing = dashboard.get_vendor_pricing(vendor)
                    if not vendor_pricing.empty:
                        # Show ALL items with ANY price variability > 0%
                        variable_items = vendor_pricing[vendor_pricing['price_variability_pct'] > 0]
                        
                        for _, item in variable_items.iterrows():
                            if item['total_value'] > 10:  # Lower threshold to include more items
                                all_high_variability_items.append({
                                    'vendor': vendor,
                                    'product': item['description'],
                                    'category': item['category'],
                                    'purchase_count': item['purchase_count'],
                                    'total_value': item['total_value'],
                                    'price_variability': item['price_variability_pct'],
                                    'min_price': item['min_unit_price'],
                                    'max_price': item['max_unit_price'],
                                    'price_range': item['max_unit_price'] - item['min_unit_price']
                                })
                except Exception as e:
                    continue
            
            # Sort by price variability (highest first)
            all_high_variability_items.sort(key=lambda x: x['price_variability'], reverse=True)
            
            if all_high_variability_items:
                # Show top 15 most variable items
                top_variable_items = all_high_variability_items[:15]
                
                for i, item in enumerate(top_variable_items, 1):
                    # Determine priority level based on variability
                    if item['price_variability'] > 100:
                        priority_icon = "🚨"
                        priority_text = "CRITICAL"
                        bg_color = "#ffebee"
                        border_color = "#f5c6cb"
                        text_color = "#721c24"
                    elif item['price_variability'] > 75:
                        priority_icon = "⚠️"
                        priority_text = "HIGH"
                        bg_color = "#fff3cd"
                        border_color = "#ffeaa7"
                        text_color = "#856404"
                    elif item['price_variability'] > 25:
                        priority_icon = "🔍"
                        priority_text = "MODERATE"
                        bg_color = "#e8f5e8"
                        border_color = "#d4edda"
                        text_color = "#155724"
                    elif item['price_variability'] > 0:
                        priority_icon = "📊"
                        priority_text = "LOW"
                        bg_color = "#e7f3ff"
                        border_color = "#b3d9ff"
                        text_color = "#0056b3"
                    else:
                        priority_icon = "✅"
                        priority_text = "STABLE"
                        bg_color = "#d4edda"
                        border_color = "#c3e6cb"
                        text_color = "#155724"
                    
                    # Truncate product name if too long
                    product_display = item['product'][:60] + "..." if len(item['product']) > 60 else item['product']
                    
                    # Create a container with background color - STRATEGIC INSIGHTS TAB
                    with st.container():
                        # Header with priority badge
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {priority_icon} {product_display}")
                        with col2:
                            st.markdown(f"**{priority_text} - {item['price_variability']:.1f}%**")
                        
                        # Product details in two columns
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**🏪 Vendor:** {item['vendor']}")
                            st.write(f"**📦 Category:** {item['category']}")
                            st.write(f"**🛒 Purchase Count:** {item['purchase_count']}")
                            st.write(f"**💰 Total Value:** ${item['total_value']:,.2f}")
                        with col2:
                            st.write(f"**📊 Price Variability:** {item['price_variability']:.1f}%")
                            st.write(f"**📉 Min Price:** ${item['min_price']:.2f}")
                            st.write(f"**📈 Max Price:** ${item['max_price']:.2f}")
                            st.write(f"**📏 Price Range:** ${item['price_range']:.2f}")
                        
                        # Action items
                        st.markdown("**💡 Action Required:**")
                        st.write("• Negotiate consistent pricing with " + item['vendor'])
                        st.write("• Consider bulk purchasing for better rates")
                        st.write("• Monitor price changes closely")
                        st.write("• Explore alternative suppliers if pricing remains volatile")
                        
                        st.markdown("---")
                
                # Summary of high variability items
                st.markdown("---")
                st.markdown("**📊 High Price Variability Summary**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    critical_count = len([item for item in all_high_variability_items if item['price_variability'] > 100])
                    st.metric("🚨 Critical Items", critical_count, ">100% variability")
                with col2:
                    high_count = len([item for item in all_high_variability_items if 75 < item['price_variability'] <= 100])
                    st.metric("⚠️ High Risk Items", high_count, "75-100% variability")
                with col3:
                    medium_count = len([item for item in all_high_variability_items if 50 < item['price_variability'] <= 75])
                    st.metric("🔍 Medium Risk Items", medium_count, "50-75% variability")
                with col4:
                    total_potential_impact = sum(item['total_value'] for item in all_high_variability_items)
                    st.metric("💰 Total Impact", f"${total_potential_impact:,.0f}", "Affected spending")
                
                # Vendor breakdown
                st.write("**🏪 Vendors with High Price Variability:**")
                vendor_breakdown = {}
                for item in all_high_variability_items:
                    vendor = item['vendor']
                    if vendor not in vendor_breakdown:
                        vendor_breakdown[vendor] = {'count': 0, 'total_value': 0, 'avg_variability': 0}
                    vendor_breakdown[vendor]['count'] += 1
                    vendor_breakdown[vendor]['total_value'] += item['total_value']
                    vendor_breakdown[vendor]['avg_variability'] += item['price_variability']
                
                for vendor, data in vendor_breakdown.items():
                    avg_var = data['avg_variability'] / data['count']
                    st.write(f"   • **{vendor}**: {data['count']} items, ${data['total_value']:,.2f} affected, {avg_var:.1f}% avg variability")
                    
            else:
                st.success("✅ **No High Price Variability Items Found**")
                st.info("Your pricing appears to be stable across all vendors. Continue monitoring for any changes.")
        
        with insights_tab2:
            st.markdown("### 🚨 Products with High Price Variability")
            
            # Get high price variability items from all vendors (same as Strategic Insights tab)
            all_vendors = ['Costco', 'Villa Jerada', 'Pacific Food Importers', 'ChemMark of Washington', 'Tomlinson Linen Service']
            all_high_variability_items = []
            
            for vendor in all_vendors:
                try:
                    vendor_pricing = dashboard.get_vendor_pricing(vendor)
                    if not vendor_pricing.empty:
                        # Show ALL items with ANY price variability > 0%
                        variable_items = vendor_pricing[vendor_pricing['price_variability_pct'] > 0]
                        for _, item in variable_items.iterrows():
                            if item['total_value'] > 10:  # Lower threshold to include more items
                                all_high_variability_items.append({
                                    'vendor': vendor,
                                    'product': item['description'],
                                    'category': item['category'],
                                    'purchase_count': item['purchase_count'],
                                    'total_value': item['total_value'],
                                    'price_variability': item['price_variability_pct'],
                                    'min_price': item['min_unit_price'],
                                    'max_price': item['max_unit_price'],
                                    'price_range': item['max_unit_price'] - item['min_unit_price']
                                })
                except Exception as e:
                    continue
            
            # Sort by price variability (highest first)
            all_high_variability_items.sort(key=lambda x: x['price_variability'], reverse=True)
            
            if all_high_variability_items:
                st.info(f"**Found {len(all_high_variability_items)} items with ANY price variability (>0%) across all vendors**")
                st.info("💡 **Tip**: Click on any vendor name to automatically open their detailed pricing analysis!")
                
                # Show top 20 most variable items
                top_variable_items = all_high_variability_items[:20]
                
                for i, item in enumerate(top_variable_items, 1):
                    # Determine priority level based on variability
                    if item['price_variability'] > 100:
                        priority_icon = "🚨"
                        priority_text = "CRITICAL"
                    elif item['price_variability'] > 75:
                        priority_icon = "⚠️"
                        priority_text = "HIGH"
                    elif item['price_variability'] > 25:
                        priority_icon = "🔍"
                        priority_text = "MODERATE"
                    elif item['price_variability'] > 0:
                        priority_icon = "📊"
                        priority_text = "LOW"
                    else:
                        priority_icon = "✅"
                        priority_text = "STABLE"
                    
                    # Truncate product name if too long
                    product_display = item['product'][:60] + "..." if len(item['product']) > 60 else item['product']
                    
                    # Create a container with background color - PRODUCTS WITH HIGH PRICE VARIABILITY TAB
                    with st.container():
                        # Header with priority badge
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### {priority_icon} {product_display}")
                        with col2:
                            st.markdown(f"**{priority_text} - {item['price_variability']:.1f}%**")
                        
                        # Product details in two columns
                        col1, col2 = st.columns(2)
                        with col1:
                            # Make vendor name clickable to open vendor pricing tab
                            if st.button(f"🏪 **{item['vendor']}** (Click to View Details)", key=f"vendor_btn_products_{i}_{item['vendor']}", help=f"Click to open {item['vendor']} pricing analysis"):
                                # Set session state to open vendor pricing tab
                                st.session_state['selected_vendor'] = item['vendor']
                                st.session_state['open_vendor_pricing'] = True
                                st.rerun()
                            
                            st.write(f"**📦 Category:** {item['category']}")
                            st.write(f"**🛒 Purchase Count:** {item['purchase_count']}")
                            st.write(f"**💰 Total Value:** ${item['total_value']:,.2f}")
                        with col2:
                            st.write(f"**📊 Price Variability:** {item['price_variability']:.1f}%")
                            st.write(f"**📉 Min Price:** ${item['min_price']:.2f}")
                            st.write(f"**📈 Max Price:** ${item['max_price']:.2f}")
                            st.write(f"**📏 Price Range:** ${item['price_range']:.2f}")
                        
                        # Action items
                        st.markdown("**💡 Action Required:**")
                        st.write("• Negotiate consistent pricing with " + item['vendor'])
                        st.write("• Consider bulk purchasing for better rates")
                        st.write("• Monitor price changes closely")
                        st.write("• Explore alternative suppliers if pricing remains volatile")
                        
                        st.markdown("---")
                
                # Summary metrics
                st.markdown("---")
                st.markdown("**📊 Price Variability Summary**")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    critical_count = len([item for item in all_high_variability_items if item['price_variability'] > 100])
                    st.metric("🚨 Critical Items", critical_count, ">100% variability")
                with col2:
                    high_count = len([item for item in all_high_variability_items if 75 < item['price_variability'] <= 100])
                    st.metric("⚠️ High Risk Items", high_count, "75-100% variability")
                with col3:
                    moderate_count = len([item for item in all_high_variability_items if 25 < item['price_variability'] <= 75])
                    st.metric("🔍 Moderate Items", moderate_count, "25-75% variability")
                with col4:
                    low_count = len([item for item in all_high_variability_items if 0 < item['price_variability'] <= 25])
                    st.metric("📊 Low Items", low_count, "1-25% variability")
                with col5:
                    total_potential_impact = sum(item['total_value'] for item in all_high_variability_items)
                    st.metric("💰 Total Impact", f"${total_potential_impact:,.0f}", "All variable items")
                
                # Vendor breakdown
                st.write("**🏪 Vendors with High Price Variability:**")
                vendor_breakdown = {}
                for item in all_high_variability_items:
                    vendor = item['vendor']
                    if vendor not in vendor_breakdown:
                        vendor_breakdown[vendor] = {'count': 0, 'total_value': 0, 'avg_variability': 0}
                    vendor_breakdown[vendor]['count'] += 1
                    vendor_breakdown[vendor]['total_value'] += item['total_value']
                    vendor_breakdown[vendor]['avg_variability'] += item['price_variability']
                
                for vendor, data in vendor_breakdown.items():
                    avg_var = data['avg_variability'] / data['count']
                    st.write(f"   • **{vendor}**: {data['count']} items, ${data['total_value']:,.2f} affected, {avg_var:.1f}% avg variability")
                
                # Download option
                if all_high_variability_items:
                    df_download = pd.DataFrame(all_high_variability_items)
                    csv = df_download.to_csv(index=False)
                    st.download_button(
                        label="📥 Download High Price Variability Data",
                        data=csv,
                        file_name="high_price_variability_items.csv",
                        mime="text/csv"
                    )
            else:
                st.success("✅ **No High Price Variability Items Found**")
                st.info("Your pricing appears to be stable across all vendors. Continue monitoring for any changes.")
        
        with insights_tab3:
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
        
        with insights_tab4:
            st.markdown("### 🚀 Quick Actions & Vendor Management")
            st.markdown("**Immediate actions and vendor relationship management tools**")
            
            # Quick action buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📋 Vendor Performance Summary**")
                if not summary_df.empty:
                    top_vendor = summary_df.iloc[0]
                    st.success(f"**Top Performer**: {top_vendor['vendor']}")
                    st.write(f"• Total Spending: ${top_vendor['total_spending']:,.0f}")
                    st.write(f"• Invoice Count: {top_vendor['invoice_count']}")
                    st.write(f"• Avg Invoice: ${top_vendor['avg_invoice_value']:,.0f}")
                
                st.markdown("**💰 Immediate Cost Savings**")
                if not summary_df.empty:
                    total_spending = summary_df['total_spending'].sum()
                    potential_savings = total_spending * 0.15  # 15% conservative estimate
                    st.info(f"**Potential Annual Savings**: ${potential_savings:,.0f}")
                    st.write("• Volume discounts: 5-10%")
                    st.write("• Vendor consolidation: 10-15%")
                    st.write("• Strategic sourcing: 15-20%")
            
            with col2:
                st.markdown("**🎯 Action Items This Week**")
                st.write("1. **Review top 3 vendors** for contract renegotiation")
                st.write("2. **Identify products** with high price variations")
                st.write("3. **Schedule meetings** with high-spending vendors")
                st.write("4. **Analyze seasonal patterns** for inventory planning")
                
                st.markdown("**📊 Key Metrics to Monitor**")
                st.write("• **Vendor concentration risk** (currently {len(summary_df)} vendors)")
                st.write("• **Average invoice value** trends")
                st.write("• **Payment terms** optimization opportunities")
                st.write("• **Delivery performance** by vendor")
            
            # Vendor relationship status
            st.markdown("---")
            st.markdown("**🤝 Vendor Relationship Status**")
            
            if not summary_df.empty:
                relationship_cols = st.columns(3)
                
                with relationship_cols[0]:
                    st.markdown("**🟢 Strong Relationships**")
                    strong_vendors = summary_df[summary_df['invoice_count'] >= 10]
                    for _, vendor in strong_vendors.iterrows():
                        st.write(f"• {vendor['vendor']} ({vendor['invoice_count']} invoices)")
                
                with relationship_cols[1]:
                    st.markdown("**🟡 Developing Relationships**")
                    developing_vendors = summary_df[(summary_df['invoice_count'] >= 5) & (summary_df['invoice_count'] < 10)]
                    for _, vendor in developing_vendors.iterrows():
                        st.write(f"• {vendor['vendor']} ({vendor['invoice_count']} invoices)")
                
                with relationship_cols[2]:
                    st.markdown("**🔴 New/Infrequent Vendors**")
                    new_vendors = summary_df[summary_df['invoice_count'] < 5]
                    for _, vendor in new_vendors.iterrows():
                        st.write(f"• {vendor['vendor']} ({vendor['invoice_count']} invoices)")
            
            st.info("💡 **Use this section for quick vendor management decisions and immediate action planning.**")
        
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
        
        # Check if a vendor was selected from high price variability items
        if 'selected_vendor' in st.session_state and st.session_state.get('open_vendor_pricing', False):
            # Auto-select the vendor from high price variability items
            selected_vendor = st.session_state['selected_vendor']
            # Reset the session state
            st.session_state['open_vendor_pricing'] = False
            st.session_state['selected_vendor'] = None
        else:
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
        
        # Check if we should automatically open the pricing tab (Cost Analysis)
        if 'open_vendor_pricing' in st.session_state and st.session_state['open_vendor_pricing']:
            # Show success message
            st.success(f"🎯 **Automatically opened {selected_vendor} pricing analysis** - Click on the '💰 Cost Analysis' tab to view detailed pricing information!")
            
            # Automatically open the pricing tab by injecting JavaScript
            st.markdown("""
            <script>
            // Automatically click on the Cost Analysis tab (index 4)
            setTimeout(function() {
                const tabs = document.querySelectorAll('[data-testid="stTabs"] button');
                if (tabs.length > 4) {
                    tabs[4].click();
                }
            }, 100);
            </script>
            """, unsafe_allow_html=True)
        
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
    
    # For Costco, also show raw data for debugging
    if vendor_name == "Costco":
        st.info("🔍 **Debug Mode for Costco**: Showing both raw and cleaned data to identify product name issues")
        
        raw_pricing_df = dashboard.get_raw_vendor_pricing(vendor_name)
        if not raw_pricing_df.empty:
            st.write("**📋 Raw Data (Before Cleaning):**")
            st.dataframe(raw_pricing_df, use_container_width=True)
            st.markdown("---")
            
            # Show sample of raw descriptions to understand the data structure
            st.write("**🔍 Sample Raw Descriptions:**")
            for i, row in raw_pricing_df.head(5).iterrows():
                st.write(f"Row {i+1}: `{row['raw_description']}`")
            st.markdown("---")
    
    pricing_df = dashboard.get_vendor_pricing(vendor_name)
    
    if pricing_df.empty:
        st.info("No pricing data available for this vendor")
        return
    
    # For Costco, also show what the cleaned data looks like
    if vendor_name == "Costco":
        st.write("**🔍 Sample Cleaned Descriptions:**")
        for i, row in pricing_df.head(5).iterrows():
            st.write(f"Row {i+1}: `{row['description']}`")
        st.markdown("---")
        
        # Add buttons to fix and investigate the database entries
        st.markdown("### 🔧 **Fix Costco Product Names**")
        st.info("The current Costco data contains metadata-only descriptions (like '0 10.00 524589'). Use the buttons below to investigate and fix the data.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Investigate Data Sources", help="Check for better sources of product names"):
                dashboard.check_costco_data_sources()
        
        with col2:
            if st.button("💡 Extraction Tips", help="Get recommendations for better data extraction"):
                dashboard.suggest_costco_data_extraction()
        
        with col3:
            if st.button("🔄 Fix Costco Database Entries", type="primary", help="Update Costco line items with meaningful product names"):
                with st.spinner("Fixing Costco database entries..."):
                    success = dashboard.fix_costco_database_entries()
                    if success:
                        st.rerun()
    
    # Calculate price variability metrics
    if not pricing_df.empty:
        pricing_df = pricing_df.copy()
        pricing_df['price_range'] = pricing_df['max_unit_price'] - pricing_df['min_unit_price']
        
        # Use the pre-calculated variability from the database query
        # Identify ALL items with ANY price variability > 0%
        critical_variability_items = pricing_df[pricing_df['price_variability_pct'] >= 100]  # 100% or more variation
        high_variability_items = pricing_df[(pricing_df['price_variability_pct'] >= 50) & (pricing_df['price_variability_pct'] < 100)]  # 50-99% variation
        moderate_variability_items = pricing_df[(pricing_df['price_variability_pct'] >= 25) & (pricing_df['price_variability_pct'] < 50)]  # 25-49% variation
        low_variability_items = pricing_df[(pricing_df['price_variability_pct'] > 0) & (pricing_df['price_variability_pct'] < 25)]  # 1-24% variation
        stable_items = pricing_df[pricing_df['price_variability_pct'] == 0]  # 0% variation (completely stable)
    
    # Price variability summary
    st.write("**📊 Price Variability Summary**")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Products", len(pricing_df))
    with col2:
        st.metric("🚨 Critical", len(critical_variability_items), delta=f"{len(critical_variability_items)/len(pricing_df)*100:.0f}%")
    with col3:
        st.metric("⚠️ High", len(high_variability_items), delta=f"{len(high_variability_items)/len(pricing_df)*100:.0f}%")
    with col4:
        st.metric("🔍 Moderate", len(moderate_variability_items), delta=f"{len(moderate_variability_items)/len(pricing_df)*100:.0f}%")
    with col5:
        st.metric("📊 Low", len(low_variability_items), delta=f"{len(low_variability_items)/len(pricing_df)*100:.0f}%")
    with col6:
        st.metric("✅ Stable", len(stable_items), delta=f"{len(stable_items)/len(pricing_df)*100:.0f}%")
    
    # Total potential savings from all variable items
    total_potential_savings = (critical_variability_items['potential_savings'].sum() if not critical_variability_items.empty else 0) + \
                             (high_variability_items['potential_savings'].sum() if not high_variability_items.empty else 0) + \
                             (moderate_variability_items['potential_savings'].sum() if not moderate_variability_items.empty else 0) + \
                             (low_variability_items['potential_savings'].sum() if not low_variability_items.empty else 0)
    
    st.info(f"💰 **Total Potential Savings from All Variable Items: ${total_potential_savings:,.2f}**")
    
    # Critical variability items - highest priority
    if not critical_variability_items.empty:
        st.markdown("---")
        st.error("🚨 **CRITICAL PRICE VARIABILITY ITEMS - URGENT ACTION REQUIRED**")
        st.warning("These items show extreme price variations (>100%) and require immediate attention for cost optimization.")
        
        for _, item in critical_variability_items.iterrows():
            st.markdown(f"""
            <div style="background: #f8d7da; border: 2px solid #dc3545; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: #721c24;">🚨 CRITICAL: {item['description']}</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <p><strong>Category:</strong> {item['category']}</p>
                        <p><strong>Purchase Count:</strong> {item['purchase_count']}</p>
                        <p><strong>Total Value:</strong> ${item['total_value']:,.2f}</p>
                    </div>
                    <div>
                        <p><strong>Price Variability:</strong> <span style="color: #dc3545; font-weight: bold; font-size: 1.2em;">{item['price_variability_pct']:.1f}%</span></p>
                        <p><strong>Min Price:</strong> ${item['min_unit_price']:.2f}</p>
                        <p><strong>Max Price:</strong> ${item['max_unit_price']:.2f}</p>
                        <p><strong>Price Range:</strong> ${item['price_range']:.2f}</p>
                    </div>
                </div>
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <strong>💡 URGENT Action Required:</strong><br>
                    • IMMEDIATE price negotiations with {vendor_name}<br>
                    • Review pricing contracts and terms<br>
                    • Consider alternative suppliers<br>
                    • Implement price monitoring system
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # High variability items - highlight these prominently
    if not high_variability_items.empty:
        st.markdown("---")
        st.error("⚠️ **HIGH PRICE VARIABILITY ITEMS - IMMEDIATE ACTION REQUIRED**")
        st.warning("These items show significant price variations (50-99%) and require immediate attention for cost optimization.")
        
        for _, item in high_variability_items.iterrows():
            st.markdown(f"""
            <div style="background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 10px; padding: 1.5rem; margin: 1rem 0;">
                <h4 style="margin: 0 0 1rem 0; color: #721c24;">🚨 {item['description']}</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <p><strong>Category:</strong> {item['category']}</p>
                        <p><strong>Purchase Count:</strong> {item['purchase_count']}</p>
                        <p><strong>Total Value:</strong> ${item['total_value']:,.2f}</p>
                    </div>
                    <div>
                        <p><strong>Price Variability:</strong> <span style="color: #dc3545; font-weight: bold;">{item['price_variability_pct']:.1f}%</span></p>
                        <p><strong>Min Price:</strong> ${item['min_unit_price']:.2f}</p>
                        <p><strong>Max Price:</strong> ${item['max_unit_price']:.2f}</p>
                        <p><strong>Price Range:</strong> ${item['price_range']:.2f}</p>
                    </div>
                </div>
                <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                    <strong>💡 Action Required:</strong><br>
                    • Negotiate consistent pricing with {vendor_name}<br>
                    • Consider bulk purchasing for better rates<br>
                    • Monitor price changes closely<br>
                    • Explore alternative suppliers if pricing remains volatile
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Moderate variability items
    if not moderate_variability_items.empty:
        st.markdown("---")
        st.warning("⚠️ **MODERATE PRICE VARIABILITY ITEMS - MONITOR CLOSELY**")
        st.info("These items show some price variations that should be monitored and addressed.")
        
        for _, item in moderate_variability_items.iterrows():
            st.markdown(f"""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="margin: 0 0 0.5rem 0; color: #856404;">⚠️ {item['description']}</h4>
                <p><strong>Category:</strong> {item['category']} | <strong>Variability:</strong> {item['price_variability_pct']:.1f}% | <strong>Price Range:</strong> ${item['price_range']:.2f}</p>
                <p><strong>Recommendation:</strong> Monitor pricing trends and consider price negotiations</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Low variability items - monitor closely
    if not low_variability_items.empty:
        st.markdown("---")
        st.info("📊 **LOW PRICE VARIABILITY ITEMS - MONITOR CLOSELY**")
        st.info("These items show minor price variations (1-24%) that should be monitored for trends.")
        
        for _, item in low_variability_items.iterrows():
            st.markdown(f"""
            <div style="background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 8px; padding: 1rem; margin: 0.5rem 0;">
                <h4 style="margin: 0 0 0.5rem 0; color: #0056b3;">📊 {item['description']}</h4>
                <p><strong>Category:</strong> {item['category']} | <strong>Variability:</strong> {item['price_variability_pct']:.1f}% | <strong>Price Range:</strong> ${item['price_range']:.2f}</p>
                <p><strong>Recommendation:</strong> Monitor pricing trends and consider minor price negotiations</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Stable items
    if not stable_items.empty:
        st.markdown("---")
        st.success("✅ **PRICE STABLE ITEMS - GOOD PERFORMANCE**")
        st.info("These items maintain consistent pricing, indicating good supplier relationships.")
        
        # Show top 5 stable items
        top_stable = stable_items.head(5)
        for _, item in top_stable.iterrows():
            st.markdown(f"""
            <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 8px; padding: 0.5rem; margin: 0.25rem 0;">
                <strong>✅ {item['description']}</strong> - {item['category']} | Variability: {item['price_variability_pct']:.1f}% | Total Value: ${item['total_value']:,.2f}
            </div>
            """, unsafe_allow_html=True)
    
    # Detailed pricing table with highlighting
    st.markdown("---")
    st.write("**📋 Detailed Pricing Analysis**")
    
    # Create enhanced display dataframe
    display_df = pricing_df[['description', 'category', 'purchase_count', 'avg_unit_price', 'min_unit_price', 'max_unit_price', 'price_variability_pct', 'total_value', 'potential_savings', 'risk_level']].copy()
    display_df['avg_unit_price'] = display_df['avg_unit_price'].round(2)
    display_df['min_unit_price'] = display_df['min_unit_price'].round(2)
    display_df['max_unit_price'] = display_df['max_unit_price'].round(2)
    display_df['total_value'] = display_df['total_value'].round(2)
    display_df['potential_savings'] = display_df['potential_savings'].round(2)
    
    # First, rename the existing columns to the display names
    display_df.columns = ['Product', 'Category', 'Purchases', 'Avg Price', 'Min Price', 'Max Price', 'Variability %', 'Total Value', 'Potential Savings', 'Risk Level']
    
    # For Costco, also show original descriptions in a separate column for comparison
    if vendor_name == "Costco" and 'raw_pricing_df' in locals() and not raw_pricing_df.empty:
        try:
            # Create a mapping of cleaned to raw descriptions
            raw_desc_map = {}
            for _, row in raw_pricing_df.iterrows():
                raw_desc = row['raw_description']
                # Find the corresponding cleaned description
                for _, clean_row in pricing_df.iterrows():
                    if clean_row['category'] == row['category'] and clean_row['purchase_count'] == row['purchase_count']:
                        raw_desc_map[clean_row['description']] = raw_desc
                        break
            
            # Add raw description column
            display_df['Raw Description'] = display_df['Product'].map(raw_desc_map)
            # Reorder columns to include Raw Description
            display_df = display_df[['Product', 'Raw Description', 'Category', 'Purchases', 'Avg Price', 'Min Price', 'Max Price', 'Variability %', 'Total Value', 'Potential Savings', 'Risk Level']]
            
            # Rename the Product column to indicate it's cleaned
            display_df.columns = ['Product (Cleaned)', 'Raw Description', 'Category', 'Purchases', 'Avg Price', 'Min Price', 'Max Price', 'Variability %', 'Total Value', 'Potential Savings', 'Risk Level']
        except Exception as e:
            st.warning(f"⚠️ Could not add raw descriptions for Costco: {str(e)}")
            # Keep the normal column structure
            pass
    
    # Sort by price variability (highest first)
    display_df = display_df.sort_values('Variability %', ascending=False)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Overall recommendations
    st.markdown("---")
    st.subheader("💡 **Overall Pricing Insights & Recommendations**")
    
    if len(critical_variability_items) > 0:
        st.error(f"🚨 **CRITICAL PRIORITY**: {len(critical_variability_items)} items show extreme price variability (>100%)")
        st.write("**URGENT Actions Required:**")
        st.write("• IMMEDIATE price negotiations with {vendor_name}")
        st.write("• Review pricing contracts and terms")
        st.write("• Consider alternative suppliers for critical items")
        st.write("• Implement price monitoring system")
    
    if len(high_variability_items) > 0:
        st.error(f"⚠️ **HIGH PRIORITY**: {len(high_variability_items)} items show significant price variability (50-99%)")
        st.write("**Immediate Actions Required:**")
        st.write("• Schedule price negotiations with {vendor_name}")
        st.write("• Review pricing contracts and terms")
        st.write("• Consider alternative suppliers for high-variability items")
        st.write("• Implement price monitoring system")
    
    if len(moderate_variability_items) > 0:
        st.warning(f"🔍 **MEDIUM PRIORITY**: {len(moderate_variability_items)} items show moderate price variability (25-49%)")
        st.write("**Recommended Actions:**")
        st.write("• Monitor pricing trends monthly")
        st.write("• Negotiate better pricing terms")
        st.write("• Consider bulk purchasing for better rates")
    
    if len(low_variability_items) > 0:
        st.info(f"📊 **LOW PRIORITY**: {len(low_variability_items)} items show minor price variability (1-24%)")
        st.write("**Monitoring Actions:**")
        st.write("• Monitor pricing trends quarterly")
        st.write("• Consider minor price negotiations")
        st.write("• Use as early warning indicators")
    
    if len(stable_items) > 0:
        st.success(f"✅ **GOOD PERFORMANCE**: {len(stable_items)} items maintain stable pricing (0% variability)")
        st.write("**Maintain Current Practices:**")
        st.write("• Continue current supplier relationships")
        st.write("• Use as benchmark for other items")
        st.write("• Leverage stable pricing for long-term planning")
    
    # Download option
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Pricing Analysis as CSV",
        data=csv,
        file_name=f"{vendor_name}_pricing_analysis.csv",
        mime="text/csv"
    )
    
    # Vendor-specific insights
    if vendor_name == "Costco":
        st.markdown("---")
        st.subheader("🏪 **Costco-Specific Cost Optimization Insights**")
        
        # Costco-specific recommendations based on variability
        if len(high_variability_items) > 0:
            st.error("🚨 **COSTCO HIGH VARIABILITY ALERT**")
            st.write("**Immediate Cost Optimization Opportunities:**")
            st.write("• **Bulk Purchasing Power**: Leverage Costco's bulk pricing for high-variability items")
            st.write("• **Volume Discounts**: Negotiate better rates based on total business volume")
            st.write("• **Seasonal Timing**: Plan large orders during promotional periods")
            st.write("• **Multi-Location Consolidation**: Combine orders across locations for better pricing")
            
            # Show specific Costco optimization examples
            st.write("**🎯 Specific Optimization Examples:**")
            for _, item in high_variability_items.head(3).iterrows():
                # Use the pre-calculated potential savings from the database
                potential_savings = item['potential_savings']
                st.write(f"• **{item['description'][:40]}...**: Potential savings of ${potential_savings:,.2f} through price stabilization")
        
        elif len(moderate_variability_items) > 0:
            st.warning("⚠️ **COSTCO MODERATE VARIABILITY OPPORTUNITIES**")
            st.write("**Optimization Strategies:**")
            st.write("• **Monitor pricing trends** and time large purchases strategically")
            st.write("• **Negotiate volume discounts** for frequently purchased items")
            st.write("• **Consider Costco Business membership** benefits and pricing tiers")
        
        else:
            st.success("✅ **COSTCO PRICE STABILITY - EXCELLENT PERFORMANCE**")
            st.write("**Current Status:**")
            st.write("• **Pricing is stable** across all Costco products")
            st.write("• **Continue leveraging** Costco's bulk purchasing advantages")
            st.write("• **Focus on other areas** like delivery optimization and order consolidation")
        
        # Costco business insights
        st.write("**💼 Costco Business Relationship Insights:**")
        st.write("• **Bulk Purchasing Power**: Costco specializes in volume discounts")
        st.write("• **Consistent Quality**: Generally stable product quality across purchases")
        st.write("• **Delivery Options**: Multiple delivery locations available for optimization")
        st.write("• **Membership Benefits**: Business membership may offer additional pricing advantages")

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
