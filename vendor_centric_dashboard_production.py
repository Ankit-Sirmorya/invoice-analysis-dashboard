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
        
        # 2. Risk Analysis (High Concentration Risk)
        st.subheader("🚨 Supply Chain Risk Assessment")
        if not summary_df.empty:
            top_vendor = summary_df.iloc[0]
            top_vendor_pct = (top_vendor['total_spending'] / total_spending) * 100
            if top_vendor_pct > 40:
                st.warning(f"⚠️ **High Concentration Risk**: {top_vendor['vendor']} represents {top_vendor_pct:.1f}% of total spending")
                st.info("💡 **Recommendation**: Consider diversifying suppliers to reduce dependency")
            else:
                st.success("✅ **Good Vendor Diversity**: No single vendor represents more than 40% of spending")
        
        # 3. Optimization Opportunities
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
        
        # 6. Comprehensive Time-Based Analysis
        st.subheader("🕒 Time-Based Spending Analysis & Trends")
        st.markdown("**Restaurant spending patterns, seasonal trends, and optimization opportunities**")
        
        # Time-based analysis tabs
        time_tab1, time_tab2, time_tab3, time_tab4 = st.tabs([
            "📅 Monthly Trends", 
            "🌤️ Seasonal Patterns", 
            "📊 Weekly Patterns", 
            "💡 Time-Based Insights"
        ])
        
        with time_tab1:
            st.markdown("### 📅 Monthly Spending Trends Across All Vendors")
            monthly_data = dashboard.get_all_vendors_monthly_trends()
            if not monthly_data.empty:
                # Monthly spending line chart
                fig_monthly = px.line(monthly_data, x='month', y='total_spending', color='vendor',
                                    title="Monthly Spending Trends by Vendor",
                                    labels={'total_spending': 'Total Spending ($)', 'month': 'Month'})
                fig_monthly.update_layout(height=500)
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Monthly spending heatmap
                monthly_pivot = monthly_data.pivot(index='month_name', columns='vendor', values='total_spending').fillna(0)
                fig_heatmap = px.imshow(monthly_pivot, 
                                      title="Monthly Spending Heatmap by Vendor",
                                      labels=dict(x="Vendor", y="Month", color="Spending ($)"))
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                st.write("**Monthly Spending Summary:**")
                st.dataframe(monthly_data.groupby('month_name')['total_spending'].sum().reset_index(), use_container_width=True)
            else:
                st.info("No monthly trend data available")
        
        with time_tab2:
            st.markdown("### 🌤️ Seasonal Spending Patterns")
            seasonal_data = dashboard.get_all_vendors_seasonal_analysis()
            if not seasonal_data.empty:
                # Seasonal spending by vendor
                fig_seasonal = px.bar(seasonal_data, x='season', y='total_spending', color='vendor',
                                    title="Seasonal Spending Patterns by Vendor",
                                    labels={'total_spending': 'Total Spending ($)', 'season': 'Season'})
                fig_seasonal.update_layout(height=500)
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
                # Seasonal summary
                seasonal_summary = seasonal_data.groupby('season').agg({
                    'total_spending': 'sum',
                    'invoice_count': 'sum'
                }).reset_index()
                seasonal_summary['avg_spending'] = seasonal_summary['total_spending'] / seasonal_summary['invoice_count']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Seasonal Spending Summary:**")
                    st.dataframe(seasonal_summary, use_container_width=True)
                with col2:
                    st.write("**Seasonal Insights:**")
                    peak_season = seasonal_summary.loc[seasonal_summary['total_spending'].idxmax()]
                    st.success(f"**Peak Season**: {peak_season['season']} - ${peak_season['total_spending']:,.0f}")
                    st.info(f"**Average Invoice**: ${seasonal_summary['avg_spending'].mean():,.0f}")
            else:
                st.info("No seasonal data available")
        
        with time_tab3:
            st.markdown("### 📊 Weekly Spending Patterns")
            weekly_data = dashboard.get_all_vendors_weekly_patterns()
            if not weekly_data.empty:
                # Weekly spending trends
                fig_weekly = px.line(weekly_data, x='week_label', y='total_spending', color='vendor',
                                   title="Weekly Spending Patterns by Vendor",
                                   labels={'total_spending': 'Total Spending ($)', 'week_label': 'Week'})
                fig_weekly.update_layout(height=500)
                st.plotly_chart(fig_weekly, use_container_width=True)
                
                # Weekly summary statistics
                weekly_summary = weekly_data.groupby('vendor').agg({
                    'total_spending': ['mean', 'std', 'min', 'max']
                }).round(2)
                weekly_summary.columns = ['Avg Weekly Spending', 'Std Dev', 'Min Weekly', 'Max Weekly']
                weekly_summary = weekly_summary.reset_index()
                
                st.write("**Weekly Spending Statistics by Vendor:**")
                st.dataframe(weekly_summary, use_container_width=True)
            else:
                st.info("No weekly pattern data available")
        
        with time_tab4:
            st.markdown("### 💡 Time-Based Optimization Insights")
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
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: 'Georgia', serif;
    }
    .main-header .restaurant-name {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0.5rem 0;
        opacity: 0.95;
        font-family: 'Georgia', serif;
    }
    .main-header .dashboard-subtitle {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.9;
        font-weight: 300;
    }
    .main-header .business-info {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1.5rem;
        display: inline-block;
        backdrop-filter: blur(10px);
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
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .vendor-header h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Georgia', serif;
    }
    .business-summary {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    .business-summary h3 {
        color: #2c3e50;
        margin: 0 0 1rem 0;
        font-size: 1.5rem;
        font-weight: 600;
        font-family: 'Georgia', serif;
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
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
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
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem;">
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
        
        # Enhanced vendor header for restaurant operations
        st.markdown(f"""
        <div class="vendor-header">
            <h2>🏢 {vendor_name}</h2>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">Supply Chain Partner Analysis & Cost Optimization</p>
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
