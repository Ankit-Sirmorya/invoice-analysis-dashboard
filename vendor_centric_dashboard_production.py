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
            
            query = """
            SELECT 
                COUNT(DISTINCT i.filename) as invoice_count,
                SUM(i.total_amount) as total_spending,
                AVG(i.total_amount) as avg_invoice_value,
                MIN(i.date) as first_order,
                MAX(i.date) as last_order,
                COUNT(l.id) as line_item_count
            FROM invoices i
            LEFT JOIN line_items l ON i.filename = l.filename
            WHERE i.vendor = ?
            GROUP BY i.vendor
            """
            df = pd.read_sql_query(query, self.conn, params=(vendor_name,))
            return df
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

def show_combined_analysis(dashboard):
    """Show combined analysis across all vendors."""
    st.subheader("🌐 Combined Vendor Analysis")
    st.markdown("**Cross-vendor insights and optimization opportunities**")
    
    try:
        conn = dashboard.conn
        if not conn:
            st.error("Database connection not available")
            return
            
        cursor = conn.cursor()
        
        # 1. Vendor Summary
        st.subheader("📊 Vendor Summary")
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
        st.subheader("🚨 Risk Analysis")
        if not summary_df.empty:
            top_vendor = summary_df.iloc[0]
            top_vendor_pct = (top_vendor['total_spending'] / total_spending) * 100
            if top_vendor_pct > 40:
                st.warning(f"⚠️ **High Concentration Risk**: {top_vendor['vendor']} represents {top_vendor_pct:.1f}% of total spending")
                st.info("💡 **Recommendation**: Consider diversifying suppliers to reduce dependency")
            else:
                st.success("✅ **Good Vendor Diversity**: No single vendor represents more than 40% of spending")
        
        # 3. Optimization Opportunities
        st.subheader("💰 Cost Optimization Opportunities")
        high_spending = summary_df[summary_df['total_spending'] > 1000]
        if not high_spending.empty:
            st.write("**High-Spending Vendors (Negotiation Priority):**")
            for _, row in high_spending.iterrows():
                st.write(f"   • **{row['vendor']}**: ${row['total_spending']:,.2f} across {row['invoice_count']} invoices")
        
        # 4. Strategic Recommendations
        st.subheader("🎯 Strategic Recommendations")
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
        st.subheader("💡 Estimated Savings Potential")
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
        
        # Download combined report
        st.subheader("📥 Download Combined Analysis")
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
            label="📥 Download Combined Analysis CSV",
            data=csv,
            file_name="combined_vendor_analysis.csv",
            mime="text/csv"
        )
            
    except Exception as e:
        st.error(f"Error generating combined analysis: {e}")
        st.info("Please ensure all vendor data is properly loaded.")

def main():
    st.markdown('<h1 class="main-header">🏢 Vendor-Centric Invoice Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### **🌐 Combined Analysis is always visible above | Select any vendor below to see vendor-specific analysis in organized tabs**")
    
    # Deployment information
    st.markdown("""
    <div class="deployment-info">
        <strong>🚀 Deployment Status:</strong> Production Ready<br>
        <strong>📊 Database:</strong> SQLite (invoice_line_items.db)<br>
        <strong>🌐 Access:</strong> Available on all devices in network
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
        
        # Vendor selection
        st.sidebar.title("🎯 Vendor Selection")
        selected_vendor = st.sidebar.selectbox(
            "Choose a vendor to analyze:",
            vendors,
            index=0
        )
        
        if selected_vendor:
            show_vendor_analysis(dashboard, selected_vendor)
    
    finally:
        dashboard.disconnect_db()

def show_vendor_analysis(dashboard, vendor_name):
    """Show comprehensive analysis for the selected vendor."""
    
    try:
        # Combined Analysis Section (Always visible)
        st.markdown("---")
        with st.expander("🌐 **Combined Vendor Analysis** - Click to expand/collapse", expanded=True):
            st.markdown('<h3 style="text-align: center; color: #667eea;">🌐 Combined Vendor Analysis</h3>', unsafe_allow_html=True)
            st.markdown("**Cross-vendor insights and optimization opportunities**")
            
            # Show combined analysis
            show_combined_analysis(dashboard)
        
        st.markdown("---")
        
        # Vendor header
        st.markdown(f'<h2 class="vendor-header">🏢 {vendor_name}</h2>', unsafe_allow_html=True)
        
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
        
        # Analysis tabs (now only 6 tabs, Combined Analysis moved above)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Overview", 
            "🚨 Duplicates", 
            "📅 Trends", 
            "📦 Categories", 
            "💰 Pricing", 
            "📋 Details"
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

if __name__ == "__main__":
    main()
