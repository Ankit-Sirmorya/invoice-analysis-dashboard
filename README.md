# 🏢 **Vendor-Centric Invoice Analysis Dashboard**

A comprehensive Streamlit-based dashboard for analyzing vendor spending patterns, identifying cost optimization opportunities, and providing strategic insights across multiple suppliers.

## 🌟 **Features**

### **🌐 Combined Analysis Section**
- **Total Business Spending**: $47,618.61 (accurately calculated)
- **Vendor Rankings & Percentages**
- **Risk Analysis** (vendor concentration assessment)
- **Optimization Opportunities**
- **Strategic Recommendations**
- **Estimated Annual Savings**: $13,400 - $33,600

### **🏢 Vendor-Specific Analysis**
- **7 Major Vendors** with complete analysis
- **6 Analysis Tabs** per vendor:
  1. **Overview** - Key metrics and insights
  2. **Duplicates** - Duplicate detection and analysis
  3. **Trends** - Monthly, daily, weekly, seasonal patterns
  4. **Categories** - Spending breakdown by category
  5. **Pricing** - Price variability and optimization
  6. **Details** - Individual invoice information

### **📊 Current Vendor Breakdown**
1. **Costco**: $18,481.86 (38.8%) - Major supplier
2. **Villa Jerada**: $12,912.06 (27.1%) - Strategic partner
3. **Pacific Food Importers**: $6,946.73 (14.6%) - Specialized supplier
4. **ChemMark**: $4,073.46 (8.6%) - Chemical supplies
5. **Over the Moon Coffee**: $2,520.00 (5.3%) - Coffee supplier
6. **Tomlinson Linen**: $1,634.74 (3.4%) - Linen services
7. **Franks Quality Produce**: $1,049.76 (2.2%) - Produce supplier

## 🚀 **Quick Start**

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run vendor_centric_dashboard_production.py
```

### **Local Network Deployment**
```bash
# Deploy for team access
python deploy_local_network.py
```

### **Docker Deployment**
```bash
# Build and run container
docker build -t invoice-dashboard .
docker run -p 8501:8501 invoice-dashboard
```

## 🌐 **Deployment Options**

### **1. 🆓 Streamlit Cloud (Recommended)**
- Free hosting
- Automatic deployments from GitHub
- Perfect for demos and stakeholder access

### **2. 🐳 Docker**
- Consistent environments
- Easy deployment to any cloud platform
- Professional hosting solution

### **3. 🌍 Local Network**
- Team access on your local network
- No external hosting required
- Perfect for internal presentations

### **4. ☁️ Heroku/AWS**
- Production-grade hosting
- Custom domains
- Advanced monitoring and scaling

## 📋 **Requirements**

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 1.5.0+
- Plotly 5.15.0+
- SQLite3

## 🔧 **Configuration**

The dashboard automatically detects and connects to the SQLite database (`invoice_line_items.db`) with multiple path fallbacks for different deployment scenarios.

## 📁 **Project Structure**

```
invoices_review/
├── vendor_centric_dashboard_production.py  # Main dashboard
├── requirements.txt                        # Python dependencies
├── .streamlit/config.toml                 # Streamlit configuration
├── invoice_line_items.db                  # SQLite database
├── Dockerfile                             # Docker configuration
├── deploy_local_network.py               # Local network deployment
├── DEPLOYMENT_GUIDE.md                   # Comprehensive deployment guide
└── README.md                             # This file
```

## 🎯 **Use Cases**

- **Business Intelligence**: Vendor spending analysis and insights
- **Cost Optimization**: Identify savings opportunities and negotiation leverage
- **Strategic Planning**: Vendor consolidation and relationship management
- **Stakeholder Presentations**: Professional dashboards for executives
- **Team Collaboration**: Shared access to vendor analytics

## 🚀 **Deploy to Streamlit Cloud**

1. **Fork/Clone** this repository to your GitHub account
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with GitHub
4. **Click "New app"**
5. **Select** your repository
6. **Set main file path**: `vendor_centric_dashboard_production.py`
7. **Click "Deploy"**

Your app will be live at: `https://your-app-name.streamlit.app`

## 📞 **Support**

This dashboard is production-ready with:
- ✅ All data issues resolved
- ✅ Comprehensive error handling
- ✅ Multiple deployment options
- ✅ Professional UI/UX
- ✅ Accurate financial calculations

## 📄 **License**

This project is for business use and analysis purposes.

---

**Ready to optimize your vendor spending!** 🎉
