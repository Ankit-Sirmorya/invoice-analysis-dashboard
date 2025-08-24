# 🚀 Dashboard Deployment Guide

## 🌐 **Option 1: Streamlit Cloud (Recommended - Free)**

### **Step 1: Prepare Your Repository**
1. Make sure your code is in a GitHub repository
2. Ensure `requirements.txt` is in the root directory
3. Ensure `.streamlit/config.toml` is in the root directory

### **Step 2: Deploy to Streamlit Cloud**
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set the main file path: `vendor_centric_dashboard_fixed.py`
6. Click "Deploy"

**Your app will be live at**: `https://your-app-name.streamlit.app`

---

## 🐳 **Option 2: Docker Deployment**

### **Step 1: Create Dockerfile**
```bash
docker build -t invoice-dashboard .
```

### **Step 2: Run Container**
```bash
docker run -p 8501:8501 invoice-dashboard
```

### **Step 3: Access Dashboard**
Open `http://localhost:8501` in your browser

---

## ☁️ **Option 3: Heroku Deployment**

### **Step 1: Install Heroku CLI**
```bash
# macOS
brew install heroku/brew/heroku

# Or download from: https://devcenter.heroku.com/articles/heroku-cli
```

### **Step 2: Create Heroku App**
```bash
heroku create your-app-name
```

### **Step 3: Deploy**
```bash
git add .
git commit -m "Deploy dashboard"
git push heroku main
```

---

## 🌍 **Option 4: Local Network Deployment**

### **Step 1: Find Your IP Address**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

### **Step 2: Run Dashboard**
```bash
streamlit run vendor_centric_dashboard_fixed.py --server.address 0.0.0.0 --server.port 8501
```

### **Step 3: Access from Other Devices**
Other devices on your network can access: `http://YOUR_IP:8501`

---

## 📋 **Pre-Deployment Checklist**

- [ ] Database file (`invoice_line_items.db`) is included
- [ ] All required Python packages are in `requirements.txt`
- [ ] `.streamlit/config.toml` is configured
- [ ] No hardcoded file paths
- [ ] Error handling is in place
- [ ] Dashboard works locally

---

## 🔧 **Troubleshooting**

### **Common Issues:**
1. **Database not found**: Ensure `invoice_line_items.db` is in the repository
2. **Missing packages**: Check `requirements.txt` includes all dependencies
3. **Port conflicts**: Change port in `.streamlit/config.toml`
4. **File permissions**: Ensure proper read permissions on database

### **Streamlit Cloud Specific:**
- App size limit: 1GB
- Database files are read-only
- No persistent storage between sessions

---

## 📱 **Production Considerations**

### **For Business Use:**
- Use Heroku or AWS for production
- Set up proper authentication
- Implement rate limiting
- Regular database backups
- Monitoring and logging

### **For Demo/Internal Use:**
- Streamlit Cloud is perfect
- Local network deployment for team access
- Docker for consistent environments
