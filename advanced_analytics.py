"""
Advanced Data Analytics Engine for XelBot
Professional-grade data analysis capabilities
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedAnalytics:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        self.datetime_cols = self._detect_datetime_columns()
        
    def _detect_datetime_columns(self):
        """Detect potential datetime columns"""
        datetime_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'datetime64[ns]':
                datetime_cols.append(col)
            elif self.df[col].dtype == 'object':
                # Try to parse as datetime
                try:
                    pd.to_datetime(self.df[col].head(10))
                    datetime_cols.append(col)
                except:
                    pass
        return datetime_cols
    
    def data_profiling(self):
        """Comprehensive data profiling"""
        profile = {
            'overview': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                'missing_values': self.df.isnull().sum().sum(),
                'duplicate_rows': self.df.duplicated().sum()
            },
            'column_types': {
                'numeric': len(self.numeric_cols),
                'categorical': len(self.categorical_cols),
                'datetime': len(self.datetime_cols)
            },
            'data_quality': self._assess_data_quality(),
            'statistical_summary': self._statistical_summary()
        }
        return profile
    
    def _assess_data_quality(self):
        """Assess data quality issues"""
        quality_issues = []
        
        # Missing values
        missing_pct = (self.df.isnull().sum() / len(self.df) * 100)
        high_missing = missing_pct[missing_pct > 20].to_dict()
        if high_missing:
            quality_issues.append(f"High missing values: {high_missing}")
        
        # Duplicate rows
        if self.df.duplicated().sum() > 0:
            quality_issues.append(f"Duplicate rows: {self.df.duplicated().sum()}")
        
        # Outliers in numeric columns
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(self.df) * 0.05:  # More than 5% outliers
                quality_issues.append(f"Potential outliers in {col}: {outliers}")
        
        return quality_issues if quality_issues else ["Data quality looks good!"]
    
    def _statistical_summary(self):
        """Enhanced statistical summary"""
        summary = {}
        
        # Numeric columns
        if self.numeric_cols:
            numeric_summary = self.df[self.numeric_cols].describe()
            summary['numeric'] = numeric_summary.to_dict()
        
        # Categorical columns
        if self.categorical_cols:
            cat_summary = {}
            for col in self.categorical_cols:
                cat_summary[col] = {
                    'unique_values': self.df[col].nunique(),
                    'most_frequent': self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None,
                    'frequency': self.df[col].value_counts().head(3).to_dict()
                }
            summary['categorical'] = cat_summary
        
        return summary
    
    def business_insights(self):
        """Generate business-focused insights"""
        insights = []
        
        # Revenue analysis (if revenue-related columns exist)
        revenue_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                       for keyword in ['revenue', 'sales', 'price', 'amount', 'value'])]
        
        if revenue_cols:
            for col in revenue_cols:
                if col in self.numeric_cols:
                    total = self.df[col].sum()
                    avg = self.df[col].mean()
                    insights.append(f"💰 Total {col}: ${total:,.2f}")
                    insights.append(f"📊 Average {col}: ${avg:,.2f}")
        
        # Growth trends (if date columns exist)
        if self.datetime_cols and revenue_cols:
            insights.extend(self._analyze_trends())
        
        # Customer analysis
        customer_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                        for keyword in ['customer', 'user', 'client'])]
        
        if customer_cols:
            for col in customer_cols:
                unique_customers = self.df[col].nunique()
                insights.append(f"👥 Unique customers: {unique_customers:,}")
        
        # Performance metrics
        insights.extend(self._performance_metrics())
        
        return insights
    
    def _analyze_trends(self):
        """Analyze trends over time"""
        trends = []
        
        # This is a simplified trend analysis
        # In a real implementation, you'd do more sophisticated time series analysis
        if len(self.datetime_cols) > 0 and len(self.numeric_cols) > 0:
            trends.append("📈 Trend analysis available for time-series data")
            trends.append("🔍 Consider seasonal patterns and growth rates")
        
        return trends
    
    def _performance_metrics(self):
        """Calculate key performance metrics"""
        metrics = []
        
        # Conversion rates (if applicable)
        if any('conversion' in col.lower() for col in self.df.columns):
            metrics.append("🎯 Conversion rate analysis available")
        
        # Engagement metrics (for social media data)
        engagement_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                          for keyword in ['like', 'share', 'comment', 'view', 'engagement'])]
        
        if engagement_cols:
            metrics.append("📱 Social media engagement metrics detected")
        
        return metrics
    
    def predictive_analysis(self):
        """Simple predictive analysis"""
        predictions = {}
        
        if len(self.numeric_cols) >= 2:
            # Simple correlation analysis
            corr_matrix = self.df[self.numeric_cols].corr()
            high_corr = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_corr.append({
                            'variables': f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}",
                            'correlation': round(corr_val, 3)
                        })
            
            predictions['correlations'] = high_corr
        
        return predictions
    
    def generate_recommendations(self):
        """Generate actionable business recommendations"""
        recommendations = []
        
        # Data quality recommendations
        if self.df.isnull().sum().sum() > 0:
            recommendations.append("🔧 Clean missing data to improve analysis accuracy")
        
        # Business growth recommendations
        if any('revenue' in col.lower() for col in self.df.columns):
            recommendations.append("💡 Focus on high-revenue segments for growth")
        
        # Customer retention recommendations
        if any('customer' in col.lower() for col in self.df.columns):
            recommendations.append("🎯 Implement customer retention strategies")
        
        # Performance optimization
        recommendations.append("📊 Monitor key metrics regularly for better decision making")
        recommendations.append("🚀 Use data-driven insights to optimize business processes")
        
        return recommendations

class BusinessIntelligence:
    """Business Intelligence specific analysis"""
    
    def __init__(self, dataframe):
        self.df = dataframe
        self.analytics = AdvancedAnalytics(dataframe)
    
    def executive_summary(self):
        """Generate executive summary"""
        profile = self.analytics.data_profiling()
        insights = self.analytics.business_insights()
        recommendations = self.analytics.generate_recommendations()
        
        summary = {
            'data_overview': profile['overview'],
            'key_insights': insights[:5],  # Top 5 insights
            'recommendations': recommendations[:3],  # Top 3 recommendations
            'data_quality_score': self._calculate_quality_score(profile)
        }
        
        return summary
    
    def _calculate_quality_score(self, profile):
        """Calculate data quality score (0-100)"""
        score = 100
        
        # Deduct for missing values
        missing_pct = (profile['overview']['missing_values'] / 
                      (profile['overview']['rows'] * profile['overview']['columns'])) * 100
        score -= missing_pct * 2
        
        # Deduct for duplicates
        duplicate_pct = (profile['overview']['duplicate_rows'] / profile['overview']['rows']) * 100
        score -= duplicate_pct * 3
        
        return max(0, min(100, round(score, 1)))
