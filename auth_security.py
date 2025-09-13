"""
Authentication & Security Module
===============================
Implements user authentication, rate limiting, and security monitoring.
"""
import bcrypt
import secrets
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from flask_login import UserMixin, login_required
from functools import wraps
from flask import request, jsonify, session, redirect, url_for
from secure_config import Config

# Security logging
security_logger = logging.getLogger('security')
security_logger.setLevel(getattr(logging, Config.LOG_LEVEL))

class User(UserMixin):
    """Simple user class for Flask-Login"""
    def __init__(self, username):
        self.id = username
        self.username = username
    
    @staticmethod
    def validate_credentials(username, password):
        """Validate user credentials"""
        if username == Config.ADMIN_USERNAME and password == Config.ADMIN_PASSWORD:
            return User(username)
        return None
    
    @staticmethod
    def get(user_id):
        """Get user by ID"""
        if user_id == Config.ADMIN_USERNAME:
            return User(user_id)
        return None

class SecurityMonitor:
    """Monitor and track security events"""
    def __init__(self):
        self.failed_attempts = defaultdict(list)
        self.blocked_ips = set()
        self.suspicious_activity = []
    
    def record_failed_login(self, ip_address, username):
        """Record failed login attempt"""
        now = datetime.now()
        self.failed_attempts[ip_address].append({
            'timestamp': now,
            'username': username
        })
        
        # Clean old attempts (older than 1 hour)
        self.failed_attempts[ip_address] = [
            attempt for attempt in self.failed_attempts[ip_address]
            if now - attempt['timestamp'] < timedelta(hours=1)
        ]
        
        # Block IP after 5 failed attempts in 1 hour
        if len(self.failed_attempts[ip_address]) >= 5:
            self.blocked_ips.add(ip_address)
            security_logger.warning(f"IP {ip_address} blocked after multiple failed login attempts")
    
    def is_ip_blocked(self, ip_address):
        """Check if IP is blocked"""
        return ip_address in self.blocked_ips
    
    def record_suspicious_activity(self, ip_address, activity_type, details):
        """Record suspicious activity"""
        event = {
            'timestamp': datetime.now(),
            'ip_address': ip_address,
            'activity_type': activity_type,
            'details': details
        }
        self.suspicious_activity.append(event)
        security_logger.warning(f"Suspicious activity: {activity_type} from {ip_address} - {details}")

# Global security monitor instance
security_monitor = SecurityMonitor()

def require_api_key(f):
    """Decorator to require API key for webhook endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.form.get('api_key')
        
        if not api_key:
            security_monitor.record_suspicious_activity(
                request.remote_addr, 
                'missing_api_key',
                f"Attempted access to {request.endpoint} without API key"
            )
            return jsonify({'error': 'API key required'}), 401
        
        if api_key != Config.WEBHOOK_API_KEY:
            security_monitor.record_suspicious_activity(
                request.remote_addr,
                'invalid_api_key', 
                f"Invalid API key attempted for {request.endpoint}"
            )
            return jsonify({'error': 'Invalid API key'}), 401
            
        return f(*args, **kwargs)
    return decorated_function

def check_ip_blocked(f):
    """Decorator to check if IP is blocked"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if security_monitor.is_ip_blocked(request.remote_addr):
            return jsonify({'error': 'IP blocked due to suspicious activity'}), 403
        return f(*args, **kwargs)
    return decorated_function

def require_auth_or_api_key(f):
    """Decorator that requires either login OR valid API key"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check for API key first
        api_key = request.headers.get('X-API-Key')
        if api_key == Config.WEBHOOK_API_KEY:
            return f(*args, **kwargs)
        
        # Otherwise require login
        from flask_login import current_user
        if not current_user.is_authenticated:
            if request.is_json:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

def log_trade_activity(action, details):
    """Log trading activity for security audit"""
    if Config.ENABLE_TRADE_LOGGING:
        security_logger.info(f"TRADE: {action} - {details} - IP: {request.remote_addr}")

def validate_trade_request(data):
    """Validate incoming trade requests for security"""
    required_fields = ['ticker', 'action']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Validate ticker format
    ticker = data.get('ticker', '').upper()
    if not ticker.isalpha() or len(ticker) > 10:
        return False, "Invalid ticker format"
    
    # Validate action
    valid_actions = ['buy', 'sell', 'long', 'short', 'exit']
    if data.get('action', '').lower() not in valid_actions:
        return False, "Invalid trading action"
    
    return True, "Valid request"
