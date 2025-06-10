"""
Production deployment script for Trinity Memory System
"""

import os
import subprocess
import sys
from pathlib import Path

def check_environment():
    """Check production environment requirements"""
    print("🔍 Checking production environment...")
    
    required_env_vars = [
        'PINECONE_API_KEY',
        'OPENAI_API_KEY',
        'NAS_HOST',
        'NAS_USERNAME',
        'NAS_PASSWORD',
        'GOOGLE_DRIVE_FOLDER_ID'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("All required environment variables set")
    return True

def setup_systemd_services():
    """Setup systemd services for production"""
    services = {
        'trinity-api': {
            'description': 'Trinity API Server',
            'exec_start': '/opt/trinity/venv/bin/python /opt/trinity/scripts/run_enhanced_system.py'
        },
        'trinity-celery-worker': {
            'description': 'Trinity Celery Worker',
            'exec_start': '/opt/trinity/venv/bin/celery -A scripts.celery_worker worker --loglevel=info'
        },
        'trinity-celery-beat': {
            'description': 'Trinity Celery Beat Scheduler',
            'exec_start': '/opt/trinity/venv/bin/celery -A scripts.celery_worker beat --loglevel=info'
        }
    }
    
    for service_name, config in services.items():
        service_content = f"""[Unit]
Description={config['description']}
After=network.target redis.service

[Service]
Type=simple
User=trinity
Group=trinity
WorkingDirectory=/opt/trinity
Environment=PATH=/opt/trinity/venv/bin
ExecStart={config['exec_start']}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
        
        service_file = f"/etc/systemd/system/{service_name}.service"
        print(f"Creating systemd service: {service_file}")
        
        # In production, you would write this file with proper permissions
        print(f"Service content for {service_name}:")
        print(service_content)
        print("-" * 50)

def setup_nginx():
    """Setup Nginx reverse proxy configuration"""
    nginx_config = """
server {
    listen 80;
    server_name trinity-memory.yourdomain.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name trinity-memory.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/trinity-memory.crt;
    ssl_certificate_key /etc/ssl/private/trinity-memory.key;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Static files
    location /static/ {
        alias /opt/trinity/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
"""
    
    print("Nginx configuration:")
    print(nginx_config)

def main():
    """Main deployment function"""
    print("Trinity Memory System - Production Deployment")
    
    # Check environment
    if not check_environment():
        print("Environment check failed")
        return False
    
    # Setup systemd services
    setup_systemd_services()
    
    # Setup Nginx
    setup_nginx()
    
    print("\nProduction deployment setup complete!")
    print("\nNext steps:")
    print("1. Copy systemd service files to /etc/systemd/system/")
    print("2. Copy Nginx configuration to /etc/nginx/sites-available/")
    print("3. Enable and start services:")
    print("   sudo systemctl enable trinity-api trinity-celery-worker trinity-celery-beat")
    print("   sudo systemctl start trinity-api trinity-celery-worker trinity-celery-beat")
    print("4. Setup SSL certificates")
    print("5. Configure firewall rules")
    print("6. Setup log rotation")
    
    return True

if __name__ == "__main__":
    main()