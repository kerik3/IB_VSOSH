"""
VVM Platform - Production Runner
"""

import os
from app import app

if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV', 'production') == 'development'
    
    print(f"""
    ╔═══════════════════════════════════════════════════╗
    ║   VVM Online School Platform                      ║
    ║   Secure Video Learning Management System         ║
    ╚═══════════════════════════════════════════════════╝
    
    Server running on: http://{host}:{port}
    Environment: {'Development' if debug else 'Production'}
    Database: {app.config['SQLALCHEMY_DATABASE_URI']}
    
    Press CTRL+C to stop the server
    """)
    
    app.run(host=host, port=port, debug=debug)
