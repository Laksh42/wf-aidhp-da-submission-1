import uvicorn
import argparse
import logging
from uvicorn.config import Config
from uvicorn.main import Server

class CustomServer(Server):
    def run(self, sockets=None):
        self.config.http.h11_max_incomplete_event_size = 32768
        super().run(sockets=sockets)

def run_server(host="0.0.0.0", port=8000, reload=False):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create custom config with increased header limits
    config = Config(
        app="app.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
    
    # Use custom server with increased header size limit
    server = CustomServer(config)
    
    logging.info(f"Starting server at http://{host}:{port}")
    logging.info("Press CTRL+C to stop the server")
    
    server.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Wells Fargo Financial Assistant API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code changes")
    
    args = parser.parse_args()
    run_server(host=args.host, port=args.port, reload=args.reload) 