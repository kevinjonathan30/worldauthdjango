#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
# import ngrok

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'worldauthdjango.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    
    # if len(sys.argv) > 1 and sys.argv[1] == "runserver":
    #     # Establish connectivity
    #     # Authenticate with your Ngrok authtoken
    #     ngrok.set_auth_token("2ilmVPrD6tlgTaIWTnWf9PXrzLH_2hAkSRCBSeyCbHgR8hE9T")

    #     # Start a HTTP tunnel on port 8000
    #     # Start a Ngrok tunnel with a specific subdomain
    #     tunnel = ngrok.connect(port=8000, proto="http", bind_tls=True, subdomain="REMOVED-NGROK")

    #     # Print the publicly accessible URL
    #     # print(f"Public URL: {tunnel.subdomain}")

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
