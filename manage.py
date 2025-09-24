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

    # If the user runs the development server, print the local docs URL for Django Ninja.
    try:
        if len(sys.argv) >= 2 and sys.argv[1].startswith('runserver'):
            # The Django autoreloader runs the code twice: a parent process and
            # a child where the server actually runs. To avoid printing twice,
            # only print in the child process (RUN_MAIN='true'). Also print if
            # the user explicitly disables the reloader via --noreload.
            reload_disabled = any(arg.startswith('--noreload') for arg in sys.argv)
            run_main = os.environ.get('RUN_MAIN') == 'true'

            if run_main or reload_disabled:
                host = '127.0.0.1'
                port = '8000'
                # If the user supplied a different addrport, try to extract it
                if len(sys.argv) >= 3 and sys.argv[2]:
                    addrport = sys.argv[2]
                    if ':' in addrport:
                        host, port = addrport.split(':', 1)
                    else:
                        port = addrport
                print(f"\nWorldAuth API docs available at: http://{host}:{port}/api/docs#/\n")
    except Exception:
        # Never let this logging interfere with management command execution
        pass

    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
