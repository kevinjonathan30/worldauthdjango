# WorldAuthDjango

## Setup
Create virtual environment
```bash
python -m venv venv
```
And activate the environment
```bash
source venv/bin/activate
```
Then install the required libraries
```bash
pip install -r requirements.txt
```

## How to Run


Run the following commands to start the Django development server and Ngrok:

```bash
python manage.py runserver
```

If you want to test the local server outside LAN, open a separate window, run this:

```bash
./ngrok http --domain=REMOVED-NGROK 8000
```

All API endpoints are now served via Django Ninja at `/api/`.

Example endpoints:

- POST /api/register
- POST /api/authenticate
- POST /api/authenticate_two
- GET /api/get_point_cloud
- DELETE /api/delete_point_cloud

Done!
