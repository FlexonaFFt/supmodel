# Makefile
start:
	concurrently "parcel frontend/pages/checkout/checkout.html --port 1234" "python3 backend/core/manage.py runserver 8000" "sleep 5 && python3 backend/api/app.py"
