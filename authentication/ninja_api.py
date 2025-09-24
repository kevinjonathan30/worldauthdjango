from ninja import NinjaAPI
from authentication.api import router as auth_router

api = NinjaAPI(
    title="WorldAuth API",
    version="1.0.0",
    description="WorldAuth API List Documentation",
    docs_url="/docs",           # Swagger UI path (set to None to disable)
    openapi_url="/openapi.json" # Raw OpenAPI JSON
)
api.add_router("/", auth_router)
