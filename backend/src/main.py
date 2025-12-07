from starlette.middleware.cors import CORSMiddleware

from src.create_app import create_app
from src.api.v1 import router
from src.middlewares.authmiddleware import AuthMiddleware
from src.middlewares.logmiddleware import LogExecutionTimeMiddleware

app = create_app()
origins = ['*']
app.include_router(router, tags=['API model'])

app.add_middleware(AuthMiddleware)
app.add_middleware(LogExecutionTimeMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
