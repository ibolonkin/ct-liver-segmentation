from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.config import settings
from src.logger import setup_logging
from src.service.redis_conn import redis_client
from src.service.s3 import s3_client
from src.service.model import modelManager

log = setup_logging('Main')


async def debug_connect(func, name, **kwargs):
    try:
        await func(**kwargs)
        log.debug(
            f"Подключение к {name} успешно",
        )
    except Exception as e:
        log.debug(
            f"Ошибка при подключении к {name}",
            exc_info=e
        )
        exit()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info('Starting app')

    await debug_connect(s3_client.connect, "S3",
                        access_key=settings.S3_ACCESS_KEY,
                        secret_key=settings.S3_SECRET_KEY,
                        endpoint_url=settings.S3_ENDPOINTPUT,
                        region_name=settings.S3_REGION)

    await debug_connect(redis_client.connect, 'REDIS')

    try:
        modelManager.upload_model()
        log.debug('Модель загружена')
    except Exception as e:
        log.debug(
            'Ошибка в загрузке модели',
            exc_info=e
        )
        exit()

    log.info('End starting app')
    yield
    await redis_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title='Liver CT Segmentation',
        lifespan=lifespan,
    )
    return app
