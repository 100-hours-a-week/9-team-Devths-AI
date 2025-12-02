from fastapi import status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

class APIError(Exception):
    def __init__(self, message: str, status_code: int, data=None):
        self.message = message
        self.status_code = status_code
        self.data = data

def bad_request(msg: str, data=None):   return APIError(msg, status.HTTP_400_BAD_REQUEST, data)
def not_found(msg: str):                return APIError(msg, status.HTTP_404_NOT_FOUND, data=None)
def unprocessable(msg: str, data=None): return APIError(msg, status.HTTP_422_UNPROCESSABLE_ENTITY, data)
def internal_server_error(msg: str="internal_server_error"): return APIError(msg, status.HTTP_500_INTERNAL_SERVER_ERROR, data=None)

async def api_error_handler(_: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.message, "data": exc.data}
    )

async def validation_error_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": "validation_error", "data": {"details": str(exc)}}
    )

async def global_exception_handler(_: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "internal_server_error", "data": {"details": str(exc)}}
    )
