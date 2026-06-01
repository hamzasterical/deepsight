import time

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas.response import ErrorResponse, ForgeryResponse

router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024
MIN_FILE_SIZE = 12 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _validate_file(filename: str, contents: bytes) -> None:
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit",
        )
    if len(contents) < MIN_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is too small (min 12KB)",
        )

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )


@router.post(
    "/detect",
    response_model=ForgeryResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def detect_forgery(file: UploadFile = File(...)):
    start = time.perf_counter()

    try:
        contents = await file.read()
        _validate_file(file.filename or "unknown", contents)

        predictor = _get_predictor()
        result = predictor.predict_from_bytes(contents)

        elapsed = (time.perf_counter() - start) * 1000
        result["processing_time_ms"] = round(elapsed, 2)

        return ForgeryResponse(**result)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal processing error",
        )


def _get_predictor():
    from api.main import get_predictor
    return get_predictor()
