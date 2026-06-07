import pytest


class TestResponseSchema:
    def test_forgery_response_importable(self):
        from api.schemas.response import ForgeryResponse
        assert ForgeryResponse is not None

    def test_error_response_importable(self):
        from api.schemas.response import ErrorResponse
        assert ErrorResponse is not None

    def test_forgery_response_fields(self):
        from api.schemas.response import ForgeryResponse
        fields = ForgeryResponse.model_fields
        assert "verdict" in fields
        assert "confidence" in fields
        assert "processing_time_ms" in fields


class TestRouter:
    def test_router_importable(self):
        from api.routes.detect import router
        assert router is not None

    def test_detect_route_exists(self):
        from api.routes.detect import router
        routes = [r.path for r in router.routes]
        assert "/detect" in routes


class TestApp:
    def test_app_importable(self):
        from api.main import app
        assert app is not None

    def test_health_route_exists(self):
        from api.main import app
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_cors_middleware_configured(self):
        from api.main import app
        middlewares = [m.cls for m in app.user_middleware]
        import fastapi.middleware.cors
        assert fastapi.middleware.cors.CORSMiddleware in middlewares
