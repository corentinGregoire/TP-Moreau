from app.routes import router, templates
from fastapi import FastAPI, Request
from starlette.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# App configuration
app = FastAPI(
    title="Maladie detector",
    version="1.0",
    description="Maladie detector API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def get_homepage(request: Request):
    return templates.TemplateResponse(
        name='index.html',
        context={"request": request}
    )


