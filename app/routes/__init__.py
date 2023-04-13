from app import MaladieData
from app.controllers.maladie import is_sick
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request, Depends, APIRouter

templates = Jinja2Templates(directory="templates/")
router = APIRouter()


@router.post("/maladie", response_class=HTMLResponse)
async def post_maladie(request: Request, data: MaladieData = Depends(MaladieData.as_form)):
    """Checks with a given json if a human is sick or not.

    :param request: Request object checked by the auth_required decorator
    :param data: Given human data
    :return: JSON response which contains the language detected in the input audio
    """
    return templates.TemplateResponse(
        name='index.html',
        context={"request": request, "result": is_sick(data)}
    )
