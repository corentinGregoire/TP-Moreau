import os
import json
import pandas as pd
import pickle as pkl

from app import MaladieData
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from fastapi import Request, Depends, APIRouter

templates = Jinja2Templates(directory="templates/")
router = APIRouter()

with open(os.path.join("app", "models", "label_encoder.pkl"), 'rb') as f:
    le = pkl.load(f, encoding='utf-8')

with open(os.path.join("app", "models", "model.pkl"), 'rb') as f:
    model = pkl.load(f, encoding='utf-8')


def is_sick(data: MaladieData) -> str:
    """Returns the language for a given audio.

    :param data: The given data that concerns the
    :return: Res dict
    """
    data_dict = json.loads(data.json())
    df = pd.DataFrame.from_dict(data_dict, orient='index').T

    # Create slice for the age column
    ages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df['age'] = pd.cut(df['age'], bins=ages, labels=False)

    # Create slice for the height column
    tailles = [0, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250]
    df['taille'] = pd.cut(df['taille'], bins=tailles, labels=False)

    # Create slice for the weight column
    poids = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
    df['poids'] = pd.cut(df['poids'], bins=poids, labels=False)

    # Encode the cholesterol column which is a string
    df['cholesterol'] = le.fit_transform(df['cholesterol'])

    return "Malade" if bool(model.predict(df)) else "Pas malade"


@router.post("/maladie", response_class=HTMLResponse)
async def post_maladie(request: Request, data: MaladieData = Depends(MaladieData.as_form)):
    """Checks with a given json if a human is sick or not.

    :param request: Request object checked by the auth_required decorator
    :param data: Given human data
    :return: Template response containing the API response (sick or not)
    """
    return templates.TemplateResponse(
        name='index.html',
        context={"request": request, "result": is_sick(data)}
    )
